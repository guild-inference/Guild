// tests/core/audio_frontend_test.cpp - the audio front end (include/guild/audio/frontend.hpp) on synthetic signals:
// a WAV round trip, the resampler against analytic sines, the log-mel against a direct DFT, the normalization.
#include "guild/audio/frontend.hpp"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, const char* what, double got = 0, double bound = 0) {
    std::printf("%s %s (%.3g, bound %.3g)\n", ok ? "ok  " : "FAIL", what, got, bound);
    failures += ok ? 0 : 1;
}

std::vector<float> sine(int rate, double hz, double seconds, double amp = 0.5) {
    std::vector<float> x(size_t(rate * seconds));
    for (size_t i = 0; i < x.size(); ++i)
        x[i] = float(amp * std::sin(2.0 * M_PI * hz * double(i) / rate));
    return x;
}

} // namespace

int main() {
    using namespace guild::audio;
    std::string err;
    // WAV: 16-bit PCM round trip
    const auto tmp = (std::filesystem::temp_directory_path() / "guild_audio_frontend_test.wav").string();
    std::vector<float> s = sine(16000, 440.0, 0.25);
    Wave w;
    const bool wrote = write_wav_pcm16(tmp, s, 16000, err) && read_wav(tmp, w, err);
    double d = 0;
    for (size_t i = 0; wrote && i < s.size(); ++i)
        d = std::max(d, std::fabs(double(w.samples[i]) - std::round(double(s[i]) * 32768.0) / 32768.0));
    check(wrote && w.rate == 16000 && w.samples.size() == s.size() && d == 0.0, "wav pcm16 round trip", d, 0);
    std::filesystem::remove(tmp);

    // resampling: sines below the passband edge come out as the same sines at the new rate
    for (int from : {8000, 22050, 44100, 48000}) {
        for (double hz : {300.0, 2500.0}) {
            const std::vector<float> x = sine(from, hz, 1.0);
            const std::vector<float> y = resample(x, from, 16000);
            const size_t want = size_t((int64_t(x.size()) * 16000 + from - 1) / from);
            double e = 0;
            for (size_t i = 800; i + 800 < y.size(); ++i) // away from the edges
                e = std::max(e, std::fabs(double(y[i]) - 0.5 * std::sin(2.0 * M_PI * hz * double(i) / 16000.0)));
            char what[96];
            std::snprintf(what, sizeof what, "resample %d -> 16000, %.0f Hz sine", from, hz);
            check(y.size() == want && e < 2e-5, what, e, 2e-5);
        }
    }
    // a tone above the new Nyquist is removed
    {
        const std::vector<float> y = resample(sine(48000, 9500.0, 1.0), 48000, 16000);
        double e = 0;
        for (size_t i = 800; i + 800 < y.size(); ++i)
            e = std::max(e, std::fabs(double(y[i])));
        check(e < 1e-5, "resample 48000 -> 16000 stops a 9.5 kHz tone", e, 1e-5);
    }

    // log-mel against a direct DFT (double) of the same frames
    MelConfig cfg;
    cfg.window.resize(size_t(cfg.win));
    for (int i = 0; i < cfg.win; ++i)
        cfg.window[size_t(i)] = float(0.5 - 0.5 * std::cos(2.0 * M_PI * i / (cfg.win - 1)));
    const int bins = cfg.n_fft / 2 + 1;
    cfg.filters.assign(size_t(cfg.n_mels) * size_t(bins), 0.f);
    for (int m = 0; m < cfg.n_mels; ++m) // simple triangles, enough to check the arithmetic
        for (int k = 0; k < bins; ++k)
            cfg.filters[size_t(m) * size_t(bins) + size_t(k)] =
                float(std::max(0.0, 1.0 - std::fabs(k - 2.0 * m) / 3.0));
    std::vector<float> x = sine(16000, 1000.0, 0.5);
    for (size_t i = 0; i < x.size(); ++i)
        x[i] += float(0.1 * std::sin(0.37 * double(i * i % 1013)));
    Features f;
    const bool ok = log_mel(x.data(), int64_t(x.size()), cfg, f, err);
    check(ok && f.frames == int(x.size() / 160) && f.padded % 8 == 0 && f.padded >= f.frames, "log_mel frame counts",
          f.frames, double(x.size() / 160));
    double worst = 0;
    std::vector<double> e(x.size());
    for (size_t i = 0; i < x.size(); ++i)
        e[i] = i == 0 ? x[0] : double(x[i] - cfg.preemphasis * x[i - 1]);
    for (int t = 0; ok && t < f.frames; t += 7) {
        std::vector<double> p((size_t) bins);
        for (int k = 0; k < bins; ++k) {
            double re = 0, im = 0;
            for (int n = 0; n < cfg.n_fft; ++n) {
                const int64_t si = int64_t(t) * cfg.hop + n - cfg.n_fft / 2;
                const int wi = n - (cfg.n_fft - cfg.win) / 2;
                if (si < 0 || si >= int64_t(x.size()) || wi < 0 || wi >= cfg.win)
                    continue;
                const double v = e[size_t(si)] * cfg.window[size_t(wi)];
                re += v * std::cos(2.0 * M_PI * k * n / cfg.n_fft);
                im -= v * std::sin(2.0 * M_PI * k * n / cfg.n_fft);
            }
            p[size_t(k)] = re * re + im * im;
        }
        for (int m = 0; m < cfg.n_mels; ++m) {
            double acc = 0;
            for (int k = 0; k < bins; ++k)
                acc += cfg.filters[size_t(m) * size_t(bins) + size_t(k)] * p[size_t(k)];
            worst = std::max(worst,
                             std::fabs(std::log(acc + cfg.guard) - f.mel[size_t(t) * size_t(cfg.n_mels) + size_t(m)]));
        }
    }
    check(ok && worst < 2e-4, "log_mel against a direct DFT (max |d log|)", worst, 2e-4);
    bool tail_zero = true;
    for (size_t i = size_t(f.frames) * size_t(cfg.n_mels); i < f.mel.size(); ++i)
        tail_zero = tail_zero && f.mel[i] == 0.f;
    check(tail_zero, "log_mel zeroes the padded frames");

    // normalization: zero mean, unit unbiased deviation per bin over the valid frames
    const std::vector<float> nf = normalize_per_feature(f);
    double worst_mean = 0, worst_sd = 0;
    for (int m = 0; m < cfg.n_mels; ++m) {
        double s1 = 0, s2 = 0;
        for (int t = 0; t < f.frames; ++t)
            s1 += nf[size_t(t) * size_t(cfg.n_mels) + size_t(m)];
        for (int t = 0; t < f.frames; ++t)
            s2 += std::pow(nf[size_t(t) * size_t(cfg.n_mels) + size_t(m)] - s1 / f.frames, 2);
        worst_mean = std::max(worst_mean, std::fabs(s1 / f.frames));
        worst_sd = std::max(worst_sd, std::fabs(std::sqrt(s2 / (f.frames - 1)) - 1.0));
    }
    check(worst_mean < 1e-5 && worst_sd < 1e-3, "normalize_per_feature", std::max(worst_mean, worst_sd), 1e-3);
    std::printf("%s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
