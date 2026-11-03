// src/runtime/drama3.cpp - Fish Audio Drama 3 text to speech (GuildModel::synthesize).
//
// The prompt is fish_speech/models/text2semantic/fish_omni/generate.py's `build_drama3_prompt`
// (speech_template_version reference-bracket-v1): rows of [token, c0..c9], text rows with zero codes, a reference
// clip's frames as [<|semantic:c0|>, c0..c9].  Every literal is tokenized on its own, as the reference does
// (tokenizing the joined string changes ids at the seams):
//
//   <|im_start|>system\n  "convert the provided text to speech[ reference to the following:]"
//   per reference i:      "\n\n[Reference i]\nText: <|speaker:N|>transcript\nSpeech: "  + its frames
//   <|im_end|>\n  <|im_start|>user\n  "<|speaker:0|> text"  <|im_end|>\n  <|im_start|>assistant\n<|voice|>
//
// Decode is the reference's generate_drama3: the backbone's state at row p predicts frame p + 1 through the tower;
// over the prompt the tower is teacher-forced with the next row, after it each sampled frame is the next input.
#include "model_impl.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <numeric>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

namespace guild::runtime {

namespace {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

// A thread that runs one job at a time while the caller goes on: the tower's step_finish (a frame's codes 2..9) beside
// the backbone's next row, which reads only main and c1, and the codec's chunks.  The GPU has room for them: the
// backbone's step waits on its CPU experts.
class Beside {
public:
    explicit Beside(int device)
        : t_([this, device] {
              cudaSetDevice(device);
              std::unique_lock<std::mutex> lk(mu_);
              for (;;) {
                  cv_.wait(lk, [&] { return job_ || quit_; });
                  if (quit_)
                      return;
                  job_ = false;
                  auto fn = std::move(fn_);
                  lk.unlock();
                  const bool ok = fn();
                  lk.lock();
                  ok_ = ok;
                  busy_ = false;
                  cv_.notify_all();
              }
          }) {
    }
    ~Beside() {
        join();
        {
            std::lock_guard<std::mutex> lk(mu_);
            quit_ = true;
        }
        cv_.notify_all();
        t_.join();
    }
    void start(std::function<bool()> fn) {
        std::lock_guard<std::mutex> lk(mu_);
        fn_ = std::move(fn);
        job_ = busy_ = true;
        cv_.notify_all();
    }
    bool join() { // the last job's result; true when none is running
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&] { return !busy_; });
        const bool ok = ok_;
        ok_ = true;
        return ok;
    }

private:
    std::mutex mu_;
    std::condition_variable cv_;
    std::function<bool()> fn_;
    bool job_ = false, busy_ = false, quit_ = false, ok_ = true;
    std::thread t_;
};

constexpr int kWidth = 11;           // [main, c0..c9]
constexpr int kMaxPromptRows = 8192; // the reference's DRAMA3_MAX_PROMPT_ROWS
constexpr int kMaxSpeaker = 15;

// leading white space: the Unicode White_Space set, as the reference's UNICODE_WHITE_SPACE (U+0009..U+000D, space,
// U+0085, U+00A0, U+1680, U+2000..U+200A, U+2028, U+2029, U+202F, U+205F, U+3000)
size_t white_prefix(const std::string& s) {
    size_t i = 0;
    auto at = [&](size_t k) { return k < s.size() ? (unsigned char) s[k] : 0u; };
    while (i < s.size()) {
        const unsigned c = at(i);
        if (c == ' ' || (c >= 0x09 && c <= 0x0D)) {
            ++i;
        } else if (c == 0xC2 && (at(i + 1) == 0xA0 || at(i + 1) == 0x85)) {
            i += 2;
        } else if (c == 0xE1 && at(i + 1) == 0x9A && at(i + 2) == 0x80) { // U+1680
            i += 3;
        } else if (c == 0xE2 && at(i + 1) == 0x80 &&
                   ((at(i + 2) >= 0x80 && at(i + 2) <= 0x8A) || at(i + 2) == 0xA8 || at(i + 2) == 0xA9 ||
                    at(i + 2) == 0xAF)) { // U+2000..U+200A, U+2028, U+2029, U+202F
            i += 3;
        } else if (c == 0xE2 && at(i + 1) == 0x81 && at(i + 2) == 0x9F) { // U+205F
            i += 3;
        } else if (c == 0xE3 && at(i + 1) == 0x80 && at(i + 2) == 0x80) { // U+3000
            i += 3;
        } else {
            break;
        }
    }
    return i;
}

// length of a canonical <|speaker:N|> (N 0..15) at `at`, else 0
size_t speaker_tag(const std::string& s, size_t at) {
    static const std::string head = "<|speaker:";
    if (s.compare(at, head.size(), head) != 0)
        return 0;
    size_t i = at + head.size(), j = i;
    while (j < s.size() && s[j] >= '0' && s[j] <= '9')
        ++j;
    if (j == i || s.compare(j, 2, "|>") != 0)
        return 0;
    const std::string digits = s.substr(i, j - i);
    if (digits.size() > 2 || (digits.size() > 1 && digits[0] == '0') || std::stoi(digits) > kMaxSpeaker)
        return 0;
    return j + 2 - at;
}

bool validate_text(const std::string& text, const std::string& where, std::string& err) {
    if (white_prefix(text) == text.size()) {
        err = where + ": text must not be empty";
        return false;
    }
    for (const char* t : {"<|im_start|>", "<|im_end|>", "<|voice|>", "<|text|>", "<|interleave|>", "<|semantic:"}) {
        if (text.find(t) != std::string::npos) {
            err = where + ": " + t + " is written by the prompt builder";
            return false;
        }
    }
    for (size_t at = text.find("<|speaker"); at != std::string::npos; at = text.find("<|speaker", at + 1)) {
        if (speaker_tag(text, at) == 0) {
            err = where + ": speaker tags must be <|speaker:0|> to <|speaker:15|>";
            return false;
        }
    }
    return true;
}

std::string with_speaker_tag(const std::string& text) {
    return speaker_tag(text, white_prefix(text)) ? text : "<|speaker:0|> " + text;
}

// torchaudio.functional.resample's default (sinc_interp_hann, lowpass_filter_width 6, rolloff 0.99), in double:
// fish-speech resamples a reference clip this way before encoding it.
// The kernel holds (to / gcd) * taps doubles: rates sharing few factors with 44.1 kHz would take gigabytes
// (44101 Hz: 15.6 GB), so a ratio past kMaxKernel is refused.  Every usual rate is small (48 kHz: 147 x 172).
constexpr int64_t kMaxKernel = 4 << 20;
constexpr double kPi = 3.14159265358979323846;
bool resample(const float* x, size_t n, int from, int to, std::vector<float>& y, std::string& err) {
    if (from == to) {
        y.assign(x, x + n);
        return true;
    }
    const int g = std::gcd(from, to);
    const int64_t orig = from / g, nw = to / g;
    const double lpw = 6.0, base = (double) std::min(orig, nw) * 0.99;
    const int64_t width = (int64_t) std::ceil(lpw * (double) orig / base);
    const int64_t taps = 2 * width + orig;
    if (nw * taps > kMaxKernel) {
        err = "the reference audio's sample rate (" + std::to_string(from) +
              " Hz) cannot be resampled to 44.1 kHz cheaply: use 8, 11.025, 16, 22.05, 24, 32, 44.1, 48 or 96 kHz";
        return false;
    }
    std::vector<double> k((size_t) (nw * taps));
    for (int64_t j = 0; j < nw; ++j) {
        for (int64_t i = 0; i < taps; ++i) {
            double t = (-(double) j / (double) nw + (double) (i - width) / (double) orig) * base;
            t = std::clamp(t, -lpw, lpw);
            const double w = std::cos(t * kPi / lpw / 2.0);
            const double tp = t * kPi;
            k[(size_t) (j * taps + i)] = (tp == 0.0 ? 1.0 : std::sin(tp) / tp) * w * w * base / (double) orig;
        }
    }
    const size_t out_n = (size_t) std::ceil((double) nw * (double) n / (double) orig);
    y.assign(out_n, 0.0f);
    for (size_t o = 0; o < out_n; ++o) {
        const int64_t frame = (int64_t) (o / (size_t) nw), j = (int64_t) (o % (size_t) nw);
        // the input padded with `width` zeros on the left: frame f's window starts at sample f * orig - width
        const int64_t start = frame * orig - width;
        double acc = 0.0;
        for (int64_t i = 0; i < taps; ++i) {
            const int64_t at = start + i;
            if (at >= 0 && at < (int64_t) n)
                acc += k[(size_t) (j * taps + i)] * x[at];
        }
        y[o] = (float) acc;
    }
    return true;
}

} // namespace

bool GuildModel::encode_voice(const float* pcm, size_t n, int sample_rate, std::vector<int32_t>& codes,
                              std::string& error_msg) {
    Impl& m = *impl_;
    if (!m.codec) {
        error_msg = "this model was not loaded as a speech model (--drama3)";
        return false;
    }
    if (n == 0) {
        error_msg = "the reference audio is empty";
        return false;
    }
    if (sample_rate < 4000 || sample_rate > 384000) {
        error_msg = "the reference audio's sample rate (" + std::to_string(sample_rate) + " Hz) is out of range";
        return false;
    }
    std::vector<float> x;
    return resample(pcm, n, sample_rate, m.codec->sample_rate(), x, error_msg) &&
           m.codec->encode(x.data(), x.size(), codes, error_msg);
}

bool GuildModel::is_speech_model() const {
    return impl_ && impl_->tower != nullptr;
}

bool GuildModel::synthesize(const SpeechRequest& req, const SpeechCallback& on_audio, SpeechTelemetry& telemetry,
                            std::string& error_msg) {
    Impl& m = *impl_;
    if (!m.tower || !m.codec || !m.ver) {
        error_msg = "this model was not loaded as a speech model (--drama3)";
        return false;
    }
    const core::Drama3TowerGeometry& tg = m.tower->geometry();
    const int CS = tg.codebook_size;
    if (req.top_k < 1) {
        error_msg = "top_k must be at least 1";
        return false;
    }

    // ---- the prompt rows
    std::vector<int32_t> rows; // [n][kWidth]
    auto text_rows = [&](const std::string& s) -> bool {
        std::vector<int32_t> ids;
        if (!m.tokenizer->encode(s, ids, true, error_msg))
            return false;
        for (int32_t id : ids) {
            rows.push_back(id);
            rows.insert(rows.end(), kWidth - 1, 0);
        }
        return true;
    };
    if (!validate_text(req.text, "text", error_msg))
        return false;
    if (!text_rows("<|im_start|>system\n") ||
        !text_rows(req.references.empty() ? "convert the provided text to speech"
                                          : "convert the provided text to speech reference to the following:"))
        return false;
    for (size_t i = 0; i < req.references.size(); ++i) {
        const SpeechReference& r = req.references[i];
        const std::string where = "reference " + std::to_string(i + 1);
        if (!validate_text(r.text, where, error_msg))
            return false;
        const int nc = tg.n_codebooks;
        std::vector<int32_t> encoded;
        const std::vector<int32_t>* codes = &r.codes;
        if (r.codes.empty() && !r.audio.empty()) {
            if (!encode_voice(r.audio.data(), r.audio.size(), r.sample_rate, encoded, error_msg))
                return false;
            codes = &encoded;
        }
        if (codes->empty() || codes->size() % (size_t) nc) {
            error_msg = where + ": codes must be [frames >= 1][" + std::to_string(nc) + "]";
            return false;
        }
        // the tag must follow "Text: " directly; leading white space moves behind it
        const std::string tagged = with_speaker_tag(r.text);
        const size_t lead = white_prefix(tagged), tag = speaker_tag(tagged, lead);
        const std::string transcript = tagged.substr(lead, tag) + tagged.substr(0, lead) + tagged.substr(lead + tag);
        if (!text_rows("\n\n[Reference " + std::to_string(i + 1) + "]\nText: " + transcript + "\nSpeech: "))
            return false;
        const std::vector<int32_t>& cv = *codes; // the given codes, or the ones just encoded from r.audio
        for (size_t f = 0; f < cv.size(); f += (size_t) nc) {
            for (int k = 0; k < nc; ++k) {
                if (cv[f + k] < 0 || cv[f + k] >= CS) {
                    error_msg = where + ": codes must lie in [0, " + std::to_string(CS) + ")";
                    return false;
                }
            }
            rows.push_back(tg.semantic_start + cv[f]);
            rows.insert(rows.end(), cv.begin() + (std::ptrdiff_t) f, cv.begin() + (std::ptrdiff_t)(f + nc));
        }
    }
    if (!text_rows("<|im_end|>\n") || !text_rows("<|im_start|>user\n") || !text_rows(with_speaker_tag(req.text)) ||
        !text_rows("<|im_end|>\n") || !text_rows("<|im_start|>assistant\n<|voice|>"))
        return false;
    const int R = (int) (rows.size() / kWidth);
    telemetry.prompt_rows = R;
    if (const char* dump = std::getenv("GUILD_DRAMA3_DUMP_PROMPT"); dump && *dump) { // int32 [rows][11], for tests
        if (std::FILE* f = std::fopen(dump, "wb")) {
            std::fwrite(rows.data(), 4, rows.size(), f);
            std::fclose(f);
        }
    }
    if (R > kMaxPromptRows) {
        error_msg = "the prompt has " + std::to_string(R) + " rows; at most " + std::to_string(kMaxPromptRows);
        return false;
    }
    if ((int64_t) R + req.max_new_frames > m.tower->max_frames() ||
        (int64_t) R + req.max_new_frames > m.ver->attention_capacity()) {
        error_msg = "the prompt (" + std::to_string(R) + " rows) and max_new_frames do not fit the context (" +
                    std::to_string(std::min<int64_t>(m.tower->max_frames(), m.ver->attention_capacity())) + ")";
        return false;
    }

    // ---- a fresh sequence
    m.clear_prefix_cache();
    if (!m.ver->wait_commit(error_msg))
        return false;
    core::session_zero(m.ss, m.g, nullptr, m.main_stream);
    if (cudaStreamSynchronize(m.main_stream) != cudaSuccess) {
        error_msg = "failed to reset inference state";
        return false;
    }
    m.codec->reset();
    m.tower->reset_sampling();
    core::Drama3Sampling sp;
    sp.temperature = req.temperature;
    sp.top_p = req.top_p;
    sp.top_k = req.top_k;
    sp.seed = req.seed;

    std::vector<int32_t> mains; // every row's main token so far (the repetition window)
    mains.reserve((size_t) R + (size_t) req.max_new_frames);
    for (int p = 0; p < R; ++p)
        mains.push_back(rows[(size_t) p * kWidth]);
    std::vector<int32_t> frame(kWidth), pending; // pending: generated frames not yet decoded, [n][10]
    std::vector<float> pcm;
    double codec_ms = 0.0, tower_ms = 0.0;
    // The codec decodes each chunk of frames on `codec_side` while the next frames are made (GUILD_DRAMA3_OVERLAP=0:
    // here), so `on_audio` may run on that thread: one call at a time, in order, all before this returns.  codec_ms
    // is the time this thread spends on the codec (its own chunks, and waiting for the other thread's).
    const char* overlap_env = std::getenv("GUILD_DRAMA3_OVERLAP");
    const bool overlap = !(overlap_env && overlap_env[0] == '0');
    int device = 0;
    cudaGetDevice(&device);
    std::vector<int32_t> codec_frames;
    std::string codec_err;
    std::atomic<bool> codec_stopped{false}; // on_audio returned false
    auto decode = [&](bool last, std::string& err) -> bool {
        const int n = (int) (codec_frames.size() / (size_t) tg.n_codebooks);
        pcm.resize((size_t) n * (size_t) m.codec->frame_samples());
        if (n > 0 && !m.codec->decode(codec_frames.data(), n, pcm.data(), err))
            return false;
        SpeechChunk c;
        c.pcm = pcm.data();
        c.samples = pcm.size();
        c.sample_rate = m.codec->sample_rate();
        c.last = last;
        if (on_audio(c))
            return true;
        codec_stopped = true;
        return false;
    };
    std::unique_ptr<Beside> codec_side; // after what its job reads and writes, so it joins before they go
    if (overlap)
        codec_side = std::make_unique<Beside>(device);
    auto codec_join = [&]() -> bool { // false: the last chunk failed (error_msg) or on_audio stopped (no error)
        if (!codec_side)
            return true;
        const auto t0 = Clock::now();
        const bool ok = codec_side->join();
        codec_ms += ms_since(t0);
        if (!ok)
            error_msg = codec_err;
        return ok;
    };
    auto cancelled = [&] { return (req.cancel_flag && req.cancel_flag->load()) || codec_stopped; };
    auto flush = [&](bool last) -> bool {
        const int n = (int) (pending.size() / (size_t) tg.n_codebooks);
        if (n == 0 && !last)
            return true;
        if (!codec_join())
            return false;
        codec_frames.swap(pending);
        pending.clear();
        if (codec_side && !last) {
            codec_side->start([&] {
                codec_err.clear();
                return decode(false, codec_err);
            });
            return true;
        }
        const auto t0 = Clock::now();
        const bool ok = decode(last, error_msg);
        codec_ms += ms_since(t0);
        return ok;
    };
    // tower_step leaves the frame's codes 2..9 to `beside` (GUILD_DRAMA3_OVERLAP=0: none); tower_join waits for them.
    // tower_ms is the time this thread spends in the tower.  `tower_row` is the step's frame row, kept until the join.
    std::vector<int32_t> tower_row(kWidth);
    std::string beside_err;
    std::unique_ptr<Beside> beside; // after what its job reads and writes, so it joins before they go
    if (overlap)
        beside = std::make_unique<Beside>(device);
    auto tower_join = [&]() -> bool {
        if (!beside)
            return true;
        const auto t0 = Clock::now();
        const bool ok = beside->join();
        tower_ms += ms_since(t0);
        if (!ok)
            error_msg = beside_err;
        return ok;
    };
    auto tower_step = [&](const float* hidden, int64_t p, const int32_t* row, const int32_t* forced) -> bool {
        const auto t0 = Clock::now();
        std::copy(row, row + kWidth, tower_row.begin());
        const int n_hist = (int) std::min<size_t>(mains.size(), (size_t) p + 1);
        bool ok =
            m.tower->step_begin(hidden, p, tower_row.data(), forced, mains.data() + (n_hist - std::min(n_hist, 16)),
                                std::min(n_hist, 16), sp, frame.data(), error_msg);
        if (ok && beside)
            beside->start([&] { return m.tower->step_finish(beside_err); });
        else if (ok)
            ok = m.tower->step_finish(error_msg);
        tower_ms += ms_since(t0);
        return ok;
    };
    auto backbone = [&](const int32_t* tokens, const int32_t* code1, int T, int64_t p0) -> bool {
        std::vector<int32_t> out((size_t) T);
        m.ver->set_codes(code1, T);
        m.drive.d.layers = 0;
        m.drive.d.experts = 0;
        return m.ver->run(T, tokens, p0, drive_pool_multi, &m.drive, out.data(), error_msg) &&
               m.ver->commit(T, error_msg);
    };

    // Test hooks: GUILD_DRAMA3_FORCE (int32 [n][11]) teacher-forces the decode with those frames;
    // GUILD_DRAMA3_DUMP_HIDDEN writes the backbone's final-norm state of every position (float32 [rows][n_embd]).
    std::vector<int32_t> forced_frames;
    if (const char* f = std::getenv("GUILD_DRAMA3_FORCE"); f && *f) {
        if (std::FILE* in = std::fopen(f, "rb")) {
            int32_t v;
            while (std::fread(&v, 4, 1, in) == 1)
                forced_frames.push_back(v);
            std::fclose(in);
        }
    }
    std::FILE* hidden_dump = nullptr;
    if (const char* f = std::getenv("GUILD_DRAMA3_DUMP_HIDDEN"); f && *f)
        hidden_dump = std::fopen(f, "wb");
    std::vector<float> hidden_host((size_t) m.g.n_embd);
    auto dump_hidden = [&](const float* dev) {
        if (hidden_dump &&
            cudaMemcpy(hidden_host.data(), dev, hidden_host.size() * 4, cudaMemcpyDeviceToHost) == cudaSuccess)
            std::fwrite(hidden_host.data(), 4, hidden_host.size(), hidden_dump);
    };
    auto forced_at = [&](int64_t frame_index) -> const int32_t* {
        return (frame_index + 1) * kWidth <= (int64_t) forced_frames.size()
                   ? &forced_frames[(size_t) frame_index * kWidth]
                   : nullptr;
    };

    // ---- the prompt: the backbone in windows of up to `spec` rows, every row's state kept; then the tower learns
    // each row's next row in one batched pass (Drama3Tower::prefill), and the last row samples the first frame
    const auto t_prompt = Clock::now();
    const int S = std::max(1, m.config.spec);
    const size_t D = (size_t) m.g.n_embd;
    struct DeviceRows {
        float* p = nullptr;
        ~DeviceRows() {
            cudaFree(p);
        }
    } prompt_hidden;
    if (cudaMalloc(&prompt_hidden.p, (size_t) R * D * 4) != cudaSuccess) {
        cudaGetLastError();
        error_msg = "drama3: no VRAM for the prompt's " + std::to_string(R) + " hidden rows";
        return false;
    }
    std::vector<int32_t> tok((size_t) S), c1((size_t) S);
    for (int p0 = 0; p0 < R; p0 += S) {
        if (cancelled()) {
            telemetry.finish_reason = "cancel";
            return true;
        }
        const int T = std::min(S, R - p0);
        for (int t = 0; t < T; ++t) {
            tok[(size_t) t] = rows[(size_t) (p0 + t) * kWidth];
            c1[(size_t) t] = rows[(size_t) (p0 + t) * kWidth + 2];
        }
        if (!backbone(tok.data(), c1.data(), T, p0))
            return false;
        for (int t = 0; t < T; ++t) { // the head inputs are one row apart: one copy per row keeps no layout assumption
            if (cudaMemcpy(prompt_hidden.p + (size_t) (p0 + t) * D, m.ver->head_input(t), D * 4,
                           cudaMemcpyDeviceToDevice) != cudaSuccess) {
                error_msg = "drama3: copying the prompt's hidden states failed";
                return false;
            }
            dump_hidden(m.ver->head_input(t));
        }
    }
    const auto t_tower = Clock::now();
    if (R > 1 && !m.tower->prefill(prompt_hidden.p, 0, R - 1, rows.data(), error_msg))
        return false;
    tower_ms += ms_since(t_tower);
    if (!tower_step(prompt_hidden.p + (size_t) (R - 1) * D, R - 1, &rows[(size_t) (R - 1) * kWidth], forced_at(0)))
        return false;
    telemetry.prompt_ms = ms_since(t_prompt);
    // the loop's `frame` holds main, c0 and c1 at the top; the rest after tower_join

    // ---- decode: `frame` holds the frame after row R - 1
    const auto t_decode = Clock::now();
    telemetry.finish_reason = "length";
    for (int step = 0; step < req.max_new_frames; ++step) {
        if (frame[0] == tg.eos) {
            telemetry.finish_reason = "eos";
            break;
        }
        if (cancelled()) {
            telemetry.finish_reason = "cancel";
            break;
        }
        ++telemetry.frames;
        mains.push_back(frame[0]);
        const bool last = step + 1 == req.max_new_frames;
        const int64_t p = (int64_t) R + step; // the new frame's row
        const int32_t main = frame[0], code1 = frame[2];
        if (!last && !backbone(&main, &code1, 1, p)) {
            tower_join();
            return false;
        }
        if (!tower_join())
            return false;
        pending.insert(pending.end(), frame.begin() + 1, frame.end());
        if ((int) (pending.size() / (size_t) tg.n_codebooks) >= std::max(1, req.chunk_frames) && !flush(false)) {
            if (!error_msg.empty())
                return false;
            telemetry.finish_reason = "cancel";
            break;
        }
        if (last)
            break;
        const std::vector<int32_t> row = frame;
        dump_hidden(m.ver->head_input(0));
        if (!tower_step(m.ver->head_input(0), p, row.data(), forced_at(step + 1)))
            return false;
    }
    if (!tower_join())
        return false;
    const bool codec_ok = codec_join(); // the chunk on codec_side counts toward the decode, as when it ran here
    telemetry.decode_ms = ms_since(t_decode);
    if (hidden_dump)
        std::fclose(hidden_dump);
    if (!codec_ok) {
        if (!error_msg.empty())
            return false;
        telemetry.finish_reason = "cancel";
    } else if (!flush(true) && !error_msg.empty()) {
        return false;
    }
    telemetry.tower_ms = tower_ms;
    telemetry.codec_ms = codec_ms;
    return true;
}

} // namespace guild::runtime
