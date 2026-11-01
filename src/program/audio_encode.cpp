// src/program/audio_encode.cpp - Nemotron 3.5 Transcribe's audio front end and audio model on WAV files, every
// stage written out for comparison with the reference (tools/nemotron_audio_parity.py).
//
//   audio_encode --audio <runtime dir> --out <dir> [--repeat N] [--keep-wide] a.wav b.wav ...
//
// Per file <name>.wav: <name>.mel.f32 (frames x n_mels, the log-mel before normalization, padded frames included),
// <name>.asr.f32 (tokens x 1280, the speech encoder), <name>.spk.f32 (tokens * 8 x 8, the speaker logits),
// <name>.audio.f32 (tokens x 1280, with the speaker term) and <name>.emb.f32 (tokens x 2688, the embeddings), and a
// line of timings on stdout.  Audio at another rate is resampled first.  --keep-wide: the float32 matrices kept on the
// device (AudioEncoder::keep_wide).
#include "guild/audio/encoder.hpp"
#include "guild/audio/frontend.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

bool write_f32(const std::string& path, const std::vector<float>& v) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(v.data()), std::streamsize(v.size() * sizeof(float)));
    return bool(f);
}

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

} // namespace

int main(int argc, char** argv) {
    std::string dir, out;
    int repeat = 1;
    bool keep_wide = false;
    std::vector<std::string> wavs;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--audio" && i + 1 < argc)
            dir = argv[++i];
        else if (a == "--out" && i + 1 < argc)
            out = argv[++i];
        else if (a == "--repeat" && i + 1 < argc)
            repeat = std::max(1, std::atoi(argv[++i]));
        else if (a == "--keep-wide")
            keep_wide = true;
        else if (a.rfind("--", 0) == 0) {
            std::fprintf(
                stderr,
                "usage: audio_encode --audio <runtime dir> --out <dir> [--repeat N] [--keep-wide] file.wav ...\n");
            return 2;
        } else
            wavs.push_back(a);
    }
    if (dir.empty() || out.empty() || wavs.empty()) {
        std::fprintf(stderr,
                     "usage: audio_encode --audio <runtime dir> --out <dir> [--repeat N] [--keep-wide] file.wav ...\n");
        return 2;
    }
    std::filesystem::create_directories(out);
    guild::audio::AudioEncoder enc;
    std::string err;
    auto t0 = std::chrono::steady_clock::now();
    if (!enc.load(dir, err)) {
        std::fprintf(stderr, "audio_encode: %s\n", err.c_str());
        return 1;
    }
    if (keep_wide && !enc.keep_wide(0))
        std::fprintf(stderr, "audio_encode: no room for the float32 matrices; widening a layer at a time\n");
    std::printf("audio_encode: loaded in %.0f ms, %.0f MiB on the GPU\n", ms_since(t0),
                double(enc.vram_bytes()) / 1048576.0);
    const auto& cfg = enc.mel_config();
    for (const std::string& path : wavs) {
        const std::string name = std::filesystem::path(path).stem().string();
        for (int r = 0; r < repeat; ++r) {
            guild::audio::Wave w;
            t0 = std::chrono::steady_clock::now();
            if (!guild::audio::read_wav(path, w, err)) {
                std::fprintf(stderr, "audio_encode: %s\n", err.c_str());
                return 1;
            }
            if (w.rate != cfg.rate)
                w.samples = guild::audio::resample(w.samples, w.rate, cfg.rate);
            guild::audio::Features f;
            if (!guild::audio::log_mel(w.samples.data(), int64_t(w.samples.size()), cfg, f, err)) {
                std::fprintf(stderr, "audio_encode: %s\n", err.c_str());
                return 1;
            }
            const double fe_ms = ms_since(t0);
            std::vector<float> emb;
            guild::audio::AudioDetail d;
            if (!enc.encode(f, emb, err, &d)) {
                std::fprintf(stderr, "audio_encode: %s: %s\n", name.c_str(), err.c_str());
                return 1;
            }
            std::printf("%s: %.2f s at %d Hz, %d frames, %d tokens; front end %.1f ms, audio model %.1f ms\n",
                        name.c_str(), double(w.samples.size()) / cfg.rate, w.rate, f.frames, f.padded / cfg.stack,
                        fe_ms, enc.last_ms());
            if (r == 0 &&
                !(write_f32(out + "/" + name + ".mel.f32", f.mel) && write_f32(out + "/" + name + ".asr.f32", d.asr) &&
                  write_f32(out + "/" + name + ".spk.f32", d.speaker_logits) &&
                  write_f32(out + "/" + name + ".audio.f32", d.audio) &&
                  write_f32(out + "/" + name + ".emb.f32", emb))) {
                std::fprintf(stderr, "audio_encode: cannot write to %s\n", out.c_str());
                return 1;
            }
        }
    }
    return 0;
}
