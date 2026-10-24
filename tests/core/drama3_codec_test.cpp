// tests/core/drama3_codec_test.cpp - Drama3Codec against fish-speech's own decode of the same frames.
//
//     drama3_codec_test <codec dir> <codes.i32> <reference pcm.f32> [chunk frames]
//
// codes.i32: int32 [n][10]; pcm.f32: the reference's float32 samples (n * 2048), decoded in one call.  The engine
// decodes the frames as a stream of chunks (default 7 frames, so the carried context is exercised at every layer)
// and must match the one-call reference: signal-to-error ratio >= 40 dB, the worst sample within 0.01.
#include "guild/core/drama3_codec.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

template <typename T> static std::vector<T> read_all(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f)
        return {};
    std::vector<T> v((size_t) f.tellg() / sizeof(T));
    f.seekg(0);
    f.read(reinterpret_cast<char*>(v.data()), (std::streamsize)(v.size() * sizeof(T)));
    return v;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: drama3_codec_test <codec dir> <codes.i32> <pcm.f32> [chunk]\n");
        return 2;
    }
    const auto codes = read_all<int32_t>(argv[2]);
    const auto ref = read_all<float>(argv[3]);
    if (codes.empty() || ref.empty()) {
        std::printf("SKIP: no reference files\n");
        return 77;
    }
    const int chunk = argc > 4 ? std::atoi(argv[4]) : 7;
    guild::core::Drama3Codec codec;
    std::string err;
    if (!codec.load(argv[1], 32, err)) {
        std::fprintf(stderr, "load: %s\n", err.c_str());
        return 1;
    }
    const int n = (int) (codes.size() / 10), F = codec.frame_samples();
    if (ref.size() != (size_t) n * F) {
        std::fprintf(stderr, "reference has %zu samples, not %d frames x %d\n", ref.size(), n, F);
        return 1;
    }
    std::vector<float> pcm((size_t) n * F);
    codec.reset();
    for (int f0 = 0; f0 < n; f0 += chunk) {
        const int c = std::min(chunk, n - f0);
        if (!codec.decode(codes.data() + (size_t) f0 * 10, c, pcm.data() + (size_t) f0 * F, err)) {
            std::fprintf(stderr, "decode: %s\n", err.c_str());
            return 1;
        }
    }
    double sig = 0, noise = 0, worst = 0;
    for (size_t i = 0; i < pcm.size(); ++i) {
        const double e = (double) pcm[i] - ref[i];
        sig += (double) ref[i] * ref[i];
        noise += e * e;
        worst = std::max(worst, std::fabs(e));
    }
    const double snr = 10.0 * std::log10(sig / std::max(noise, 1e-30));
    std::printf("%d frames in chunks of %d: SNR %.1f dB, worst sample error %.5f\n", n, chunk, snr, worst);
    const bool pass = snr >= 40.0 && worst <= 0.01;
    std::printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
