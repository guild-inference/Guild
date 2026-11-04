// tests/core/drama3_codec_test.cpp - Drama3Codec against fish-speech's own decode of the same frames.
//
//     drama3_codec_test <codec dir> <codes.i32> <reference pcm.f32> [chunk frames]
//
// codes.i32: int32 [n][10]; pcm.f32: the reference's float32 samples (n * 2048), decoded in one call.  The engine
// decodes the frames as a stream of chunks (default 7 frames, so the carried context is exercised at every layer)
// and must match the one-call reference: signal-to-error ratio >= 40 dB, the worst sample within 0.01.
//
//     drama3_codec_test --encode <codec dir> <pcm.f32> <reference codes.i32>
//
// The encoder against the reference's codes of the same samples (one call there, chunks of 16 frames here): the
// share of equal codes per codebook.  A quantizer picks the nearest of 1024 or 4096 rows, so rounding flips some near
// ties, and a flipped code changes the residual every later codebook quantizes; pass: >= 99% of codebook 0's codes
// and >= 90% of all codes equal.
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

static int encode_test(int argc, char** argv) {
    if (argc < 5) {
        std::fprintf(stderr, "usage: drama3_codec_test --encode <codec dir> <pcm.f32> <codes.i32>\n");
        return 2;
    }
    const auto pcm = read_all<float>(argv[3]);
    const auto ref = read_all<int32_t>(argv[4]);
    if (pcm.empty() || ref.empty()) {
        std::printf("SKIP: no reference files\n");
        return 77;
    }
    guild::core::Drama3Codec codec;
    std::string err;
    if (!codec.load(argv[2], 16, err)) {
        std::fprintf(stderr, "load: %s\n", err.c_str());
        return 1;
    }
    std::vector<int32_t> codes;
    if (!codec.encode(pcm.data(), pcm.size(), codes, err)) {
        std::fprintf(stderr, "encode: %s\n", err.c_str());
        return 1;
    }
    if (codes.size() != ref.size()) {
        std::fprintf(stderr, "%zu codes, the reference has %zu\n", codes.size(), ref.size());
        return 1;
    }
    const size_t n = codes.size() / 10;
    int same[10] = {}, total = 0;
    for (size_t f = 0; f < n; ++f)
        for (int k = 0; k < 10; ++k) {
            same[k] += codes[f * 10 + k] == ref[f * 10 + k];
            total += codes[f * 10 + k] == ref[f * 10 + k];
        }
    std::printf("%zu frames; equal codes per codebook:", n);
    for (int k = 0; k < 10; ++k)
        std::printf(" %.3f", (double) same[k] / n);
    const double all = (double) total / (10.0 * n);
    std::printf("; all %.3f\n", all);
    const bool pass = same[0] >= 0.99 * n && all >= 0.90;
    std::printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--encode")
        return encode_test(argc, argv);
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
