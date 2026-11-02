// tests/core/dflash_check.cpp - the DFlash 2 drafter (src/core/dflash.cpp) on a sequence qwen_feat_dump wrote: the
// draft (block B) at every position of the target's greedy text, how many lead the text, and how often each draft
// position equals a reference's (the offline PyTorch port's drafts: int32 rows of B - 1).
//
//   dflash_check DRAFTER_DIR GGUF VOCAB PREFIX B [REF]
#include "guild/artifact/gguf_split.hpp"
#include "guild/core/dflash.hpp"
#include "guild/core/native_head.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

template <typename T> static std::vector<T> read_all(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    std::vector<char> raw((std::istreambuf_iterator<char>(f)), {});
    std::vector<T> out(raw.size() / sizeof(T));
    std::copy(raw.begin(), raw.begin() + (std::ptrdiff_t)(out.size() * sizeof(T)), (char*) out.data());
    return out;
}

int main(int argc, char** argv) {
    if (argc != 6 && argc != 7) {
        std::cerr << "usage: dflash_check DRAFTER_DIR GGUF VOCAB PREFIX B [REF]\n";
        return 2;
    }
    const int B = std::atoi(argv[5]);
    const int64_t N = 2560, V = 248320;
    std::string err;
    const auto shards = guild::gguf_split_paths(argv[2]);
    guild::core::NativeEmbed emb;
    guild::core::NativeHead head;
    if (!emb.load(shards, N, V, err) || !head.load(shards, N, V, err)) {
        std::cerr << err << '\n';
        return 1;
    }
    guild::core::DFlashDrafter d;
    if (!d.load(argv[1], head, emb, argv[3], err)) {
        std::cerr << err << '\n';
        return 1;
    }
    const std::string pre = argv[4];
    const auto seq = read_all<int32_t>(pre + ".tok");
    const auto feats = read_all<float>(pre + ".feat");
    std::ifstream idf(pre + ".ids");
    std::string ids((std::istreambuf_iterator<char>(idf)), {});
    int P = 1;
    for (char c : ids)
        P += c == ',';
    const size_t F = (size_t) d.target_layers().size() * N;
    const int n = (int) (feats.size() / F);
    std::vector<int32_t> ref;
    if (argc == 7)
        ref = read_all<int32_t>(argv[6]);
    float* df = nullptr;
    cudaMalloc((void**) &df, feats.size() * 4);
    cudaMemcpy(df, feats.data(), feats.size() * 4, cudaMemcpyHostToDevice);
    cudaStream_t st;
    cudaStreamCreate(&st);
    if (!d.add_context(df, F, P, 0, st, err)) {
        std::cerr << err << '\n';
        return 1;
    }
    const int end = std::min(n, P + 256 + B);
    std::vector<int> hist((size_t) B, 0);
    std::vector<int> same((size_t) B, 0);
    int steps = 0, refs = 0;
    std::vector<int32_t> drafts((size_t) B);
    std::vector<float> probs((size_t) B);
    for (int start = P; start + B <= end; ++start) {
        if (!d.draft(seq[(size_t) start], start, B, drafts.data(), probs.data(), st, err)) {
            std::cerr << err << '\n';
            return 1;
        }
        int a = 0;
        while (a < B - 1 && drafts[(size_t) a] == seq[(size_t) (start + 1 + a)])
            ++a;
        ++hist[(size_t) a];
        if ((size_t) (steps + 1) * (B - 1) <= ref.size()) {
            for (int i = 0; i < B - 1; ++i)
                same[(size_t) i] += drafts[(size_t) i] == ref[(size_t) steps * (B - 1) + i];
            ++refs;
        }
        ++steps;
        if (!d.add_context(df + (size_t) start * F, F, 1, start, st, err)) {
            std::cerr << err << '\n';
            return 1;
        }
    }
    double mean = 0;
    for (int a = 0; a < B; ++a)
        mean += (double) a * hist[(size_t) a];
    mean /= steps;
    std::printf("%s: block %d, %d positions, mean accepted of %d %.3f; histogram", pre.c_str(), B, steps, B - 1, mean);
    for (int a = 0; a < B; ++a)
        std::printf(" %d", hist[(size_t) a]);
    std::printf("\n");
    if (refs) {
        std::printf("the same as the reference's draft, by position:");
        for (int i = 0; i < B - 1; ++i)
            std::printf(" %.1f%%", 100.0 * same[(size_t) i] / refs);
        std::printf(" (%d positions)\n", refs);
    }
    std::printf("%.3f ms a draft, %.3f ms a context row\n", d.ms_draft / d.rounds, d.ms_context / (steps + 1));
    return 0;
}
