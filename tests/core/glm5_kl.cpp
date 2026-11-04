// tests/core/glm5_kl.cpp - GLM-5.3-Flash's next-token distributions under teacher forcing, to measure what a change
// of the weights' numbers (GUILD_GLM_DENSE=q5_k, ...) does to the model; tests/core/qwen_kl.cpp's measure for
// GLM-5.3-Flash.
//
//   glm5_kl GGUF TOKENIZER_DIR TEXT OUT [BASE]
//
// TEXT (a UTF-8 file) is tokenized (its first 4096 tokens); the first token runs alone (forward), then the rest goes
// through verify windows of two tokens, as decode runs a window whose draft is accepted: both rows' logits are read
// and both tokens kept.  So position i's distribution is the model's prediction of token i + 1 given tokens 0 .. i,
// from the kernels and the placement decode uses.  OUT gets each position's log-softmax as FP16 (an int32 count, an
// int32 vocabulary size, then the rows).  With BASE (an earlier OUT of the same text) it prints, against BASE:
//   top-1 agreement, KL(BASE || this) mean / median / 99th percentile / max, and both perplexities of the text,
// as llama.cpp's perplexity --kl-divergence does.  GUILD_GLM_DENSE (auto, q8_0, q6_k, q5_k, q4_k) picks the
// attention matrices' format; GUILD_GLM_KL_CTX the context (8192 tokens).
#include "guild/core/glm5.hpp"
#include "guild/runtime/tokenizer.hpp"

#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 5 && argc != 6) {
        std::cerr << "usage: glm5_kl GGUF TOKENIZER_DIR TEXT OUT [BASE]\n";
        return 2;
    }
    std::ifstream tf(argv[3]);
    if (!tf.is_open()) {
        std::cerr << "glm5_kl: cannot read " << argv[3] << '\n';
        return 1;
    }
    const std::string text((std::istreambuf_iterator<char>(tf)), {});
    std::string err;
    guild::runtime::Tokenizer tok;
    std::vector<int32_t> ids;
    if (!tok.load(argv[2], err) || !tok.encode(text, ids, false, err)) {
        std::cerr << "glm5_kl: " << err << '\n';
        return 1;
    }

    guild::core::Glm5Options opt;
    opt.max_context = 8192;
    if (const char* c = std::getenv("GUILD_GLM_KL_CTX"))
        opt.max_context = std::atoll(c);
    if (const char* d = std::getenv("GUILD_GLM_DENSE")) {
        const std::string s = d;
        opt.dense = s == "q8_0"   ? guild::core::Glm5Dense::Q8_0
                    : s == "q6_k" ? guild::core::Glm5Dense::Q6_K
                    : s == "q5_k" ? guild::core::Glm5Dense::Q5_K
                    : s == "q4_k" ? guild::core::Glm5Dense::Q4_K
                                  : guild::core::Glm5Dense::Auto;
    }
    guild::core::Glm5Engine eng;
    if (!eng.load(argv[1], opt, err)) {
        std::cerr << "glm5_kl: " << err << '\n';
        return 1;
    }
    std::fprintf(stderr, "guild %s", eng.placement().c_str());
    if (!eng.has_mtp()) {
        std::cerr << "glm5_kl: verify windows of two tokens need the NextN block\n";
        return 1;
    }
    const int64_t V = eng.geometry().n_vocab;
    const int n = (int) std::min<size_t>(ids.size(), 4096);
    if (n < 2) {
        std::cerr << "glm5_kl: the text is shorter than two tokens\n";
        return 1;
    }
    std::fprintf(stderr, "glm5_kl: %d tokens, vocabulary %lld\n", n, (long long) V);

    std::vector<__half> out((size_t) (n - 1) * V);
    std::vector<float> rows((size_t) 2 * V);
    std::vector<double> nll(n - 1);
    // position pos's log-softmax from its logits, and the text's next token's negative log-likelihood
    auto keep = [&](int pos, const float* row) {
        double mx = -1e300;
        for (int64_t k = 0; k < V; ++k)
            mx = std::max(mx, (double) row[k]);
        double s = 0;
        for (int64_t k = 0; k < V; ++k)
            s += std::exp((double) row[k] - mx);
        const double lse = mx + std::log(s);
        __half* o = out.data() + (size_t) pos * V;
        for (int64_t k = 0; k < V; ++k)
            o[k] = __float2half((float) ((double) row[k] - lse));
        nll[(size_t) pos] = lse - (double) row[ids[(size_t) pos + 1]];
    };
    if (!eng.forward(ids.data(), 1, 0, rows.data(), err)) {
        std::cerr << "glm5_kl: " << err << '\n';
        return 1;
    }
    keep(0, rows.data());
    for (int pos = 1; pos + 1 < n;) {
        const int T = std::min(2, n - 1 - pos);
        if (!eng.verify(ids.data() + pos, T, pos, rows.data(), err) || !eng.accept(T, err)) {
            std::cerr << "glm5_kl: " << err << '\n';
            return 1;
        }
        for (int t = 0; t < T; ++t)
            keep(pos + t, rows.data() + (size_t) t * V);
        pos += T;
    }
    double s = 0;
    for (double v : nll)
        s += v;
    std::printf("perplexity of the text: %.4f (%d predictions)\n", std::exp(s / (n - 1)), n - 1);
    {
        std::ofstream of(argv[4], std::ios::binary);
        const int32_t hdr[2] = {n - 1, (int32_t) V};
        of.write((const char*) hdr, sizeof hdr);
        of.write((const char*) out.data(), (std::streamsize)(out.size() * sizeof(__half)));
        if (!of) {
            std::cerr << "glm5_kl: cannot write " << argv[4] << '\n';
            return 1;
        }
    }
    if (argc == 6) {
        std::ifstream bf(argv[5], std::ios::binary);
        int32_t hdr[2] = {0, 0};
        bf.read((char*) hdr, sizeof hdr);
        if (!bf || hdr[0] != n - 1 || hdr[1] != (int32_t) V) {
            std::cerr << "glm5_kl: " << argv[5] << " is not an OUT of this text\n";
            return 1;
        }
        std::vector<__half> base((size_t) V);
        std::vector<double> kl;
        int same = 0;
        double base_nll = 0;
        for (int i = 0; i < n - 1; ++i) {
            bf.read((char*) base.data(), (std::streamsize)(V * sizeof(__half)));
            if (!bf) {
                std::cerr << "glm5_kl: " << argv[5] << " is short\n";
                return 1;
            }
            const __half* o = out.data() + (size_t) i * V;
            double k = 0;
            int64_t ab = 0, ao = 0;
            float bb = -1e30f, bo = -1e30f;
            for (int64_t j = 0; j < V; ++j) {
                const float lb = __half2float(base[(size_t) j]), lo = __half2float(o[j]);
                if (lb > bb) {
                    bb = lb;
                    ab = j;
                }
                if (lo > bo) {
                    bo = lo;
                    ao = j;
                }
                if (lb > -30.f)
                    k += std::exp((double) lb) * ((double) lb - (double) lo);
            }
            kl.push_back(std::max(0.0, k));
            same += ab == ao;
            base_nll -= __half2float(base[(size_t) ids[(size_t) (i + 1)]]);
        }
        std::vector<double> sorted = kl;
        std::sort(sorted.begin(), sorted.end());
        double mean = 0;
        for (double v : kl)
            mean += v;
        mean /= (double) kl.size();
        std::printf("against %s: top-1 the same %d of %d (%.2f%%); KL mean %.6f, median %.6f, 99th percentile %.6f, "
                    "max %.6f; perplexity %.4f (base) -> %.4f\n",
                    argv[5], same, n - 1, 100.0 * same / (n - 1), mean, sorted[sorted.size() / 2],
                    sorted[(size_t) (0.99 * (double) (sorted.size() - 1))], sorted.back(), std::exp(base_nll / (n - 1)),
                    std::exp(s / (n - 1)));
    }
    return 0;
}
