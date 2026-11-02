// tests/core/qwen_kl.cpp - Qwen3.8-Flash-Next's next-token distributions under teacher forcing, to measure what a
// change of the weights' numbers (GUILD_DENSE=q6_k, ...) does to the model.
//
//   qwen_kl MODEL PACK PROFILE TEXT OUT [BASE]
//
// TEXT (a UTF-8 file) is tokenized and run through the verifier four tokens at a time from position 0: every row's
// logits are read and every token is committed, so position i's distribution is the model's prediction of token
// i + 1 given the text's tokens 0 .. i.  OUT gets each position's log-softmax as FP16 (an int32 count, an int32
// vocabulary size, then the rows).  With BASE (an earlier OUT of the same text) it prints, against BASE:
//   top-1 agreement, KL(BASE || this) mean / median / 99th percentile / max, and both perplexities of the text,
// as llama.cpp's perplexity --kl-divergence does.  The experts run on the CPU (no GPU expert cache).
#include "../check.hpp"
#include "../../src/runtime/model_impl.hpp"
#include "guild/server/native_engine.hpp"

#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace guild;

int main(int argc, char** argv) {
    if (argc != 6 && argc != 7) {
        std::cerr << "usage: qwen_kl MODEL PACK PROFILE TEXT OUT [BASE]\n";
        return 2;
    }
    std::ifstream tf(argv[4]);
    CHECK(tf.is_open());
    const std::string text((std::istreambuf_iterator<char>(tf)), {});

    server::NativeInferenceEngineOptions opts;
    opts.paths.primary_model_path = argv[1];
    opts.paths.pack_dir = argv[2];
    opts.paths.tokenizer_dir = std::string(argv[2]) + "/tokenizer";
    opts.paths.expert_profile_path = argv[3];
    opts.plan.context_length = 8192;
    opts.plan.mtp_spec_tokens = 4; // windows of up to four rows
    opts.plan.kv_mode = memory::KvMode::HostOnly;
    opts.plan.prefill_chunk = 512;
    opts.plan.pcie_expert_fraction = 0;
    opts.plan.routed_experts_in_gpu = 0;
    opts.plan.expert_cache_auto = false;
    server::NativeInferenceEngine engine(opts);
    std::string err;
    if (!engine.init(err)) {
        std::cerr << err << '\n';
        return 1;
    }
    std::vector<int32_t> ids;
    if (!engine.encode(text, ids, err)) {
        std::cerr << err << '\n';
        return 1;
    }
    auto* m = engine.model()->impl();
    const int64_t V = m->ver->vocab();
    const int n = (int) std::min<size_t>(ids.size(), 4096);
    CHECK(n >= 2);
    std::fprintf(stderr, "qwen_kl: %d tokens, vocabulary %lld\n", n, (long long) V);

    std::vector<__half> out((size_t) (n - 1) * V);
    std::vector<float> row((size_t) V);
    std::vector<double> nll(n - 1);
    m->ver->set_sampling(kernels::SamplerParams{});
    m->ver->set_history(nullptr, 0);
    for (int pos = 0; pos + 1 < n;) {
        const int T = std::min(4, n - 1 - pos);
        m->drive.d.layers = 0;
        m->drive.d.experts = 0;
        int32_t argmax[4] = {0, 0, 0, 0};
        if (!m->ver->run(T, ids.data() + pos, pos, runtime::drive_pool_multi, &m->drive, argmax, err)) {
            std::cerr << err << '\n';
            return 1;
        }
        for (int t = 0; t < T; ++t) {
            CHECK(m->ver->copy_logits(t, row.data()));
            double mx = -1e300;
            for (float v : row)
                mx = std::max(mx, (double) v);
            double s = 0;
            for (float v : row)
                s += std::exp((double) v - mx);
            const double lse = mx + std::log(s);
            __half* o = out.data() + (size_t) (pos + t) * V;
            for (int64_t k = 0; k < V; ++k)
                o[k] = __float2half((float) ((double) row[(size_t) k] - lse));
            nll[(size_t) (pos + t)] = lse - (double) row[(size_t) ids[(size_t) (pos + t + 1)]];
        }
        if (!m->ver->commit(T, err)) {
            std::cerr << err << '\n';
            return 1;
        }
        pos += T;
    }
    double s = 0;
    for (double v : nll)
        s += v;
    std::printf("perplexity of the text: %.4f (%d predictions)\n", std::exp(s / (n - 1)), n - 1);
    {
        std::ofstream of(argv[5], std::ios::binary);
        const int32_t hdr[2] = {n - 1, (int32_t) V};
        of.write((const char*) hdr, sizeof hdr);
        of.write((const char*) out.data(), (std::streamsize)(out.size() * sizeof(__half)));
        CHECK(bool(of));
    }
    if (argc == 7) {
        std::ifstream bf(argv[6], std::ios::binary);
        CHECK(bf.is_open());
        int32_t hdr[2];
        bf.read((char*) hdr, sizeof hdr);
        CHECK(hdr[0] == n - 1 && hdr[1] == (int32_t) V);
        std::vector<__half> base((size_t) V);
        std::vector<double> kl;
        int same = 0;
        double base_nll = 0;
        for (int i = 0; i < n - 1; ++i) {
            bf.read((char*) base.data(), (std::streamsize)(V * sizeof(__half)));
            CHECK(bool(bf));
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
                    argv[6], same, n - 1, 100.0 * same / (n - 1), mean, sorted[sorted.size() / 2],
                    sorted[(size_t) (0.99 * (double) (sorted.size() - 1))], sorted.back(), std::exp(base_nll / (n - 1)),
                    std::exp(s / (n - 1)));
    }
    return 0;
}
