// tests/core/qwen_feat_dump.cpp - Qwen3.8-Flash-Next's own greedy continuation of a prompt with every position's
// context features for a DFlash drafter (its target layers' residual streams as each completes, averaged over the
// hyper-connection streams, as the verifier computes them for core/dflash.hpp): the drafter is then evaluated offline
// on the target's own greedy text.
//
//   qwen_feat_dump MODEL PACK PROFILE IDS N_NEW OUT DRAFTER
//
// IDS: the prompt's token ids (a file, separated by commas, spaces or newlines).  The prompt runs through the
// verifier four tokens at a time from position 0, then N_NEW tokens are decoded greedily one a window.  OUT.tok gets
// the int32 ids (prompt, then the generated ones) and OUT.feat each position's (target layers) x n_embd float32
// features.  DRAFTER: the DFlash checkpoint (GUILD_DFLASH, set here before the engine starts).  QWEN_DUMP_DENSE and
// QWEN_DUMP_HC_Q8 give the server's --dense and --hc q8_0.
#include "../check.hpp"
#include "../../src/runtime/model_impl.hpp"
#include "guild/server/native_engine.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace guild;

int main(int argc, char** argv) {
    if (argc != 8) {
        std::cerr << "usage: qwen_feat_dump MODEL PACK PROFILE IDS N_NEW OUT DRAFTER\n";
        return 2;
    }
    setenv("GUILD_DFLASH", argv[7], 1);
    std::vector<int32_t> ids;
    {
        std::ifstream f(argv[4]);
        CHECK(f.is_open());
        std::string s((std::istreambuf_iterator<char>(f)), {});
        for (char& c : s)
            if (c == ',')
                c = ' ';
        std::istringstream is(s);
        for (long v; is >> v;)
            ids.push_back((int32_t) v);
    }
    const int n_new = std::atoi(argv[5]);
    CHECK(!ids.empty() && n_new > 0);

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
    if (const char* d = std::getenv("QWEN_DUMP_DENSE")) // the server's --dense: 14 q6_k, 13 q5_k, 12 q4_k
        opts.plan.dense_type = std::atoi(d);
    if (std::getenv("QWEN_DUMP_HC_Q8") != nullptr) // --hc q8_0
        opts.plan.hc_q8 = true;
    server::NativeInferenceEngine engine(opts);
    std::string err;
    if (!engine.init(err)) {
        std::cerr << err << '\n';
        return 1;
    }
    auto* m = engine.model()->impl();
    const int64_t N = m->g.n_embd;
    CHECK(m->dflash != nullptr && m->ver->features() != nullptr);
    const size_t F = (size_t) m->ver->feature_count() * (size_t) N;
    m->ver->set_sampling(kernels::SamplerParams{});
    m->ver->set_history(nullptr, 0);

    std::vector<int32_t> seq = ids;
    std::vector<float> feats; // [position][target layers][N]
    auto keep = [&](int t) {
        const size_t at = feats.size();
        feats.resize(at + F);
        CHECK(cudaMemcpy(feats.data() + at, m->ver->features() + (size_t) t * F, F * sizeof(float),
                         cudaMemcpyDeviceToHost) == cudaSuccess);
    };
    int32_t next = 0;
    for (int pos = 0; pos < (int) ids.size();) {
        const int T = std::min(4, (int) ids.size() - pos);
        m->drive.d.layers = 0;
        m->drive.d.experts = 0;
        int32_t out[4] = {0, 0, 0, 0};
        if (!m->ver->run(T, ids.data() + pos, pos, runtime::drive_pool_multi, &m->drive, out, err) ||
            !m->ver->commit(T, err)) {
            std::cerr << err << '\n';
            return 1;
        }
        for (int t = 0; t < T; ++t)
            keep(t);
        next = out[T - 1];
        pos += T;
    }
    for (int i = 0; i < n_new; ++i) {
        const int64_t pos = (int64_t) seq.size();
        seq.push_back(next);
        m->drive.d.layers = 0;
        m->drive.d.experts = 0;
        int32_t out[4] = {0, 0, 0, 0};
        if (!m->ver->run(1, &seq.back(), pos, runtime::drive_pool_multi, &m->drive, out, err) ||
            !m->ver->commit(1, err)) {
            std::cerr << err << '\n';
            return 1;
        }
        keep(0);
        next = out[0];
    }
    {
        std::ofstream of(std::string(argv[6]) + ".tok", std::ios::binary);
        of.write((const char*) seq.data(), (std::streamsize)(seq.size() * sizeof(int32_t)));
        CHECK(bool(of));
        std::ofstream ff(std::string(argv[6]) + ".feat", std::ios::binary);
        ff.write((const char*) feats.data(), (std::streamsize)(feats.size() * sizeof(float)));
        CHECK(bool(ff));
    }
    std::fprintf(stderr, "qwen_feat_dump: %zu prompt + %d generated tokens, features of %zu positions\n", ids.size(),
                 n_new, feats.size() / F);
    return 0;
}
