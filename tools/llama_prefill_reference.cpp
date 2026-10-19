// Optional development oracle, linked to the user's installed llama.cpp.
// Identical numeric IDs, greedy sampling, no template/tokenizer or speculative path.
#include "llama.h"
#include "ggml-backend.h"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <cstdio>
#include <sys/resource.h>
using Clock = std::chrono::steady_clock;
double ms(Clock::time_point a) {
    return std::chrono::duration<double, std::milli>(Clock::now() - a).count();
}
int main(int argc, char** argv) try {
    if (argc < 8)
        throw std::invalid_argument("usage: reference MODEL IDS CONTEXT GENERATED NGL THREADS OUT_PREFIX [--runs N] "
                                    "[--batch N] [--aligned] [--host-kv]");
    int runs = 3, batch = 512;
    bool aligned = false, host_kv = false, cpu_experts = false;
    for (int i = 8; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--aligned")
            aligned = true;
        else if (a == "--host-kv")
            host_kv = true;
        else if (a == "--cpu-experts")
            cpu_experts = true;
        else if ((a == "--runs" || a == "--batch") && i + 1 < argc) {
            int v = std::stoi(argv[++i]);
            if (a == "--runs")
                runs = v;
            else
                batch = v;
        } else
            throw std::invalid_argument("unknown/missing option: " + a);
    }
    if (runs < 1 || batch < 1)
        throw std::invalid_argument("runs and batch must be positive");
    std::ifstream f(argv[2]);
    std::vector<llama_token> ids;
    int id;
    while (f >> id)
        ids.push_back(id);
    if (ids.empty())
        throw std::invalid_argument("empty IDs file");
    llama_log_set(
        [](ggml_log_level level, const char* text, void*) {
            if (level == GGML_LOG_LEVEL_ERROR)
                std::fputs(text, stderr);
        },
        nullptr);
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = std::stoi(argv[5]);
    llama_model_tensor_buft_override overrides[2]{};
    if (cpu_experts) {
        auto* device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (!device)
            throw std::runtime_error("CPU device unavailable");
        overrides[0] = {"ffn_.*_exps\\.weight", ggml_backend_dev_buffer_type(device)};
        mp.tensor_buft_overrides = overrides;
    }
    const auto load_start = Clock::now();
    auto* model = llama_model_load_from_file(argv[1], mp);
    if (!model)
        throw std::runtime_error("model load failed");
    auto cp = llama_context_default_params();
    cp.n_ctx = std::stoi(argv[3]);
    cp.n_batch = batch;
    cp.n_ubatch = batch;
    cp.n_threads = std::stoi(argv[6]);
    cp.n_threads_batch = cp.n_threads;
    cp.type_k = GGML_TYPE_F16;
    cp.type_v = GGML_TYPE_F16;
    if (mp.n_gpu_layers == 0) {
        cp.offload_kqv = false;
        cp.op_offload = false;
    }
    if (host_kv)
        cp.offload_kqv = false;
    auto* ctx = llama_init_from_model(model, cp);
    if (!ctx)
        throw std::runtime_error("context load failed");
    const double load_ms = ms(load_start);
    const int generated = std::stoi(argv[4]);
    auto* vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    std::ofstream csv(std::string(argv[7]) + ".csv");
    csv << "run,prompt,generated,prefill_ms,decode_ms,load_ms,wall_ms,host_peak_bytes,prefill_tokens,decode_evals\n";
    for (int run = -1; run < runs; ++run) {
        llama_memory_clear(llama_get_memory(ctx), true);
        llama_batch b = llama_batch_init(batch, 0, 1);
        const auto start = Clock::now();
        const size_t pp_tokens = ids.size() - int(aligned);
        for (size_t off = 0; off < pp_tokens; off += batch) {
            b.n_tokens = int(std::min<size_t>(batch, pp_tokens - off));
            for (int j = 0; j < b.n_tokens; ++j) {
                b.token[j] = ids[off + j];
                b.pos[j] = int(off + j);
                b.n_seq_id[j] = 1;
                b.seq_id[j][0] = 0;
                b.logits[j] = (off + j + 1 == ids.size());
            }
            if (llama_decode(ctx, b))
                throw std::runtime_error("prefill failed");
        }
        llama_synchronize(ctx);
        const double pp = ms(start);
        std::vector<llama_token> output;
        const auto decode_start = Clock::now();
        if (aligned) {
            b.n_tokens = 1;
            b.token[0] = ids.back();
            b.pos[0] = int(ids.size()) - 1;
            b.n_seq_id[0] = 1;
            b.seq_id[0][0] = 0;
            b.logits[0] = 1;
            if (llama_decode(ctx, b))
                throw std::runtime_error("final prompt token failed");
        }
        if (run == 0) {
            const auto* logits = llama_get_logits_ith(ctx, -1);
            std::ofstream o(std::string(argv[7]) + ".f32", std::ios::binary);
            o.write(reinterpret_cast<const char*>(logits), size_t(n_vocab) * 4);
        }
        for (int j = 0; j < generated; ++j) {
            const auto* logits = llama_get_logits_ith(ctx, -1);
            const int next = int(std::max_element(logits, logits + n_vocab) - logits);
            output.push_back(next);
            if (llama_vocab_is_eog(vocab, next) || j + 1 == generated)
                break;
            b.n_tokens = 1;
            b.token[0] = next;
            b.pos[0] = int(ids.size()) + j;
            b.n_seq_id[0] = 1;
            b.seq_id[0][0] = 0;
            b.logits[0] = 1;
            if (llama_decode(ctx, b))
                throw std::runtime_error("decode failed");
        }
        llama_synchronize(ctx);
        const double tg = ms(decode_start);
        rusage usage{};
        getrusage(RUSAGE_SELF, &usage);
        if (run >= 0) {
            csv << run << ',' << ids.size() << ',' << output.size() << ',' << pp << ',' << tg << ',' << load_ms << ','
                << ms(start) << ',' << uint64_t(usage.ru_maxrss) * 1024 << ',' << pp_tokens << ','
                << int(output.size()) - 1 + int(aligned) << '\n';
            csv.flush();
            std::ofstream o(std::string(argv[7]) + ".ids");
            for (auto t : output)
                o << t << ' ';
        }
        llama_batch_free(b);
    }
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
} catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
}
