// tests/core/glm5_run.cpp - GLM-5.3-Flash from token ids: greedy decode, timings, and the prompt's last logits.
//
//   glm5_run <gguf shard> <comma-separated prompt ids> [new tokens] [--logits out.f32] [--threads N] [--ctx N]
//            [--step] [--spec] [--no-mtp] [--no-numa] [--no-prefault]
//   --spec: decode with the NextN block's drafts (greedy; the output equals plain greedy decoding)
//   --oracle <file>: --spec with the drafts replaced by a plain greedy output (tests the accept path)
//   --step: the prompt one token at a time, as decode runs, instead of windows of 64
//   --no-numa: the experts stay in the GGUF mapping instead of half of each on either NUMA node
//
// Prints the generated ids (one line) for a tokenizer to decode, then prompt and decode rates.
#include "guild/core/glm5.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: glm5_run <gguf> <ids> [new] [--logits f] [--threads n] [--ctx n] [--step]\n");
        return 2;
    }
    std::vector<int32_t> ids;
    {
        std::stringstream ss(argv[2]);
        std::string tok;
        while (std::getline(ss, tok, ','))
            if (!tok.empty())
                ids.push_back((int32_t) std::atol(tok.c_str()));
    }
    int n_new = 32;
    std::string logits_path;
    bool step = false, spec = false;
    std::vector<int32_t> oracle;
    guild::core::Glm5Options opt;
    for (int i = 3; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--logits" && i + 1 < argc)
            logits_path = argv[++i];
        else if (a == "--threads" && i + 1 < argc)
            opt.threads = std::atoi(argv[++i]);
        else if (a == "--ctx" && i + 1 < argc)
            opt.max_context = std::atol(argv[++i]);
        else if (a == "--step")
            step = true;
        else if (a == "--no-prefault")
            opt.prefault = false;
        else if (a == "--no-numa")
            opt.numa_split = false;
        else if (a == "--spec")
            spec = true;
        else if (a == "--no-mtp")
            opt.mtp = false;
        else if (a == "--q6k")
            opt.dense_q6k = true;
        else if (a == "--oracle" && i + 1 < argc) { // test: drafts are this greedy output's tokens (all accepted)
            std::ifstream f(argv[++i]);
            std::string line, tok;
            std::getline(f, line);
            std::stringstream ss(line);
            while (std::getline(ss, tok, ','))
                if (!tok.empty())
                    oracle.push_back((int32_t) std::atol(tok.c_str()));
            spec = true;
        } else
            n_new = std::atoi(a.c_str());
    }
    guild::core::Glm5Engine eng;
    std::string err;
    const auto t0 = std::chrono::steady_clock::now();
    if (!eng.load(argv[1], opt, err)) {
        std::fprintf(stderr, "glm5_run: %s\n", err.c_str());
        return 1;
    }
    auto since = [](std::chrono::steady_clock::time_point t) {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
    };
    std::fprintf(stderr, "%sglm5_run: loaded in %.1f s\n", eng.placement().c_str(), since(t0) / 1000.0);
    const auto& g = eng.geometry();
    std::vector<float> logits((size_t) g.n_vocab);

    auto t1 = std::chrono::steady_clock::now();
    if (step) {
        for (size_t i = 0; i < ids.size(); ++i)
            if (!eng.forward(&ids[i], 1, (int64_t) i, i + 1 == ids.size() ? logits.data() : nullptr, err)) {
                std::fprintf(stderr, "glm5_run: %s\n", err.c_str());
                return 1;
            }
    } else if (!eng.forward(ids.data(), (int) ids.size(), 0, logits.data(), err)) {
        std::fprintf(stderr, "glm5_run: %s\n", err.c_str());
        return 1;
    }
    const double prompt_ms = since(t1);
    std::fprintf(stderr, "prompt: CPU experts %.0f ms of %.0f\n", eng.last_cpu_ms(), prompt_ms);
    eng.print_timing("prompt");
    if (!logits_path.empty()) {
        std::ofstream o(logits_path, std::ios::binary);
        o.write((const char*) logits.data(), (std::streamsize)(logits.size() * 4));
    }
    {
        std::vector<int> top((size_t) g.n_vocab);
        for (int i = 0; i < (int) top.size(); ++i)
            top[(size_t) i] = i;
        std::partial_sort(top.begin(), top.begin() + 5, top.end(),
                          [&](int x, int y) { return logits[(size_t) x] > logits[(size_t) y]; });
        std::fprintf(stderr, "top after the prompt:");
        for (int k = 0; k < 5; ++k)
            std::fprintf(stderr, " %d (%.3f)", top[(size_t) k], logits[(size_t) top[(size_t) k]]);
        std::fprintf(stderr, "\n");
    }
    std::vector<int32_t> out;
    int64_t pos = (int64_t) ids.size();
    double cpu_ms = 0.0;
    int offered = 0, accepted = 0;
    auto argmax = [](const float* l, size_t n) { return (int32_t) (std::max_element(l, l + n) - l); };
    auto is_eos = [&](int32_t t) { return std::find(g.eos_ids.begin(), g.eos_ids.end(), t) != g.eos_ids.end(); };
    t1 = std::chrono::steady_clock::now();
    int32_t next = argmax(logits.data(), logits.size());
    std::vector<float> rows(2 * logits.size());
    while ((int) out.size() < n_new) {
        out.push_back(next);
        if (is_eos(next) || (int) out.size() >= n_new)
            break;
        if (spec && eng.has_mtp()) { // greedy speculation: the main model's argmax over the first row decides
            int32_t d = 0;
            const int32_t win[2] = {next, 0};
            if (!eng.draft(next, pos, d, err)) {
                std::fprintf(stderr, "glm5_run: %s\n", err.c_str());
                return 1;
            }
            if (out.size() < oracle.size())
                d = oracle[out.size()];
            const int32_t w2[2] = {win[0], d};
            if (!eng.verify(w2, 2, pos, rows.data(), err)) {
                std::fprintf(stderr, "glm5_run: %s\n", err.c_str());
                return 1;
            }
            cpu_ms += eng.last_cpu_ms();
            const int32_t y1 = argmax(rows.data(), logits.size());
            ++offered;
            if (y1 == d) {
                ++accepted;
                if (!eng.accept(2, err))
                    return 1;
                pos += 2;
                out.push_back(d);
                if (is_eos(d) || (int) out.size() >= n_new)
                    break;
                next = argmax(rows.data() + logits.size(), logits.size());
            } else {
                if (!eng.accept(1, err))
                    return 1;
                pos += 1;
                next = y1;
            }
            continue;
        }
        if (!eng.forward(&next, 1, pos++, logits.data(), err)) {
            std::fprintf(stderr, "glm5_run: %s\n", err.c_str());
            return 1;
        }
        cpu_ms += eng.last_cpu_ms();
        next = argmax(logits.data(), logits.size());
    }
    const double dec_ms = since(t1);
    eng.print_timing("prompt + decode");
    if (offered)
        std::fprintf(stderr, "MTP: %d of %d drafts accepted (%.0f%%)\n", accepted, offered, 100.0 * accepted / offered);
    for (size_t i = 0; i < out.size(); ++i)
        std::printf("%s%d", i ? "," : "", out[i]);
    std::printf("\n");
    std::fprintf(stderr,
                 "prompt %zu tokens in %.0f ms (%.2f tok/s); decode %zu tokens in %.0f ms (%.2f tok/s, CPU "
                 "experts %.1f ms per step)\n",
                 ids.size(), prompt_ms, ids.size() * 1000.0 / prompt_ms, out.size(), dec_ms,
                 out.size() > 1 ? (out.size() - 1) * 1000.0 / dec_ms : 0.0,
                 cpu_ms / std::max(1, offered ? offered : (int) out.size() - 1));
    return 0;
}
