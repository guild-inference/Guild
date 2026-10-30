// tests/core/glm5_run.cpp - GLM-5.3-Flash from token ids: greedy decode, timings, and the prompt's last logits.
//
//   glm5_run <gguf shard> <comma-separated prompt ids> [new tokens] [--logits out.f32] [--threads N] [--ctx N]
//            [--step] [--spec] [--no-mtp] [--no-numa] [--no-prefault] [--q6k | --q8]
//   --spec: decode with the NextN block's drafts (greedy; the output equals plain greedy decoding with
//   GUILD_GLM_SHEXP_SPLIT=0, and leaves it by default: see main_window in src/core/glm5.cpp)
//   --oracle <file>: --spec with the drafts replaced by a plain greedy output (tests the accept path)
//   --step: the prompt one token at a time, as decode runs, instead of windows of 64
//   --no-numa: the experts stay in the GGUF mapping instead of half of each on either NUMA node
//   --dump <file>: append every decode step's logits (the row that picks the next token) to <file>; with --spec
//   a window's rows that pick tokens (the first, and the second when the draft is accepted), so the files compare
//   --q6k / --q8: the attention's Q8_0 matrices re-quantized to Q6_K, or kept (default: Q6_K when Q8_0 would hold
//   less context than --ctx, up to 16k tokens)
//   --draft-vocab <file>: drafts over this token subset of the head (tools/draft_vocab.py; GUILD_GLM_DRAFT_VOCAB
//   overrides it, 0 for the whole head)
//   --prompts <file>: instead of <ids>, one prompt a line ("name id,id,..."), each decoded with --spec from a reset
//   context; --repeat N rounds of them; --ab: each prompt twice a round, drafting over the --draft-vocab subset and
//   over the whole head (which goes first alternates by round); --plain: each prompt also decoded without drafts.  A
//   line a generation, then its ids, then its drafts (id:accepted:ms).
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

// --prompts: every prompt of `file`, `repeat` rounds, greedy with the NextN block's drafts (and with --plain, without:
// the reference a speculative decode should equal)
static int run_prompts(guild::core::Glm5Engine& eng, const std::string& file, int n_new, int repeat, bool ab,
                       bool plain) {
    std::vector<std::pair<std::string, std::vector<int32_t>>> prompts;
    {
        std::ifstream f(file);
        std::string line;
        while (std::getline(f, line)) {
            std::stringstream ls(line);
            std::string name, list, tok;
            if (!(ls >> name >> list))
                continue;
            std::vector<int32_t> ids;
            std::stringstream ss(list);
            while (std::getline(ss, tok, ','))
                if (!tok.empty())
                    ids.push_back((int32_t) std::atol(tok.c_str()));
            prompts.emplace_back(name, ids);
        }
    }
    if (prompts.empty() || !eng.has_mtp()) {
        std::fprintf(stderr, "glm5_run: --prompts needs prompt lines and the NextN block\n");
        return 2;
    }
    const auto& g = eng.geometry();
    const size_t V = (size_t) g.n_vocab;
    std::vector<float> logits(V), rows(2 * V);
    auto argmax = [V](const float* l) { return (int32_t) (std::max_element(l, l + V) - l); };
    auto is_eos = [&](int32_t t) { return std::find(g.eos_ids.begin(), g.eos_ids.end(), t) != g.eos_ids.end(); };
    using clk = std::chrono::steady_clock;
    auto ms = [](clk::time_point t) { return std::chrono::duration<double, std::milli>(clk::now() - t).count(); };
    auto pct = [](std::vector<double> v, double q) {
        if (v.empty())
            return 0.0;
        std::sort(v.begin(), v.end());
        return v[(size_t) (q * (double) (v.size() - 1))];
    };
    std::string err;
    for (int r = 0; r < repeat; ++r)
        for (const auto& [name, ids] : prompts) {
            std::vector<std::string> arms;
            if (ab && r % 2 == 0)
                arms = {"subset", "full"};
            else if (ab)
                arms = {"full", "subset"};
            else
                arms = {eng.use_draft_vocab(true) ? "subset" : "full"};
            if (plain)
                arms.push_back("plain");
            for (const std::string& arm : arms) {
                eng.use_draft_vocab(arm == "subset");
                eng.reset();
                if (!eng.forward(ids.data(), (int) ids.size(), 0, logits.data(), err)) {
                    std::fprintf(stderr, "glm5_run: %s\n", err.c_str());
                    return 1;
                }
                std::vector<int32_t> out;
                std::vector<std::pair<int32_t, bool>> drafts; // each step's draft and whether it was accepted
                std::vector<double> dms, sms;                 // each step's draft and whole time
                int64_t pos = (int64_t) ids.size();
                int offered = 0, accepted = 0;
                double verify_ms = 0, accept_ms = 0, cpu_ms = 0;
                const auto t1 = clk::now();
                int32_t next = argmax(logits.data());
                while ((int) out.size() < n_new) {
                    out.push_back(next);
                    if (is_eos(next) || (int) out.size() >= n_new)
                        break;
                    const auto ts = clk::now();
                    if (arm == "plain") {
                        if (!eng.forward(&next, 1, pos++, logits.data(), err)) {
                            std::fprintf(stderr, "glm5_run: %s\n", err.c_str());
                            return 1;
                        }
                        next = argmax(logits.data());
                        sms.push_back(ms(ts));
                        continue;
                    }
                    int32_t d = 0;
                    if (!eng.draft(next, pos, d, err)) {
                        std::fprintf(stderr, "glm5_run: %s\n", err.c_str());
                        return 1;
                    }
                    dms.push_back(ms(ts));
                    auto tc = clk::now();
                    const int32_t w[2] = {next, d};
                    if (!eng.verify(w, 2, pos, rows.data(), err)) {
                        std::fprintf(stderr, "glm5_run: %s\n", err.c_str());
                        return 1;
                    }
                    verify_ms += ms(tc);
                    cpu_ms += eng.last_cpu_ms();
                    tc = clk::now();
                    const int32_t y1 = argmax(rows.data());
                    const bool ok = y1 == d;
                    drafts.emplace_back(d, ok);
                    ++offered;
                    if (!eng.accept(ok ? 2 : 1, err)) {
                        std::fprintf(stderr, "glm5_run: %s\n", err.c_str());
                        return 1;
                    }
                    accept_ms += ms(tc);
                    sms.push_back(ms(ts));
                    if (ok) {
                        ++accepted;
                        pos += 2;
                        out.push_back(d);
                        if (is_eos(d) || (int) out.size() >= n_new)
                            break;
                        next = argmax(rows.data() + V);
                    } else {
                        pos += 1;
                        next = y1;
                    }
                }
                const double dec_ms = ms(t1);
                const int steps = std::max(1, offered);
                double dsum = 0;
                for (double x : dms)
                    dsum += x;
                std::printf("gen %s %s round %d: %zu tokens, %zu steps, %d drafts accepted (%.1f%%); per step: draft "
                            "%.3f ms (median %.3f, 90th %.3f), verify %.2f, accept %.2f, CPU experts %.1f, step median "
                            "%.2f; %.2f tok/s\n",
                            name.c_str(), arm.c_str(), r, out.size(), sms.size(), accepted, 100.0 * accepted / steps,
                            dsum / steps, pct(dms, 0.5), pct(dms, 0.9), verify_ms / steps, accept_ms / steps,
                            cpu_ms / steps, pct(sms, 0.5),
                            out.size() > 1 ? (double) (out.size() - 1) * 1000.0 / dec_ms : 0.0);
                std::printf("ids %s %s round %d:", name.c_str(), arm.c_str(), r);
                for (size_t i = 0; i < out.size(); ++i)
                    std::printf("%s%d", i ? "," : " ", out[i]);
                std::printf("\ndrafts %s %s round %d:", name.c_str(), arm.c_str(), r);
                for (size_t i = 0; i < drafts.size(); ++i)
                    std::printf("%s%d:%d:%.2f", i ? "," : " ", drafts[i].first, (int) drafts[i].second, dms[i]);
                std::printf("\n");
                std::fflush(stdout);
            }
        }
    return 0;
}

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
    int n_new = 32, repeat = 1;
    std::string logits_path, dump_path, prompts_path;
    bool step = false, spec = false, ab = false, plain = false;
    std::vector<int32_t> oracle;
    guild::core::Glm5Options opt;
    for (int i = 3; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--logits" && i + 1 < argc)
            logits_path = argv[++i];
        else if (a == "--dump" && i + 1 < argc) // every plain decode step's logits, one row each
            dump_path = argv[++i];
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
            opt.dense = guild::core::Glm5Dense::Q6_K;
        else if (a == "--q8")
            opt.dense = guild::core::Glm5Dense::Q8_0;
        else if (a == "--draft-vocab" && i + 1 < argc)
            opt.draft_vocab = argv[++i];
        else if (a == "--prompts" && i + 1 < argc)
            prompts_path = argv[++i];
        else if (a == "--repeat" && i + 1 < argc)
            repeat = std::atoi(argv[++i]);
        else if (a == "--ab")
            ab = true;
        else if (a == "--plain")
            plain = true;
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
    if (!prompts_path.empty())
        return run_prompts(eng, prompts_path, n_new, std::max(1, repeat), ab, plain);
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
    std::vector<double> step_ms, step_cpu; // per forward: wall time, and the CPU experts' part
    auto tstep = std::chrono::steady_clock::now();
    auto mark = [&] {
        step_ms.push_back(since(tstep));
        step_cpu.push_back(eng.last_cpu_ms());
        tstep = std::chrono::steady_clock::now();
    };
    int offered = 0, accepted = 0;
    double draft_ms = 0, verify_ms = 0, accept_ms = 0; // --spec: wall time in each call
    auto argmax = [](const float* l, size_t n) { return (int32_t) (std::max_element(l, l + n) - l); };
    auto is_eos = [&](int32_t t) { return std::find(g.eos_ids.begin(), g.eos_ids.end(), t) != g.eos_ids.end(); };
    t1 = tstep = std::chrono::steady_clock::now();
    int32_t next = argmax(logits.data(), logits.size());
    std::vector<float> rows(2 * logits.size());
    while ((int) out.size() < n_new) {
        out.push_back(next);
        if (is_eos(next) || (int) out.size() >= n_new)
            break;
        if (spec && eng.has_mtp()) { // greedy speculation: the main model's argmax over the first row decides
            int32_t d = 0;
            const int32_t win[2] = {next, 0};
            auto tc = std::chrono::steady_clock::now();
            if (!eng.draft(next, pos, d, err)) {
                std::fprintf(stderr, "glm5_run: %s\n", err.c_str());
                return 1;
            }
            if (out.size() < oracle.size())
                d = oracle[out.size()];
            const int32_t w2[2] = {win[0], d};
            draft_ms += since(tc);
            tc = std::chrono::steady_clock::now();
            if (!eng.verify(w2, 2, pos, rows.data(), err)) {
                std::fprintf(stderr, "glm5_run: %s\n", err.c_str());
                return 1;
            }
            cpu_ms += eng.last_cpu_ms();
            verify_ms += since(tc);
            tc = std::chrono::steady_clock::now();
            const int32_t y1 = argmax(rows.data(), logits.size());
            ++offered;
            if (!dump_path.empty()) { // the rows that pick tokens, as plain decode's --dump writes them
                std::ofstream o(dump_path, std::ios::binary | std::ios::app);
                o.write((const char*) rows.data(), (std::streamsize)((y1 == d ? 2 : 1) * logits.size() * 4));
            }
            if (y1 == d) {
                ++accepted;
                if (!eng.accept(2, err))
                    return 1;
                accept_ms += since(tc);
                mark();
                pos += 2;
                out.push_back(d);
                if (is_eos(d) || (int) out.size() >= n_new)
                    break;
                next = argmax(rows.data() + logits.size(), logits.size());
            } else {
                if (!eng.accept(1, err))
                    return 1;
                accept_ms += since(tc);
                mark();
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
        mark();
        if (!dump_path.empty()) {
            std::ofstream o(dump_path, std::ios::binary | std::ios::app);
            o.write((const char*) logits.data(), (std::streamsize)(logits.size() * 4));
        }
        next = argmax(logits.data(), logits.size());
    }
    const double dec_ms = since(t1);
    eng.print_timing("prompt + decode");
    if (offered)
        std::fprintf(stderr,
                     "MTP: %d of %d drafts accepted (%.0f%%); per step: draft %.1f ms, verify %.1f, accept %.1f\n",
                     accepted, offered, 100.0 * accepted / offered, draft_ms / offered, verify_ms / offered,
                     accept_ms / offered);
    if (std::getenv("GLM5_RUN_STEPS")) {
        std::fprintf(stderr, "steps (ms, CPU ms):");
        for (size_t i = 0; i < step_ms.size(); ++i)
            std::fprintf(stderr, " %.0f/%.0f", step_ms[i], step_cpu[i]);
        std::fprintf(stderr, "\n");
    }
    if (step_ms.size() >= 4) {
        auto pct = [](std::vector<double> v, double q) {
            std::sort(v.begin(), v.end());
            return v[(size_t) (q * (double) (v.size() - 1))];
        };
        std::fprintf(stderr,
                     "per step: %.1f / %.1f / %.1f ms (10th / 50th / 90th percentile), CPU experts %.1f / %.1f / "
                     "%.1f\n",
                     pct(step_ms, 0.1), pct(step_ms, 0.5), pct(step_ms, 0.9), pct(step_cpu, 0.1), pct(step_cpu, 0.5),
                     pct(step_cpu, 0.9));
    }
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
