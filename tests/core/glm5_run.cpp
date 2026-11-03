// tests/core/glm5_run.cpp - GLM-5.3-Flash from token ids: greedy decode, timings, and the prompt's last logits.
//
//   glm5_run <gguf shard> <prompt> [new tokens] [--logits out.f32] [--threads N] [--ctx N]
//            [--step] [--spec] [--no-mtp] [--no-numa] [--no-prefault] [--q6k | --q8 | --q5k | --q4k]
//   <prompt>: comma-separated ids, @<file> for a file of ids (separated by commas, spaces or newlines: argv holds
//   ~18k ids at most), or text:<file> with --tokenizer
//   --tokenizer <dir>: the tokenizer that encodes a text:<file> prompt and decodes the output (printed to stderr)
//   --prompt-tokens N: the prompt's first N ids (a long text cut to a context's length)
//   --save-ids <file>: write the prompt's ids (comma-separated) to <file>; with "-" for the gguf, only that
//   --spec: decode with the NextN block's drafts (greedy; the output equals plain greedy decoding with
//   GUILD_GLM_SHEXP_SPLIT=0 or --q5k / --q4k, and leaves it otherwise: see main_window in src/core/glm5.cpp)
//   --oracle <file>: --spec with the drafts replaced by a plain greedy output (tests the accept path)
//   --step: the prompt one token at a time, as decode runs, instead of windows of 64
//   --no-numa: the experts stay in the GGUF mapping instead of half of each on either NUMA node
//   --dump <file>: append every decode step's logits (the row that picks the next token) to <file>; with --spec
//   a window's rows that pick tokens (the first, and the second when the draft is accepted), so the files compare
//   --q6k / --q8: the attention's Q8_0 matrices re-quantized to Q6_K, or kept (default: Q6_K when Q8_0 would hold
//   less context than --ctx, up to 16k tokens); --q5k / --q4k: to Q5_K / Q4_K, the shared experts' GPU copies too
//   --draft-vocab <file>: drafts over this token subset of the head (tools/draft_vocab.py; GUILD_GLM_DRAFT_VOCAB
//   overrides it, 0 for the whole head)
//   --dflash <dir>: --spec drafts with a DFlash 2 drafter in the NextN block's place (its first draft verified), and
//   counts how often its second draft would have been accepted in a 3-token window; --dflash-vocab <file>: its draft
//   head's token subset
//
//   A/B arms (several decodes in one process: the same load and context):
//   --ab VAR=a,b [--reps N]: the decode N times (default 3) with VAR=a then VAR=b, the order reversed every other
//   round (a b, b a, a b, ...: a drift over the run weighs on both arms alike); repeated, --ab sets several
//   variables an arm (VAR1=a,b --ab VAR2=c,d: arms {a, c} and {b, d}).  Two variables are glm5_run's own:
//   GLM5_RUN_PLAIN=1 decodes without the drafts (with --spec: speculative and plain decoding in one A/B; on
//   GLM-5.3-Flash they part after a few tokens unless GUILD_GLM_SHEXP_SPLIT=0, since a verify window's second token
//   computes its shared experts on the CPU) and GLM5_RUN_DRAFT_VOCAB=1 / 0 drafts over the --draft-vocab subset /
//   the whole head
//   --ab (no VAR=): the arms GLM5_RUN_DRAFT_VOCAB=1 and =0 ("subset" and "full")
//   --plain: one more arm, GLM5_RUN_PLAIN=1 ("plain")
//   --repeat N: the same as --reps N
//   Without --prompts, every arm decodes from a snapshot of the state after the prompt; the arms' ids must be equal
//   when the arms do not change the arithmetic (exit 3 when they are not).
//
//   --prompts <file>: instead of <prompt> (give any, e.g. 0), one prompt a line ("name id,id,..." or "name @file"),
//   each decoded with the drafts (--spec implied) from a reset context, every arm (one arm without --ab / --plain:
//   the draft subset when there is one), --reps rounds (default 1).  A line a generation ("gen <name> <arm> round
//   <r>: ..."), then its ids, then its drafts (id:accepted:ms), on stdout.
//
// Prints the generated ids (one line) for a tokenizer to decode, then prompt and decode rates.
#include "guild/core/glm5.hpp"
#include "guild/runtime/tokenizer.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static void parse_ids(const std::string& s, std::vector<int32_t>& ids) {
    std::string tok;
    for (size_t i = 0; i <= s.size(); ++i) {
        const char c = i < s.size() ? s[i] : ',';
        if (c == ',' || c == ' ' || c == '\n' || c == '\r' || c == '\t') {
            if (!tok.empty())
                ids.push_back((int32_t) std::atol(tok.c_str()));
            tok.clear();
        } else {
            tok += c;
        }
    }
}

static std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// an A/B arm: its name and the variables it sets
struct Arm {
    std::string name;
    std::vector<std::pair<std::string, std::string>> env;
};

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: glm5_run <gguf> <ids | @file | text:file> [new] [--logits f] [--threads n] "
                             "[--ctx n] [--step] [--spec] [--tokenizer dir] [--prompt-tokens n] [--ab VAR=a,b] "
                             "[--reps n] [--prompts file [--ab] [--plain]]\n");
        return 2;
    }
    int n_new = 32;
    std::string logits_path, dump_path, tok_dir, save_ids, prompts_path;
    std::vector<std::pair<std::string, std::vector<std::string>>> ab; // (variable, its value in each arm)
    int reps = 0;                                                     // 0: the mode's default
    int64_t prompt_tokens = 0;
    bool step = false, spec = false, ab_vocab = false, plain_arm = false;
    std::vector<int32_t> oracle;
    guild::core::Glm5Options opt;
    for (int i = 3; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--logits" && i + 1 < argc)
            logits_path = argv[++i];
        else if (a == "--dump" && i + 1 < argc) // every decode step's logits row that picks a token
            dump_path = argv[++i];
        else if (a == "--threads" && i + 1 < argc)
            opt.threads = std::atoi(argv[++i]);
        else if (a == "--ctx" && i + 1 < argc)
            opt.max_context = std::atol(argv[++i]);
        else if (a == "--tokenizer" && i + 1 < argc)
            tok_dir = argv[++i];
        else if (a == "--prompt-tokens" && i + 1 < argc)
            prompt_tokens = std::atol(argv[++i]);
        else if (a == "--save-ids" && i + 1 < argc)
            save_ids = argv[++i];
        else if ((a == "--reps" || a == "--repeat") && i + 1 < argc)
            reps = std::max(1, std::atoi(argv[++i]));
        else if (a == "--ab" && i + 1 < argc && std::strchr(argv[i + 1], '=') && argv[i + 1][0] != '-') { // VAR=a,b
            const std::string v = argv[++i];
            const size_t eq = v.find('=');
            ab.push_back({v.substr(0, eq), {}});
            std::stringstream ss(v.substr(eq + 1));
            std::string arm;
            while (std::getline(ss, arm, ','))
                ab.back().second.push_back(arm);
            if (ab.back().second.size() != ab.front().second.size()) {
                std::fprintf(stderr, "glm5_run: every --ab takes as many values\n");
                return 2;
            }
        } else if (a == "--ab")
            ab_vocab = true;
        else if (a == "--plain")
            plain_arm = true;
        else if (a == "--prompts" && i + 1 < argc)
            prompts_path = argv[++i];
        else if (a == "--draft-vocab" && i + 1 < argc)
            opt.draft_vocab = argv[++i];
        else if (a == "--dflash" && i + 1 < argc)
            opt.dflash = argv[++i];
        else if (a == "--dflash-vocab" && i + 1 < argc)
            opt.dflash_vocab = argv[++i];
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
        else if (a == "--q5k")
            opt.dense = guild::core::Glm5Dense::Q5_K;
        else if (a == "--q4k")
            opt.dense = guild::core::Glm5Dense::Q4_K;
        else if (a == "--oracle" && i + 1 < argc) { // test: drafts are this greedy output's tokens (all accepted)
            std::string line;
            std::ifstream f(argv[++i]);
            std::getline(f, line);
            parse_ids(line, oracle);
            spec = true;
        } else
            n_new = std::atoi(a.c_str());
    }
    if (ab_vocab && !ab.empty()) {
        std::fprintf(stderr, "glm5_run: --ab without VAR= (the draft subset A/B) does not mix with --ab VAR=a,b\n");
        return 2;
    }
    if (!prompts_path.empty())
        spec = true;

    // the prompt: ids, a file of ids, or a text file through the tokenizer
    guild::runtime::Tokenizer tok;
    std::string err;
    if (!tok_dir.empty() && !tok.load(tok_dir, err)) {
        std::fprintf(stderr, "glm5_run: tokenizer: %s\n", err.c_str());
        return 1;
    }
    std::vector<int32_t> ids;
    const std::string prompt_arg = argv[2];
    if (prompt_arg.rfind("text:", 0) == 0) {
        if (tok_dir.empty()) {
            std::fprintf(stderr, "glm5_run: a text prompt needs --tokenizer\n");
            return 2;
        }
        std::string text = read_file(prompt_arg.substr(5));
        // a token is ~4 characters of English or code: no need to encode all of a long file
        if (prompt_tokens > 0 && (int64_t) text.size() > prompt_tokens * 8)
            text.resize((size_t) prompt_tokens * 8);
        if (!tok.encode(text, ids, false, err)) {
            std::fprintf(stderr, "glm5_run: encode: %s\n", err.c_str());
            return 1;
        }
    } else if (!prompt_arg.empty() && prompt_arg[0] == '@') {
        parse_ids(read_file(prompt_arg.substr(1)), ids);
    } else {
        parse_ids(prompt_arg, ids);
    }
    if (prompt_tokens > 0 && (int64_t) ids.size() > prompt_tokens)
        ids.resize((size_t) prompt_tokens);
    if (ids.empty()) {
        std::fprintf(stderr, "glm5_run: an empty prompt\n");
        return 2;
    }
    if (prompts_path.empty())
        std::fprintf(stderr, "glm5_run: a prompt of %zu tokens\n", ids.size());
    if (!save_ids.empty()) {
        std::ofstream o(save_ids);
        for (size_t i = 0; i < ids.size(); ++i)
            o << (i ? "," : "") << ids[i];
        o << "\n";
    }
    if (std::string(argv[1]) == "-")
        return 0;

    // --prompts: "name id,id,..." or "name @file" a line
    std::vector<std::pair<std::string, std::vector<int32_t>>> prompts;
    if (!prompts_path.empty()) {
        std::ifstream f(prompts_path);
        std::string line;
        while (std::getline(f, line)) {
            std::stringstream ls(line);
            std::string name, list;
            if (!(ls >> name >> list))
                continue;
            std::vector<int32_t> p;
            parse_ids(list[0] == '@' ? read_file(list.substr(1)) : list, p);
            if (!p.empty())
                prompts.emplace_back(name, p);
        }
        if (prompts.empty()) {
            std::fprintf(stderr, "glm5_run: --prompts: no prompt lines in %s\n", prompts_path.c_str());
            return 2;
        }
    }

    guild::core::Glm5Engine eng;
    const auto t0 = std::chrono::steady_clock::now();
    if (!eng.load(argv[1], opt, err)) {
        std::fprintf(stderr, "glm5_run: %s\n", err.c_str());
        return 1;
    }
    auto since = [](std::chrono::steady_clock::time_point t) {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
    };
    std::fprintf(stderr, "%sglm5_run: loaded in %.1f s\n", eng.placement().c_str(), since(t0) / 1000.0);
    if (!prompts.empty() && !eng.has_mtp() && !eng.has_dflash()) {
        std::fprintf(stderr, "glm5_run: --prompts needs the NextN block or a DFlash drafter\n");
        return 2;
    }
    const auto& g = eng.geometry();
    const size_t V = (size_t) g.n_vocab;
    std::vector<float> logits(V);

    // the arms: --ab VAR=a,b, or the draft subset and the whole head (--ab), and --plain
    std::vector<Arm> arms;
    if (!ab.empty()) {
        for (size_t a = 0; a < ab.front().second.size(); ++a) {
            Arm arm;
            for (const auto& v : ab) {
                arm.name += (arm.name.empty() ? "" : " ") + v.first + "=" + v.second[a];
                arm.env.push_back({v.first, v.second[a]});
            }
            arms.push_back(arm);
        }
    } else if (ab_vocab) {
        arms.push_back({"subset", {{"GLM5_RUN_DRAFT_VOCAB", "1"}}});
        arms.push_back({"full", {{"GLM5_RUN_DRAFT_VOCAB", "0"}}});
    } else if (!prompts.empty()) {
        const bool subset = eng.use_draft_vocab(true);
        arms.push_back({subset ? "subset" : "full", {{"GLM5_RUN_DRAFT_VOCAB", subset ? "1" : "0"}}});
    }
    if (plain_arm) {
        if (arms.empty())
            arms.push_back({"spec", {}});
        for (Arm& arm : arms)
            arm.env.push_back({"GLM5_RUN_PLAIN", "0"});
        arms.push_back({"plain", {{"GLM5_RUN_PLAIN", "1"}}});
    }
    auto set_arm = [](const Arm& arm) {
        for (const auto& v : arm.env)
            setenv(v.first.c_str(), v.second.c_str(), 1);
    };
    // the A/B arms run in reverse order every other round (A B, B A, ...: a drift over the run weighs on every arm
    // alike); --plain's arm stays last
    auto arm_order = [&](int round) {
        std::vector<size_t> order;
        for (size_t a = 0; a < arms.size(); ++a)
            order.push_back(a);
        const size_t n_ab = plain_arm ? arms.size() - 1 : arms.size();
        if (round % 2 == 1 && n_ab > 1)
            std::reverse(order.begin(), order.begin() + (long) n_ab);
        return order;
    };

    // one decode after a prompt of n_prompt tokens whose last logits are first
    struct Decode {
        std::vector<int32_t> out;
        std::vector<double> step_ms, step_cpu; // per forward: wall time, and the CPU experts' part
        double cpu_ms = 0, dec_ms = 0, draft_ms = 0, verify_ms = 0, accept_ms = 0; // --spec: wall time in each call
        int offered = 0, accepted = 0;
        int second = 0; // --dflash: steps whose first draft was accepted and whose second equals the next token
        std::vector<int32_t> drafts;   // --spec: each step's draft,
        std::vector<char> draft_ok;    // whether it was accepted,
        std::vector<double> draft_one; // and its ms
    };
    auto argmax = [](const float* l, size_t n) { return (int32_t) (std::max_element(l, l + n) - l); };
    // GLM5_RUN_IGNORE_EOS=1: decode past an end token (arms whose texts differ decode the same number of tokens)
    const bool ignore_eos = [] {
        const char* e = std::getenv("GLM5_RUN_IGNORE_EOS");
        return e && e[0] == '1';
    }();
    auto is_eos = [&](int32_t t) {
        return !ignore_eos && std::find(g.eos_ids.begin(), g.eos_ids.end(), t) != g.eos_ids.end();
    };
    auto decode = [&](Decode& r, int64_t n_prompt, const std::vector<float>& first) -> bool {
        int64_t pos = n_prompt;
        auto tstep = std::chrono::steady_clock::now();
        auto mark = [&] {
            r.step_ms.push_back(since(tstep));
            r.step_cpu.push_back(eng.last_cpu_ms());
            tstep = std::chrono::steady_clock::now();
        };
        const auto td = tstep;
        int32_t next = argmax(first.data(), V);
        std::vector<float> rows(2 * V);
        const char* plain = std::getenv("GLM5_RUN_PLAIN");
        const char* vocab = std::getenv("GLM5_RUN_DRAFT_VOCAB");
        if (vocab && vocab[0])
            eng.use_draft_vocab(vocab[0] == '1');
        const bool use_spec = spec && (eng.has_mtp() || eng.has_dflash()) && !(plain && plain[0] == '1');
        int32_t dfl[16];
        float dprob[16];
        while ((int) r.out.size() < n_new) {
            r.out.push_back(next);
            if (is_eos(next) || (int) r.out.size() >= n_new)
                break;
            if (use_spec) { // greedy speculation: the main model's argmax over the first row decides
                int32_t d = 0;
                auto tc = std::chrono::steady_clock::now();
                int nd = 0;
                if (eng.has_dflash()) {
                    if (!eng.dflash_draft(next, pos, dfl, dprob, nd, err))
                        return false;
                    d = dfl[0];
                } else if (!eng.draft(next, pos, d, err))
                    return false;
                if (r.out.size() < oracle.size())
                    d = oracle[r.out.size()];
                const int32_t w2[2] = {next, d};
                const double dms = since(tc);
                r.draft_ms += dms;
                tc = std::chrono::steady_clock::now();
                if (!eng.verify(w2, 2, pos, rows.data(), err))
                    return false;
                r.cpu_ms += eng.last_cpu_ms();
                r.verify_ms += since(tc);
                tc = std::chrono::steady_clock::now();
                const int32_t y1 = argmax(rows.data(), V);
                ++r.offered;
                r.drafts.push_back(d);
                r.draft_ok.push_back(y1 == d);
                r.draft_one.push_back(dms);
                if (!dump_path.empty()) { // the rows that pick tokens, as plain decode's --dump writes them
                    std::ofstream o(dump_path, std::ios::binary | std::ios::app);
                    o.write((const char*) rows.data(), (std::streamsize)((y1 == d ? 2 : 1) * V * 4));
                }
                if (y1 == d) {
                    ++r.accepted;
                    if (!eng.accept(2, err))
                        return false;
                    r.accept_ms += since(tc);
                    mark();
                    pos += 2;
                    r.out.push_back(d);
                    if (is_eos(d) || (int) r.out.size() >= n_new)
                        break;
                    next = argmax(rows.data() + V, V);
                    if (nd > 1 && dfl[1] == next) // a 3-token window would have kept the second draft too
                        ++r.second;
                } else {
                    if (!eng.accept(1, err))
                        return false;
                    r.accept_ms += since(tc);
                    mark();
                    pos += 1;
                    next = y1;
                }
                continue;
            }
            if (!eng.forward(&next, 1, pos++, logits.data(), err))
                return false;
            r.cpu_ms += eng.last_cpu_ms();
            mark();
            if (!dump_path.empty()) {
                std::ofstream o(dump_path, std::ios::binary | std::ios::app);
                o.write((const char*) logits.data(), (std::streamsize)(V * 4));
            }
            next = argmax(logits.data(), V);
        }
        r.dec_ms = since(td);
        return true;
    };
    auto pct = [](std::vector<double> v, double q) {
        std::sort(v.begin(), v.end());
        return v.empty() ? 0.0 : v[(size_t) (q * (double) (v.size() - 1))];
    };
    auto tok_s = [](const Decode& r) { return r.out.size() > 1 ? (r.out.size() - 1) * 1000.0 / r.dec_ms : 0.0; };
    auto print_ids = [](const std::vector<int32_t>& out) {
        for (size_t i = 0; i < out.size(); ++i)
            std::printf("%s%d", i ? "," : "", out[i]);
        std::printf("\n");
    };
    auto print_text = [&](const std::vector<int32_t>& out) {
        if (tok_dir.empty())
            return;
        guild::runtime::IncrementalDecoder dec(&tok, false);
        std::string s;
        for (int32_t id : out)
            s += dec.add(id);
        s += dec.flush();
        std::fprintf(stderr, "output text: %s\n", s.c_str());
    };

    if (!prompts.empty()) {
        // every prompt from a reset context, every arm, the rounds
        const int rounds = reps > 0 ? reps : 1;
        for (int round = 0; round < rounds; ++round)
            for (const auto& [name, p] : prompts)
                for (size_t a : arm_order(round)) {
                    const Arm& arm = arms[a];
                    set_arm(arm);
                    eng.reset();
                    const auto tp = std::chrono::steady_clock::now();
                    if (!eng.forward(p.data(), (int) p.size(), 0, logits.data(), err)) {
                        std::fprintf(stderr, "glm5_run: %s\n", err.c_str());
                        return 1;
                    }
                    const double prompt_ms =
                        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tp).count();
                    uint64_t lh = 1469598103934665603ull; // FNV-1a of the prompt's last logits, to compare arms
                    for (const float v : logits) {
                        uint32_t u;
                        std::memcpy(&u, &v, 4);
                        lh = (lh ^ u) * 1099511628211ull;
                    }
                    const std::vector<float> first = logits;
                    Decode r;
                    if (!decode(r, (int64_t) p.size(), first)) {
                        std::fprintf(stderr, "glm5_run: %s\n", err.c_str());
                        return 1;
                    }
                    std::string label = arm.name;
                    std::replace(label.begin(), label.end(), ' ', ',');
                    const int steps = std::max(1, r.offered ? r.offered : (int) r.step_ms.size());
                    std::printf("gen %s %s round %d: prompt %zu tokens in %.0f ms (logits %016llx); %zu tokens, %zu "
                                "steps, %d drafts accepted (%.1f%%; the second too: %d); per step: "
                                "draft %.3f ms (median %.3f, 90th %.3f), verify %.2f, accept %.2f, CPU experts %.1f, "
                                "step median %.2f; %.2f tok/s\n",
                                name.c_str(), label.c_str(), round, p.size(), prompt_ms, (unsigned long long) lh,
                                r.out.size(), r.step_ms.size(), r.accepted, 100.0 * r.accepted / std::max(1, r.offered),
                                r.second, r.draft_ms / steps, pct(r.draft_one, 0.5), pct(r.draft_one, 0.9),
                                r.verify_ms / steps, r.accept_ms / steps, r.cpu_ms / steps, pct(r.step_ms, 0.5),
                                tok_s(r));
                    std::printf("ids %s %s round %d:", name.c_str(), label.c_str(), round);
                    for (size_t i = 0; i < r.out.size(); ++i)
                        std::printf("%s%d", i ? "," : " ", r.out[i]);
                    std::printf("\ndrafts %s %s round %d:", name.c_str(), label.c_str(), round);
                    for (size_t i = 0; i < r.drafts.size(); ++i)
                        std::printf("%s%d:%d:%.2f", i ? "," : " ", r.drafts[i], (int) r.draft_ok[i], r.draft_one[i]);
                    std::printf("\n");
                    std::fflush(stdout);
                }
        return 0;
    }

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
        o.write((const char*) logits.data(), (std::streamsize)(V * 4));
    }
    {
        std::vector<int> top(V);
        for (int i = 0; i < (int) top.size(); ++i)
            top[(size_t) i] = i;
        std::partial_sort(top.begin(), top.begin() + 5, top.end(),
                          [&](int x, int y) { return logits[(size_t) x] > logits[(size_t) y]; });
        std::fprintf(stderr, "top after the prompt:");
        for (int k = 0; k < 5; ++k)
            std::fprintf(stderr, " %d (%.3f)", top[(size_t) k], logits[(size_t) top[(size_t) k]]);
        std::fprintf(stderr, "\n");
    }
    const std::vector<float> prompt_logits = logits;
    auto report = [&](const Decode& r, const std::string& tag) {
        if (r.offered)
            std::fprintf(
                stderr,
                "%s%s: %d of %d drafts accepted (%.0f%%)%s; per step: draft %.1f ms, verify %.1f, accept %.1f\n",
                tag.c_str(), eng.has_dflash() ? "DFlash" : "MTP", r.accepted, r.offered, 100.0 * r.accepted / r.offered,
                eng.has_dflash() ? (", the second draft too at " + std::to_string(r.second) + " of them").c_str() : "",
                r.draft_ms / r.offered, r.verify_ms / r.offered, r.accept_ms / r.offered);
        if (std::getenv("GLM5_RUN_STEPS")) {
            std::fprintf(stderr, "%ssteps (ms, CPU ms):", tag.c_str());
            for (size_t i = 0; i < r.step_ms.size(); ++i)
                std::fprintf(stderr, " %.0f/%.0f", r.step_ms[i], r.step_cpu[i]);
            std::fprintf(stderr, "\n");
        }
        if (r.step_ms.size() >= 4)
            std::fprintf(stderr,
                         "%sper step: %.1f / %.1f / %.1f ms (10th / 50th / 90th percentile), CPU experts %.1f / %.1f / "
                         "%.1f\n",
                         tag.c_str(), pct(r.step_ms, 0.1), pct(r.step_ms, 0.5), pct(r.step_ms, 0.9),
                         pct(r.step_cpu, 0.1), pct(r.step_cpu, 0.5), pct(r.step_cpu, 0.9));
    };

    if (!arms.empty()) {
        // the A/B: the decode from a snapshot of the state after the prompt, the arms in turn
        if (!eng.snapshot(err)) {
            std::fprintf(stderr, "glm5_run: --ab: %s\n", err.c_str());
            return 1;
        }
        const int rounds = reps > 0 ? reps : 3;
        std::vector<std::vector<Decode>> runs(arms.size());
        for (int rep = 0; rep < rounds; ++rep)
            for (size_t a : arm_order(rep)) {
                set_arm(arms[a]);
                Decode r;
                if (!eng.restore(err) || !decode(r, (int64_t) ids.size(), prompt_logits)) {
                    std::fprintf(stderr, "glm5_run: %s\n", err.c_str());
                    return 1;
                }
                const std::string tag = arms[a].name + " #" + std::to_string(rep + 1) + ": ";
                report(r, tag);
                std::fprintf(stderr, "%sdecode %zu tokens in %.0f ms (%.2f tok/s)\n", tag.c_str(), r.out.size(),
                             r.dec_ms, tok_s(r));
                runs[a].push_back(std::move(r));
            }
        // every run against the first arm's first, and where it parts from it
        bool same = true;
        for (size_t a = 0; a < arms.size(); ++a)
            for (size_t i = 0; i < runs[a].size(); ++i) {
                const std::vector<int32_t>&x = runs[0][0].out, &y = runs[a][i].out;
                if (x == y)
                    continue;
                same = false;
                size_t at = 0;
                while (at < x.size() && at < y.size() && x[at] == y[at])
                    ++at;
                std::fprintf(stderr, "A/B ids: %s #%zu parts from %s #1 at generated token %zu\n", arms[a].name.c_str(),
                             i + 1, arms[0].name.c_str(), at + 1);
            }
        for (size_t a = 0; a < arms.size(); ++a) {
            std::vector<double> all;
            double ms = 0;
            size_t toks = 0;
            for (const Decode& r : runs[a]) {
                all.insert(all.end(), r.step_ms.begin(), r.step_ms.end());
                ms += r.dec_ms;
                toks += r.out.size() > 1 ? r.out.size() - 1 : 0;
            }
            std::fprintf(stderr, "A/B %s: per step %.1f / %.1f / %.1f ms (10th / 50th / 90th, %d runs), %.2f tok/s\n",
                         arms[a].name.c_str(), pct(all, 0.1), pct(all, 0.5), pct(all, 0.9), rounds,
                         toks * 1000.0 / std::max(1e-3, ms));
        }
        std::fprintf(stderr, "A/B ids: %s\n", same ? "identical in every run" : "DIFFERENT");
        print_ids(runs[0][0].out);
        print_text(runs[0][0].out);
        return same ? 0 : 3;
    }

    Decode r;
    if (!decode(r, (int64_t) ids.size(), prompt_logits)) {
        std::fprintf(stderr, "glm5_run: %s\n", err.c_str());
        return 1;
    }
    eng.print_timing("prompt + decode");
    report(r, "");
    print_ids(r.out);
    print_text(r.out);
    std::fprintf(stderr,
                 "prompt %zu tokens in %.0f ms (%.2f tok/s); decode %zu tokens in %.0f ms (%.2f tok/s, CPU "
                 "experts %.1f ms per step)\n",
                 ids.size(), prompt_ms, ids.size() * 1000.0 / prompt_ms, r.out.size(), r.dec_ms, tok_s(r),
                 r.cpu_ms / std::max(1, r.offered ? r.offered : (int) r.out.size() - 1));
    return 0;
}
