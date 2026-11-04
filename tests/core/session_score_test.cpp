// tests/core/session_score_test.cpp - GuildSession::score on any model guild-generate loads (Qwen, Ornith,
// StartLux-Decision, GLM-5.3-Flash):
//
//   session_score_test <guild-generate's model options> [--text <file>] [--tokens <n>]
//
//   1. score() of a prompt: its argmax equals the token generate() picks first (greedy), on the same prompt;
//   2. a repeat of the same prompt gives the same logits, bit for bit (the state is left after the prompt: a repeat
//      reads it again);
//   3. two prompts sharing a beginning, the first with keep_prefix at it: the second's logits against the same
//      prompt read from nothing (a different split of the reading: equal within --tol, and the same argmax);
//   4. a candidate subset returns those entries of the whole row.
// Exit 0 when all hold.
#include "guild/program/model_options.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace guild::runtime;

namespace {

int fails = 0;

void check(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
    fails += ok ? 0 : 1;
}

size_t argmax(const std::vector<float>& v) {
    return (size_t) (std::max_element(v.begin(), v.end()) - v.begin());
}

double max_abs(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i)
        m = std::max(m, (double) std::fabs(a[i] - b[i]));
    return m;
}

} // namespace

int main(int argc, char** argv) try {
    if (std::getenv("CUDA_MODULE_LOADING") == nullptr)
        setenv("CUDA_MODULE_LOADING", "EAGER", 0);
    guild::program::ModelOptions mo;
    mo.max_context = 8192;
    std::string text_path;
    int n_tokens = 400;
    double tol = 0.5;
    for (int i = 1; i < argc; ++i) {
        const int r = guild::program::parse_model_option(argc, argv, i, mo, "session_score_test");
        if (r == 2)
            return 2;
        if (r == 1)
            continue;
        const std::string a = argv[i];
        if (a == "--text" && i + 1 < argc)
            text_path = argv[++i];
        else if (a == "--tokens" && i + 1 < argc)
            n_tokens = std::atoi(argv[++i]);
        else if (a == "--tol" && i + 1 < argc)
            tol = std::atof(argv[++i]);
        else {
            std::fprintf(stderr, "session_score_test: unknown option %s\n", a.c_str());
            return 2;
        }
    }
    guild::program::LoadedModel lm;
    if (const int rc = guild::program::load_model(mo, "session_score_test", lm); rc != 0)
        return rc;
    GuildSession& s = *lm.session;

    std::string text =
        "The committee met on Tuesday to review the budget. After a long discussion about the costs of the new "
        "building, the members agreed to postpone the decision until the spring, when the final estimates arrive. ";
    if (!text_path.empty()) {
        std::ifstream f(text_path);
        std::stringstream ss;
        ss << f.rdbuf();
        text = ss.str();
    }
    std::vector<int32_t> ids;
    std::string err;
    while ((int) ids.size() < n_tokens + 64) {
        std::vector<int32_t> more;
        if (!lm.model->tokenizer().encode(text, more, false, err)) {
            std::fprintf(stderr, "tokenize: %s\n", err.c_str());
            return 1;
        }
        if (more.empty())
            return 1;
        ids.insert(ids.end(), more.begin(), more.end());
    }
    const std::vector<int32_t> a(ids.begin(), ids.begin() + n_tokens);
    // b shares a's first 3/4 and then goes on with other tokens
    const int64_t common = n_tokens * 3 / 4;
    std::vector<int32_t> b(ids.begin(), ids.begin() + common);
    b.insert(b.end(), ids.begin() + n_tokens, ids.begin() + n_tokens + 48);

    // 1. score against generate's first token
    s.reset();
    ScoreRequest ra;
    ra.tokens = a;
    ScoreResult full_a;
    if (!s.score(ra, full_a, err)) {
        std::fprintf(stderr, "score: %s\n", err.c_str());
        return 1;
    }
    std::printf("score: %zu logits, %lld tokens read, %.1f ms prefill, %.1f ms window\n", full_a.logits.size(),
                (long long) full_a.read_tokens, full_a.prefill_ms, full_a.window_ms);
    s.reset();
    GenerationRequest gr;
    gr.prompt_tokens = a;
    gr.max_new_tokens = 1;
    int32_t first = -1;
    RuntimeTelemetry tel;
    if (!s.generate(
            gr,
            [&](const TokenEvent& t) {
                if (first < 0)
                    first = t.token_id;
                return true;
            },
            tel, err)) {
        std::fprintf(stderr, "generate: %s\n", err.c_str());
        return 1;
    }
    check((int32_t) argmax(full_a.logits) == first, "score's argmax is generate's first token");

    // 2. a repeat from the live state
    s.reset();
    ScoreResult again1, again2;
    if (!s.score(ra, again1, err) || !s.score(ra, again2, err))
        return 1;
    std::printf("repeat: max |dlogit| %.6f; fresh read against the first fresh read %.6f\n",
                max_abs(again1.logits, again2.logits), max_abs(again1.logits, full_a.logits));
    check(max_abs(again1.logits, full_a.logits) == 0.0, "two fresh reads give the same logits, bit for bit");
    check(max_abs(again1.logits, again2.logits) == 0.0, "a repeat gives the same logits, bit for bit");

    // 3. the shared beginning
    s.reset();
    ScoreRequest kp = ra;
    kp.keep_prefix = common;
    ScoreResult kept_a, shared_b, fresh_b;
    ScoreRequest rb;
    rb.tokens = b;
    if (!s.score(kp, kept_a, err) || !s.score(rb, shared_b, err))
        return 1;
    s.reset();
    if (!s.score(rb, fresh_b, err))
        return 1;
    const double d = max_abs(shared_b.logits, fresh_b.logits);
    std::printf("shared beginning: %lld of %zu tokens from the checkpoint; max |dlogit| against a fresh read %.4f\n",
                (long long) shared_b.cached_tokens, b.size(), d);
    check(shared_b.cached_tokens == common, "the second prompt starts from the checkpoint");
    check(argmax(shared_b.logits) == argmax(fresh_b.logits) && d <= tol, "its logits match a fresh read");
    std::printf("checkpointing on the way: max |dlogit| %.6f\n", max_abs(kept_a.logits, full_a.logits));
    check(max_abs(kept_a.logits, full_a.logits) <= tol, "checkpointing on the way leaves the first prompt's logits");

    // 4. candidates
    s.reset();
    ScoreRequest rc = ra;
    rc.candidates = {(int32_t) argmax(full_a.logits), 0, 1, 2};
    ScoreResult sub;
    if (!s.score(rc, sub, err))
        return 1;
    bool same = sub.logits.size() == rc.candidates.size();
    for (size_t i = 0; same && i < rc.candidates.size(); ++i)
        same = sub.logits[i] == full_a.logits[(size_t) rc.candidates[i]];
    check(same, "a candidate subset returns those entries of the row");

    std::printf("%s (%d failures)\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "session_score_test: %s\n", e.what());
    return 1;
}
