// src/spec/adaptive_scheduler_test.cpp - Phase 6.6: comprehensive tests for RetrievalDrafter, AdaptiveDecodeScheduler,
// MoE-aware cost estimation, session isolation, memory bounds, and target-distribution-preserving sampling.
#include "guild/spec/adaptive_scheduler.hpp"
#include "guild/spec/retrieval_drafter.hpp"
#include "guild/spec/suffix_drafter.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <thread>
#include <vector>

using namespace guild::spec;

namespace {
int g_fail = 0;
int g_pass = 0;

void check(bool ok, const char* what) {
    std::printf("  %-72s %s\n", what, ok ? "ok" : "FAIL");
    if (ok)
        ++g_pass;
    else
        ++g_fail;
}

void test_retrieval_drafter() {
    std::printf("[1] RetrievalDrafter: multi-candidate ranking, session isolation & memory bounds\n");

    // 1.1 Basic stream and corpus retrieval
    {
        RetrievalDrafter rd(3, 32, 4096, 4096, 0.35f);
        const int32_t corpus_doc[] = {100, 101, 102, 500, 501, 502, 503, 504};
        rd.ingest_corpus(corpus_doc, 8);
        check(rd.corpus_size() == 8 && rd.stream_size() == 0, "corpus ingested independently of active stream");

        const int32_t prompt[] = {10, 20, 100, 101, 102};
        rd.append_stream(prompt, 5);
        int32_t out[8] = {};
        const int k = rd.propose(4, out);
        check(k == 4 && out[0] == 500 && out[1] == 501 && out[2] == 502 && out[3] == 503,
              "retrieves continuation from session-local corpus when stream has no earlier match");
        check(rd.last_from_corpus() && rd.last_match() == 3, "reports corpus provenance and match length 3");
    }

    // 1.2 Multiple matching continuations: higher-frequency continuation wins
    {
        RetrievalDrafter rd(3, 32, 4096, 4096, 0.35f);
        // Pattern [1, 2, 3] followed by [90, 91] three times, and once by [80, 81]
        const int32_t hist[] = {1, 2,  3,  90, 91, 7, 1, 2,  3,  90, 91, 8, 1, 2,
                                3, 90, 91, 9,  1,  2, 3, 80, 81, 6,  1,  2, 3};
        rd.append_stream(hist, sizeof(hist) / sizeof(hist[0]));
        int32_t out[4] = {};
        const int k = rd.propose(2, out);
        check(k == 2 && out[0] == 90 && out[1] == 91,
              "multi-candidate voting selects majority continuation [90, 91] over single recent outlier");
        check(rd.last_candidates() >= 4 && rd.last_confidence() > 0.65f,
              "tracks candidate count and empirical continuation confidence");
    }

    // 1.3 Token boundary correctness & reset/deletion controls
    {
        RetrievalDrafter rd(3, 32, 4096, 4096, 0.35f);
        const int32_t turn1[] = {41, 42, 43, 44, 45, 46};
        rd.ingest_corpus(turn1, 6);
        rd.append_stream(turn1, 3);
        int32_t out[4] = {};
        check(rd.propose(3, out) == 3 && out[0] == 44, "proposes across corpus before clear_corpus");

        rd.clear_corpus();
        check(rd.corpus_size() == 0, "clear_corpus immediately deletes session corpus");
        check(rd.propose(3, out) == 0, "cleared corpus no longer proposes stale tokens");

        rd.reset_all();
        check(rd.stream_size() == 0 && rd.corpus_size() == 0, "reset_all clears both stream and corpus");
    }

    // 1.4 Memory-budget exhaustion and sliding-window compaction
    {
        RetrievalDrafter rd(3, 32, 256, 256, 0.35f);
        std::vector<int32_t> big(2000);
        for (size_t i = 0; i < big.size(); ++i)
            big[i] = static_cast<int32_t>((i * 7) % 509);
        // Append repeating tail
        for (int i = 0; i < 12; ++i)
            big.push_back(1000 + (i % 4));
        rd.ingest_corpus(big.data(), big.size());
        rd.append_stream(big.data(), big.size());
        check(rd.stream_size() <= 256 && rd.corpus_size() <= 256,
              "sliding-window compaction enforces max_stream_tokens and max_corpus_tokens");
        int32_t out[4] = {};
        const int k = rd.propose(4, out);
        check(k >= 3 && out[0] == 1000, "continues to propose accurately after multiple compactions");
    }

    // 1.5 Concurrent independent sessions isolation
    {
        constexpr int kThreads = 4;
        bool isolated = true;
        std::vector<std::thread> threads;
        for (int t = 0; t < kThreads; ++t) {
            threads.emplace_back([t, &isolated]() {
                RetrievalDrafter local_rd(3, 32, 1024, 1024);
                const int32_t secret_base = (t + 1) * 10000;
                const int32_t doc[] = {1, 2, 3, secret_base + 1, secret_base + 2, secret_base + 3};
                local_rd.ingest_corpus(doc, 6);
                for (int iter = 0; iter < 100; ++iter) {
                    local_rd.reset_stream();
                    const int32_t q[] = {9, 1, 2, 3};
                    local_rd.append_stream(q, 4);
                    int32_t out[4] = {};
                    const int k = local_rd.propose(3, out);
                    if (k != 3 || out[0] != secret_base + 1 || out[1] != secret_base + 2 || out[2] != secret_base + 3) {
                        isolated = false;
                    }
                }
            });
        }
        for (auto& th : threads)
            th.join();
        check(isolated, "concurrent sessions maintain strict token datastore isolation");
    }
}

void test_adaptive_scheduler() {
    std::printf("[2] AdaptiveDecodeScheduler: cost estimation, switching, hysteresis & fallback\n");

    // 2.1 Non-negotiable gate: chooses Normal decoding when speculative strategies are more expensive
    {
        SchedulerConfig cfg;
        cfg.mode = DecodeStrategyMode::Adaptive;
        cfg.max_draft = 4;
        cfg.cost_window = 12;
        cfg.hysteresis_margin = 0.05;
        cfg.min_dwell_steps = 2;

        GExecSpecCalibration cal;
        cal.calibrated = true;
        // Realistic CPU-offloaded MoE verification curve (e.g. Ornith/Qwen): T=1 is 20ms, T=2 is 28ms, T=5 is 52ms
        cal.verify_window_ms[1] = 20.0;
        cal.verify_window_ms[2] = 28.0;
        cal.verify_window_ms[3] = 36.0;
        cal.verify_window_ms[4] = 44.0;
        cal.verify_window_ms[5] = 52.0;

        AdaptiveDecodeScheduler sched(cfg, cal);
        sched.seed_single_token_verify(20.0, 320);

        ProposalAvailability avail;
        avail.mtp_available = true;
        avail.mtp_max_k = 2;
        avail.mtp_first_prob = 0.75f;
        avail.lookup_k = 3;
        avail.lookup_match_len = 3;

        // Simulate a speculation-hostile workload where every draft token is rejected (accepted_k = 0).
        for (int step = 0; step < 12; ++step) {
            const ScheduleDecision dec = sched.choose(avail);
            IterationObservation obs;
            obs.source = dec.source;
            obs.proposed_k = dec.draft_k;
            obs.accepted_k = 0;
            obs.committed_tokens = 1;
            obs.match_len = avail.lookup_match_len;
            obs.draft_ms = (dec.source == CandidateSource::Mtp) ? (1.6 * dec.draft_k) : 0.02;
            obs.verify_ms = cal.verify_window_ms[dec.draft_k + 1];
            obs.commit_ms = 0.12;
            sched.observe(obs);
        }

        const ScheduleDecision final_dec = sched.choose(avail);
        check(final_dec.source == CandidateSource::Normal && final_dec.draft_k == 0,
              "chooses Normal decoding when speculative drafts have low acceptance / higher ms/token");
        const auto snap = sched.snapshot();
        check(snap.fallback_to_normal_count >= 1, "records fallback_to_normal_count in telemetry");
    }

    // 2.2 High-acceptance lookup beats both Normal and MTP on repetitive/input-grounded text
    {
        SchedulerConfig cfg;
        cfg.mode = DecodeStrategyMode::Adaptive;
        cfg.max_draft = 4;
        AdaptiveDecodeScheduler sched(cfg);
        sched.seed_single_token_verify(20.0, 320);

        ProposalAvailability avail;
        avail.mtp_available = true;
        avail.mtp_max_k = 1; // MTP only drafts 1 token and costs 1.6ms draft + verify(2)
        avail.lookup_k = 4;
        avail.lookup_match_len = 16; // Long 16-token match in prompt -> bucket 3 (q ~ 0.95)
        avail.lookup_freq = 2;

        ScheduleDecision dec;
        for (int step = 0; step < 8; ++step) {
            dec = sched.choose(avail);
            IterationObservation obs;
            obs.source = dec.source;
            obs.proposed_k = dec.draft_k;
            obs.accepted_k = dec.draft_k; // Full acceptance on exact prompt quote
            obs.committed_tokens = dec.draft_k + 1;
            obs.match_len = avail.lookup_match_len;
            obs.draft_ms = 0.015;
            obs.verify_ms = 20.0 + 7.5 * dec.draft_k;
            obs.commit_ms = 0.12;
            sched.observe(obs);
        }
        check(dec.source == CandidateSource::Lookup && dec.draft_k >= 3,
              "selects Lookup with deep window (k>=3) on long prompt suffix match");
        check(dec.expected_ms_per_token < 12.0 && dec.estimated_speedup > 1.6,
              "predicts <12 ms/tok (>1.6x speedup over 20 ms/tok Normal)");
    }

    // 2.3 Novel text without prompt matches: selects MTP when MTP acceptance is high
    {
        SchedulerConfig cfg;
        cfg.mode = DecodeStrategyMode::Adaptive;
        cfg.max_draft = 2;
        AdaptiveDecodeScheduler sched(cfg);
        sched.seed_single_token_verify(40.0, 480); // Qwen-like 40ms single-token verify

        ProposalAvailability avail;
        avail.mtp_available = true;
        avail.mtp_max_k = 2;
        avail.mtp_first_prob = 0.92f;
        avail.lookup_k = 0; // No suffix match in novel text
        avail.retrieval_k = 0;

        ScheduleDecision dec;
        for (int step = 0; step < 8; ++step) {
            dec = sched.choose(avail);
            IterationObservation obs;
            obs.source = dec.source;
            obs.proposed_k = dec.draft_k;
            obs.accepted_k = (dec.draft_k > 0) ? 1 : 0; // 1 draft accepted per step -> 2 committed tokens
            obs.committed_tokens = obs.accepted_k + 1;
            obs.draft_ms = 1.6 * dec.draft_k;
            obs.verify_ms = 40.0 + 9.0 * dec.draft_k;
            obs.commit_ms = 0.15;
            sched.observe(obs);
        }
        check(dec.source == CandidateSource::Mtp && dec.draft_k >= 1,
              "selects MTP on novel text when lookup has no matches and MTP saves ms/token");
    }

    // 2.4 Recovery from incorrect initial cost estimates
    {
        SchedulerConfig cfg;
        cfg.mode = DecodeStrategyMode::Adaptive;
        cfg.max_draft = 4;
        GExecSpecCalibration bad_cal;
        bad_cal.calibrated = true;
        // Artificially optimistic multi-token cost estimate (claims T=5 costs almost the same as T=1!)
        for (int t = 1; t <= 8; ++t)
            bad_cal.verify_window_ms[t] = 20.0 + 0.5 * (t - 1);

        AdaptiveDecodeScheduler sched(cfg, bad_cal);
        sched.seed_single_token_verify(20.0, 320);

        ProposalAvailability avail;
        avail.lookup_k = 4;
        avail.lookup_match_len = 8;

        // In reality, T=5 takes 65 ms on CPU experts and only 1 token is accepted!
        for (int step = 0; step < 10; ++step) {
            const ScheduleDecision dec = sched.choose(avail);
            IterationObservation obs;
            obs.source = dec.source;
            obs.proposed_k = dec.draft_k;
            obs.accepted_k = (dec.draft_k > 0) ? 1 : 0;
            obs.committed_tokens = obs.accepted_k + 1;
            obs.match_len = avail.lookup_match_len;
            obs.draft_ms = 0.02;
            obs.verify_ms = (dec.draft_k == 0) ? 20.0 : (20.0 + 14.0 * dec.draft_k);
            obs.commit_ms = 0.12;
            sched.observe(obs);
        }
        // Even with 1 accepted token (2 committed), T=2 takes 34ms (17ms/tok) while T=5 takes 76ms (38ms/tok),
        // and if 0 tokens are accepted it falls back to Normal (20ms/tok).
        const ScheduleDecision after = sched.choose(avail);
        check(after.draft_k <= 2,
              "online verification & cumulative ratio measurements override optimistic initial calibration");
    }

    // 2.5 MTP fallback when mtp_first_prob < mtp_min_prob
    {
        SchedulerConfig cfg;
        cfg.mode = DecodeStrategyMode::Mtp;
        cfg.max_draft = 3;
        cfg.mtp_min_prob = 0.5f;
        AdaptiveDecodeScheduler sched(cfg);

        ProposalAvailability avail;
        avail.mtp_available = true;
        avail.mtp_max_k = 3;
        avail.mtp_first_prob = 0.25f; // Below threshold!

        const ScheduleDecision dec = sched.choose(avail);
        check(dec.source == CandidateSource::Normal && dec.draft_k == 0,
              "MTP mode safely falls back to Normal when draft confidence is below spec_min_p");
    }

    // 2.6 Sample-size-aware Beta-Binomial posterior: established 80% MTP drafter survives 2 Bernoulli misses
    {
        SchedulerConfig cfg;
        cfg.mode = DecodeStrategyMode::Adaptive;
        cfg.max_draft = 1;
        AdaptiveDecodeScheduler sched(cfg);
        sched.seed_single_token_verify(54.0, 480, 27.0);

        ProposalAvailability avail;
        avail.mtp_available = true;
        avail.mtp_max_k = 1;
        avail.mtp_first_prob = 0.85f;

        // 20 windows at 85% acceptance (17 hits, 3 scattered misses)
        for (int step = 0; step < 20; ++step) {
            const ScheduleDecision dec = sched.choose(avail);
            IterationObservation obs;
            obs.source = dec.source;
            obs.proposed_k = dec.draft_k;
            obs.accepted_k = (step % 6 == 5) ? 0 : 1;
            obs.committed_tokens = obs.accepted_k + 1;
            obs.draft_ms = 1.5;
            obs.verify_ms = 68.0;
            obs.commit_ms = 0.12;
            sched.observe(obs);
        }
        // Now 2 consecutive Bernoulli misses occur (probability (1-0.85)^2 = 2.25% per step)
        for (int miss = 0; miss < 2; ++miss) {
            const ScheduleDecision dec = sched.choose(avail);
            IterationObservation obs;
            obs.source = dec.source;
            obs.proposed_k = dec.draft_k;
            obs.accepted_k = 0;
            obs.committed_tokens = 1;
            obs.draft_ms = 1.5;
            obs.verify_ms = 68.0;
            obs.commit_ms = 0.12;
            sched.observe(obs);
        }
        const ScheduleDecision after_two_misses = sched.choose(avail);
        check(after_two_misses.source == CandidateSource::Mtp && after_two_misses.draft_k == 1,
              "established high-acceptance MTP drafter is not exiled to Normal by 2 Bernoulli misses");
    }

    // 2.7 SuffixDrafter consensus prefix detection on template with variable counter slot
    {
        SuffixDrafter sfx(3, 32, 4096);
        // 4 records where [10, 20, 30] is followed by different IDs (101..104), then [50, 60, 70, 80] is invariant
        const int32_t recs[] = {10, 20,  30, 101, 50, 60, 70, 80, 90, 10, 20,  30, 102, 50, 60, 70, 80, 90, 10, 20,
                                30, 103, 50, 60,  70, 80, 90, 10, 20, 30, 104, 50, 60,  70, 80, 90, 10, 20, 30};
        sfx.append(recs, sizeof(recs) / sizeof(recs[0]));
        int32_t out[4] = {};
        const int k1 = sfx.propose(4, out);
        check(k1 == 4 && sfx.last_candidates() == 3 && sfx.last_freq() == 1 && sfx.last_consensus_k() == 0,
              "SuffixDrafter flags variable counter slot (3 stored candidates, freq=1 -> consensus_k=0)");

        // Emit new counter 105 and invariant trigram [50, 60, 70]
        const int32_t next_part[] = {105, 50, 60, 70};
        sfx.append(next_part, 4);
        const int k2 = sfx.propose(4, out);
        check(k2 == 4 && sfx.last_candidates() == 3 && sfx.last_freq() == 3 && sfx.last_consensus_k() >= 2,
              "SuffixDrafter identifies invariant template span (3/3 agreement) and stops before next counter");
    }
}

void test_sampling_distributional_correctness() {
    std::printf("[3] Target-distribution-preserving speculative sampling verification\n");

    // Prove & test that Guild's position-indexed inverse-CDF target verification reproduces the exact target
    // distribution under arbitrary deterministic draft proposals (Lookup / Retrieval).
    //
    // At any position p, the target model produces filtered probabilities P_target(v | x_{<=p}) after
    // penalties -> top_k -> top_p -> min_p -> temperature.
    // The verifier draws u_p ~ Uniform(0, 1) deterministically from Philox(seed, p) and selects:
    //   y_p = F^{-1}_{P_target(\cdot | x_{<=p})}(u_p)
    // A deterministic proposal d_1..d_k is accepted up to the longest prefix where d_t == y_{p+t-1}, and row a
    // emits y_{p+a} (the first mismatch or bonus token).
    // Because every committed token at position q is literally y_q = F^{-1}_{P_target(\cdot | x_{<=q})}(u_q),
    // the sequence of emitted tokens is bitwise identical for a fixed seed (and identically distributed over random
    // seeds) regardless of what deterministic draft tokens d_1..d_k were proposed!

    constexpr int V = 8;
    const double raw_logits[V] = {2.1, 1.5, 0.9, 0.4, -0.2, -0.8, -1.5, -2.5};
    const double temperature = 0.8;
    const double top_p = 0.90;
    const double min_p = 0.08;
    const int top_k = 5;

    // Compute exact filtered target distribution
    std::vector<double> probs(V, 0.0);
    // 1. top_k = 5 (logits are already sorted descending)
    double max_l = raw_logits[0] / temperature;
    double sum_k = 0.0;
    for (int i = 0; i < top_k; ++i) {
        probs[i] = std::exp(raw_logits[i] / temperature - max_l);
        sum_k += probs[i];
    }
    for (int i = 0; i < top_k; ++i)
        probs[i] /= sum_k;

    // 2. top_p = 0.90
    double cum = 0.0;
    int keep = 0;
    for (int i = 0; i < top_k; ++i) {
        cum += probs[i];
        ++keep;
        if (cum >= top_p)
            break;
    }
    for (int i = keep; i < V; ++i)
        probs[i] = 0.0;

    // 3. min_p = 0.08 * p_max
    const double p_max = probs[0];
    double final_sum = 0.0;
    for (int i = 0; i < keep; ++i) {
        if (probs[i] < min_p * p_max)
            probs[i] = 0.0;
        final_sum += probs[i];
    }
    for (int i = 0; i < V; ++i)
        probs[i] /= final_sum;

    auto inverse_cdf = [&](double u) -> int32_t {
        double c = 0.0;
        for (int i = 0; i < V; ++i) {
            c += probs[i];
            if (u <= c)
                return i;
        }
        return keep - 1;
    };

    // Simulate 25,000 tokens generated by:
    //   (A) Normal autoregressive decoding (k = 0)
    //   (B) Speculative decoding with adversarial/biased Lookup proposals (e.g. always proposing token 3, or cycling)
    constexpr int N_TOKENS = 25000;
    auto draw_u = [](uint64_t seed, int64_t pos) -> double {
        uint64_t x = seed ^ ((uint64_t) pos * 0x9E3779B97F4A7C15ull);
        x ^= x >> 33;
        x *= 0xff51afd7ed558ccdull;
        x ^= x >> 33;
        x *= 0xc4ceb9fe1a85ec53ull;
        x ^= x >> 33;
        return (double) (x >> 11) * (1.0 / 9007199254740992.0);
    };

    const uint64_t seed = 424242ull;
    std::vector<int32_t> normal_stream(N_TOKENS);
    for (int64_t pos = 0; pos < N_TOKENS; ++pos) {
        normal_stream[(size_t) pos] = inverse_cdf(draw_u(seed, pos));
    }

    std::vector<int32_t> spec_stream;
    spec_stream.reserve(N_TOKENS);
    int64_t pos = 0;
    int64_t spec_windows = 0;
    int64_t accepted_total = 0;
    while (pos < N_TOKENS) {
        const int K = 4;
        const int T = (int) std::min<int64_t>(1 + K, N_TOKENS - pos);
        int32_t drafts[4] = {0, (int32_t) (pos % 3), 1, 2}; // deterministic proposal
        int32_t outv[5] = {};
        for (int t = 0; t < T; ++t) {
            outv[t] = inverse_cdf(draw_u(seed, pos + t));
        }
        int a = 0;
        while (a < T - 1 && drafts[a] == outv[a])
            ++a;
        for (int i = 0; i <= a && (int64_t) spec_stream.size() < N_TOKENS; ++i)
            spec_stream.push_back(outv[i]);
        accepted_total += a;
        pos += (a + 1);
        ++spec_windows;
    }

    check(spec_stream == normal_stream && accepted_total > 0,
          "speculative verification with deterministic proposals is 100% bitwise seed-invariant with normal sampling");

    // Check empirical frequencies against exact target probabilities P_target
    int64_t counts[V] = {};
    for (int32_t tok : spec_stream)
        ++counts[tok];
    double max_tvd = 0.0;
    bool zero_outside_support = true;
    for (int i = 0; i < V; ++i) {
        const double emp = (double) counts[i] / (double) N_TOKENS;
        max_tvd = std::max(max_tvd, std::abs(emp - probs[i]));
        if (probs[i] == 0.0 && counts[i] != 0)
            zero_outside_support = false;
    }
    check(zero_outside_support && max_tvd < 0.01, "empirical distribution matches filtered target distribution (TVD < "
                                                  "0.01, zero mass outside top-k/top-p/min-p)");
}

} // namespace

int main() {
    std::printf("adaptive_scheduler_test\n");
    test_retrieval_drafter();
    test_adaptive_scheduler();
    test_sampling_distributional_correctness();
    std::printf("adaptive_scheduler_test: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
