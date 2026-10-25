// src/runtime/glm5.cpp - GLM-5.3-Flash behind GuildModel / GuildSession (src/core/glm5.cpp runs the model).
//
// The model has its own engine, so loading it skips the pack, the runner registry and the verifier: GuildModel::load
// hands a glm5-next GGUF to glm5_load, and GuildSession::generate hands its requests to glm5_generate.
//
// Its linear-attention layers carry a recurrent state that cannot be rewound to a shorter context.  So a request
// continues the live context when its prompt extends it, or restores the snapshot taken after the previous prompt
// when it extends that (a chat's next turn whose rendering of the last answer differs from the tokens generated),
// and prefills from the start otherwise.
#include "guild/core/glm5.hpp"
#include "guild/core/glm5_kernels.hpp"
#include "guild/runtime/model.hpp"
#include "guild/runtime/session.hpp"
#include "guild/runtime/tokenizer.hpp"
#include "guild/artifact/gguf_reader.hpp"
#include "model_impl.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>

namespace guild::runtime {
namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

size_t common_prefix(const std::vector<int32_t>& a, const std::vector<int32_t>& b) {
    size_t n = 0;
    while (n < a.size() && n < b.size() && a[n] == b[n])
        ++n;
    return n;
}

/// One token from the logits (modified in place): penalties, then greedy or temperature / top-k / min-p / top-p
int32_t sample(std::vector<float>& logits, const GenerationRequest& req, const std::vector<int32_t>& history,
               std::mt19937_64& rng) {
    const int V = (int) logits.size();
    if (req.penalty_last_n > 0 &&
        (req.penalty_repeat != 1.f || req.penalty_freq != 0.f || req.penalty_present != 0.f)) {
        std::vector<std::pair<int32_t, int>> counts;
        const size_t from = history.size() > (size_t) req.penalty_last_n ? history.size() - req.penalty_last_n : 0;
        std::vector<int32_t> recent(history.begin() + (std::ptrdiff_t) from, history.end());
        std::sort(recent.begin(), recent.end());
        for (size_t i = 0; i < recent.size();) {
            size_t j = i;
            while (j < recent.size() && recent[j] == recent[i])
                ++j;
            const int32_t id = recent[i];
            if (id >= 0 && id < V) {
                float& l = logits[(size_t) id];
                l = l > 0 ? l / req.penalty_repeat : l * req.penalty_repeat;
                l -= (float) (j - i) * req.penalty_freq + req.penalty_present;
            }
            i = j;
        }
    }
    const int32_t best = (int32_t) (std::max_element(logits.begin(), logits.end()) - logits.begin());
    if (req.temperature <= 0.f)
        return best;
    // candidates within 30 nats of the best at this temperature: the rest weighs below 1e-13 each
    const float maxl = logits[(size_t) best];
    const float cut = maxl - 30.f * req.temperature;
    std::vector<std::pair<float, int32_t>> c;
    c.reserve(4096);
    for (int i = 0; i < V; ++i)
        if (logits[(size_t) i] >= cut)
            c.emplace_back(logits[(size_t) i], i);
    std::sort(c.begin(), c.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    if (req.top_k > 0 && (size_t) req.top_k < c.size())
        c.resize((size_t) req.top_k);
    std::vector<double> p(c.size());
    double sum = 0;
    for (size_t i = 0; i < c.size(); ++i)
        sum += (p[i] = std::exp((double) (c[i].first - maxl) / req.temperature));
    size_t keep = c.size();
    if (req.min_p > 0.f) {
        const double floor = req.min_p * p[0];
        while (keep > 1 && p[keep - 1] < floor)
            --keep;
    }
    if (req.top_p < 1.f) {
        double acc = 0, total = std::accumulate(p.begin(), p.begin() + (std::ptrdiff_t) keep, 0.0);
        for (size_t i = 0; i < keep; ++i) {
            acc += p[i];
            if (acc >= req.top_p * total) {
                keep = i + 1;
                break;
            }
        }
    }
    double total = std::accumulate(p.begin(), p.begin() + (std::ptrdiff_t) keep, 0.0);
    double r = std::uniform_real_distribution<double>(0.0, total)(rng);
    for (size_t i = 0; i < keep; ++i) {
        r -= p[i];
        if (r <= 0)
            return c[i].second;
    }
    (void) sum;
    return c[keep - 1].second;
}

} // namespace

bool glm5_is_model(const std::string& gguf_path) {
    try {
        guild::GgufFile f(gguf_path);
        const MetaValue* a = f.get("general.architecture");
        return a && (a->s == "glm5-next" || a->s == "glm5next");
    } catch (...) {
        return false;
    }
}

bool glm5_load(GuildModel::Impl& m, const ModelPaths& paths, const memory::ExecutionPlan& plan, std::string& err) {
    core::Glm5Options o;
    o.max_context = plan.context_length > 0 ? plan.context_length : o.max_context;
    o.threads = plan.pool_workers;
    if (const char* d = std::getenv("GUILD_GLM_DENSE")) // auto (the default), q8_0 or q6_k
        o.dense = std::string(d) == "q6_k"   ? guild::core::Glm5Dense::Q6_K
                  : std::string(d) == "q8_0" ? guild::core::Glm5Dense::Q8_0
                                             : guild::core::Glm5Dense::Auto;
    if (const char* n = std::getenv("GUILD_GLM_MTP"))
        o.mtp = std::string(n) != "0";
    m.glm = std::make_unique<core::Glm5Engine>();
    if (!m.glm->load(paths.primary_model_path, o, err))
        return false;
    std::fprintf(stderr, "guild %s", m.glm->placement().c_str());
    m.config.max_context = m.glm->max_context();
    m.config.eos_token_ids.clear();
    for (int32_t id : m.glm->geometry().eos_ids)
        m.config.eos_token_ids.push_back(id);
    m.config.decode_strategy = m.glm->has_mtp() ? "mtp" : "normal";
    m.config.spec = m.glm->has_mtp() ? 2 : 1;
    return true;
}

bool glm5_generate(GuildModel::Impl& m, const GenerationRequest& req, const std::vector<int64_t>& prompt64,
                   TokenStreamCallback on_token, RuntimeTelemetry& tel, std::string& err,
                   PrefillProgressCallback on_prefill, std::atomic<bool>& cancelled) {
    core::Glm5Engine& eng = *m.glm;
    const int64_t V = eng.geometry().n_vocab;
    std::vector<int32_t> prompt(prompt64.begin(), prompt64.end());
    if ((int64_t) prompt.size() + 1 > eng.max_context()) {
        err = "prompt of " + std::to_string(prompt.size()) + " tokens does not fit the context of " +
              std::to_string(eng.max_context());
        return false;
    }
    const auto t0 = Clock::now();
    auto cancelled_now = [&] { return cancelled.load() || (req.cancel_flag && req.cancel_flag->load()); };

    // where to start: the live context, the snapshot (after the last prompt's <|assistant|>), or nothing
    std::vector<float> logits((size_t) V);
    size_t start = 0;
    const size_t live = common_prefix(m.glm_ctx, prompt), snap = common_prefix(m.glm_snap, prompt);
    if (live == m.glm_ctx.size() && live > 0 && (live < prompt.size() || !m.glm_ctx_logits.empty())) {
        start = live;
    } else if (snap == m.glm_snap.size() && snap > 0 && (snap < prompt.size() || !m.glm_snap_logits.empty())) {
        if (!eng.restore(err))
            return false;
        start = snap;
        m.glm_ctx = m.glm_snap;
        m.glm_ctx_logits = m.glm_snap_logits;
    } else {
        eng.reset();
        m.glm_ctx.clear();
        m.glm_ctx_logits.clear();
        // the snapshot's K/V rows are about to be overwritten from 0: it no longer describes anything
        m.glm_snap.clear();
        m.glm_snap_logits.clear();
    }
    tel.cached_prompt_tokens = (int) start;
    tel.prompt_tokens = (int) prompt.size();

    // The snapshot for the next turn is taken just after the prompt's last <|assistant|>: the next prompt renders this
    // turn's reply after it (with or without the reasoning a client sends back), while the generation prompt's own
    // tail (<think>) differs from the reply's - a snapshot at the prompt's end would rarely be a prefix again.
    if (m.glm_turn_id == -2) {
        std::vector<int32_t> ids;
        std::string e;
        m.glm_turn_id = m.tokenizer && m.tokenizer->is_loaded() && m.tokenizer->encode("<|assistant|>", ids, true, e) &&
                                ids.size() == 1
                            ? ids[0]
                            : -1;
    }
    size_t cut = prompt.size();
    if (m.glm_turn_id >= 0) {
        const auto it = std::find(prompt.rbegin(), prompt.rend(), m.glm_turn_id);
        if (it != prompt.rend())
            cut = (size_t) (prompt.rend() - it); // one past it
    }
    bool snapped = false;
    auto snap_here = [&](size_t at, const std::vector<float>* at_logits) {
        if (!eng.snapshot(err))
            return false;
        m.glm_snap.assign(prompt.begin(), prompt.begin() + (std::ptrdiff_t) at);
        m.glm_snap_logits = at_logits ? *at_logits : std::vector<float>();
        snapped = true;
        return true;
    };

    // prefill the rest, in windows (one ends at the cut), reporting progress
    if (start < prompt.size()) {
        constexpr int kChunk = core::glm5k::kMaxT; // the engine's widest window
        for (size_t at = start; at < prompt.size();) {
            if (cancelled_now()) { // the state matches glm_ctx; its logits are unknown, so the next request feeds more
                tel.finish_reason = "cancel";
                tel.prompt_ms = ms_since(t0);
                return true;
            }
            const size_t end = at < cut ? cut : prompt.size();
            const int n = (int) std::min<size_t>(kChunk, end - at);
            const bool last = at + (size_t) n == prompt.size();
            if (!eng.forward(&prompt[at], n, (int64_t) at, last ? logits.data() : nullptr, err)) {
                m.ready->store(false);
                return false;
            }
            at += (size_t) n;
            m.glm_ctx.assign(prompt.begin(), prompt.begin() + (std::ptrdiff_t) at);
            if (at == cut && !snap_here(at, last ? &logits : nullptr))
                return false;
            if (on_prefill) {
                PrefillProgressEvent ev;
                ev.pos = (int64_t) at;
                ev.total = (int64_t) prompt.size();
                ev.ms = ms_since(t0);
                ev.tok_s = (double) (at - start) * 1000.0 / std::max(1e-3, ev.ms);
                on_prefill(ev);
            }
        }
    } else {
        logits = m.glm_ctx_logits;
    }
    m.glm_ctx_logits.clear();
    // no cut in what this prompt fed (the reuse started past it): the snapshot held stays (it is a prefix of the live
    // context, so still valid); with none, the state after the prompt
    if (!snapped && m.glm_snap.empty() && !snap_here(prompt.size(), &logits))
        return false;
    tel.prompt_ms = ms_since(t0);
    tel.prompt_tok_s = (double) (prompt.size() - start) * 1000.0 / std::max(1e-3, tel.prompt_ms);
    tel.ttft_ms = tel.prompt_ms;

    // decode: the sampled token waits in `tid` (emitted, not fed).  With the NextN block a step drafts the token
    // after it and verifies both in one window: the main model's own sample over the first row decides, so the
    // output is what plain decoding would sample, and an equal draft gives the second row's sample for free.
    std::mt19937_64 rng(req.seed ? req.seed : (uint64_t) std::random_device{}());
    std::unique_ptr<IncrementalDecoder> decoder;
    if (m.tokenizer && m.tokenizer->is_loaded())
        decoder = std::make_unique<IncrementalDecoder>(m.tokenizer.get(), true);
    const bool spec = eng.has_mtp() && req.decode_strategy != "normal" && m.config.decode_strategy != "normal";
    std::vector<float> rows(spec ? (size_t) 2 * V : 0);
    std::string text;
    std::vector<int32_t> history = prompt;
    const auto t1 = Clock::now();
    bool eos = false, aborted = false;
    int produced = 0;
    // emits one token; false ends the generation (eos, a stop string, the length, the client)
    auto emit = [&](int32_t tid) {
        eos = std::find(m.config.eos_token_ids.begin(), m.config.eos_token_ids.end(), (int64_t) tid) !=
              m.config.eos_token_ids.end();
        std::string piece = (!eos && decoder) ? decoder->add(tid) : std::string();
        text += piece;
        for (const auto& stop : req.stop)
            if (!stop.empty() && text.size() >= stop.size() &&
                text.compare(text.size() - stop.size(), stop.size(), stop) == 0) {
                eos = true;
                break;
            }
        ++produced;
        history.push_back(tid);
        TokenEvent te{tid, piece, eos};
        if (on_token && !on_token(te)) {
            aborted = true;
            return false;
        }
        return !eos && produced < req.max_new_tokens;
    };
    int32_t tid = sample(logits, req, history, rng);
    while (true) {
        if (cancelled_now()) {
            aborted = true;
            break;
        }
        if (!emit(tid))
            break;
        const int64_t pos = (int64_t) m.glm_ctx.size();
        if (pos + 1 >= eng.max_context())
            break; // the context is full: finish with "length"
        if (spec && pos + 2 < eng.max_context()) {
            int32_t d = 0;
            if (!eng.draft(tid, pos, d, err)) {
                m.ready->store(false);
                return false;
            }
            const int32_t win[2] = {tid, d};
            if (!eng.verify(win, 2, pos, rows.data(), err)) {
                m.ready->store(false);
                return false;
            }
            std::vector<float> row0(rows.begin(), rows.begin() + V);
            const int32_t y1 = sample(row0, req, history, rng);
            ++tel.drafts_offered;
            if (y1 == d) {
                ++tel.drafts_accepted;
                if (!eng.accept(2, err)) {
                    m.ready->store(false);
                    return false;
                }
                m.glm_ctx.push_back(tid);
                m.glm_ctx.push_back(d);
                if (cancelled_now()) {
                    aborted = true;
                    break;
                }
                if (!emit(d))
                    break;
                std::vector<float> row1(rows.begin() + V, rows.end());
                tid = sample(row1, req, history, rng);
            } else {
                if (!eng.accept(1, err)) {
                    m.ready->store(false);
                    return false;
                }
                m.glm_ctx.push_back(tid);
                tid = y1;
            }
            continue;
        }
        if (!eng.forward(&tid, 1, pos, logits.data(), err)) {
            m.ready->store(false);
            return false;
        }
        m.glm_ctx.push_back(tid);
        tid = sample(logits, req, history, rng);
    }
    // the live context holds every token fed; the last sampled token was not fed (its logits do not exist yet)
    tel.completion_tokens = produced;
    tel.decode_ms = ms_since(t1);
    tel.decode_tok_s = produced > 1 ? (produced - 1) * 1000.0 / std::max(1e-3, tel.decode_ms) : 0.0;
    tel.finish_reason = aborted ? "cancel" : eos ? "stop" : "length";
    return true;
}

} // namespace guild::runtime
