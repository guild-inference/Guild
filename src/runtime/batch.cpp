// src/runtime/batch.cpp - GuildBatch: several conversations in the batch windows (include/guild/runtime/batch.hpp)
#include "guild/runtime/batch.hpp"
#include "guild/runtime/model.hpp"
#include "guild/runtime/session.hpp"

#if defined(GUILD_ENABLE_CUDA) || defined(GUILD_ENABLE_HIP)

#include "model_impl.hpp"
#include "guild/core/conversation_snapshot.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <mutex>

namespace guild::runtime {

namespace {

double ms_between(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

} // namespace

struct GuildBatch::Impl {
    GuildSession* session = nullptr;
    GuildModel::Impl* m = nullptr;
    struct Slot {
        bool active = false, stop = false;
        int32_t x = 0; ///< the token the next window feeds (not yet in the slot's state)
        int64_t p = 0; ///< its position
        int64_t produced = 0, max_new = 0;
        Clock::time_point t0;
        /// what the slot's sessions hold: the prompt, then every token a window fed it.  Kept when the request ends,
        /// so its conversation's next turn (or the same request on the solo path) continues from it.
        std::vector<int32_t> ids;
        int64_t drafts_offered = 0, drafts_accepted = 0;
    };
    std::vector<Slot> slots;
    std::string note;
    // Drafts in the batch windows: each slot's own context in the DFlash drafter (set 1 + slot), kept current every
    // window; while at most `draft_slots` slots decode, a window's `draft_rows` rows are shared out among them as
    // evenly as fits, a slot's rows its token and its drafts (GUILD_BATCH_DRAFT_SLOTS, GUILD_BATCH_ROWS; 0: none)
    bool dflash_sets = false;
    int draft_slots = 0, draft_rows = 0;
    // the drafting slots' blocks through the drafter's layers one after another, then one pass of its head over all
    // their rows and one wait (DFlashDrafter::draft_slots: the same drafts, bit for bit), else a draft() a slot
    // (GUILD_BATCH_DFLASH_SHARED_HEAD=0)
    bool shared_head = true;
    // the windows since the slots were last all idle (take_report)
    int64_t windows = 0, rows = 0, tokens = 0, offered = 0, accepted = 0;
    double run_ms = 0, commit_ms = 0, draft_ms = 0, pool0 = 0, wait0 = 0;
    int64_t miss0 = 0;
    Clock::time_point start;

    bool ready() const {
        return m != nullptr && m->ready && m->ready->load();
    }

    /// The DFlash drafter back on the solo path's context (set 0), whatever a batch function used
    struct MainContext {
        core::DFlashDrafter* d;
        ~MainContext() {
            if (d != nullptr)
                d->use_context(0);
        }
    };

    // `from` -> `to`, the first n tokens' state.  Nothing may be writing either: the main session's last commit may
    // still be in flight (the solo path does not wait for it), and the copies run on the legacy stream.
    bool copy(const core::SessionState& from, core::SessionState& to, int64_t n, std::string& err) {
        if (!m->ver->wait_commit(err))
            return false;
        if (cudaDeviceSynchronize() != cudaSuccess) {
            err = "batch: the device failed before a slot copy";
            return false;
        }
        // (with the indexer's pooled keys where the attention has an indexer; Ornith's dense attention has none)
        const bool index = m->g.idx_key_dim > 0 && m->g.idx_q_heads > 0;
        for (int64_t j = 0; j < from.qsa_alloc; ++j)
            if (!core::conversation_kv_copy(from.qsa_states[from.qsa_ord0 + j], to.qsa_states[to.qsa_ord0 + j], m->g, n,
                                            index, err))
                return false;
        if (!core::conversation_state_copy(from, to, m->g, (size_t) n, err))
            return false;
        if (cudaDeviceSynchronize() != cudaSuccess) {
            err = "batch: a slot copy failed";
            return false;
        }
        return true;
    }

    /// The drafter's context of the session (set 0) into slot b's (set 1 + b), when it ends where the slot's
    /// conversation does; else the slot's context starts empty there.  `to_slot` false: the other way round.
    bool copy_dflash(int b, int64_t end, bool to_slot, std::string& err) {
        if (!dflash_sets)
            return true;
        core::DFlashDrafter& d = *m->dflash;
        MainContext back{&d};
        d.use_context(to_slot ? 0 : 1 + b);
        const bool whole = d.context_end() == end;
        if (whole) {
            if (!d.copy_context(to_slot ? 0 : 1 + b, to_slot ? 1 + b : 0, m->dflash_stream, err))
                return false;
        } else {
            d.use_context(to_slot ? 1 + b : 0);
            d.reset(end);
        }
        if (cudaStreamSynchronize(m->dflash_stream) != cudaSuccess) {
            err = "batch: the drafter's stream failed";
            return false;
        }
        return true;
    }

    /// GUILD_BATCH_SYNC_DEBUG=1: every phase of a batch window waits for the device and names a fault
    bool phase_ok(const char* what, std::string& err) {
        static const bool on = std::getenv("GUILD_BATCH_SYNC_DEBUG") != nullptr;
        if (!on)
            return true;
        const cudaError_t e = cudaDeviceSynchronize();
        if (e == cudaSuccess)
            return true;
        err = std::string("batch (") + what + "): " + cudaGetErrorString(e);
        return false;
    }

    bool step(const std::function<void(const Token&)>& on_token, const std::function<void(const Done&)>& on_done,
              std::string& err) {
        constexpr int kMax = kernels::kVerifyMaxT;
        int S = 0, n_act = 0;
        int act[kMax] = {};
        for (int b = 0; b < (int) slots.size() && n_act < kMax; ++b)
            if (slots[(size_t) b].active)
                act[n_act++] = b;
        if (n_act == 0)
            return true;
        if (!ready()) {
            err = "model is not ready or an earlier inference failed; reload the model";
            return false;
        }
        core::ScopedModelExecution exec_scope(m->exec_state);
        std::optional<kernels::ScopedCudaDense> dense_scope;
        if (m->cuda_dense)
            dense_scope.emplace(*m->cuda_dense);
        std::optional<kernels::ScopedCudaAttn> attn_scope;
        if (m->cuda_attn)
            attn_scope.emplace(*m->cuda_attn);
        std::optional<kernels::ScopedCudaMoE> moe_scope;
        if (m->cuda_moe)
            moe_scope.emplace(*m->cuda_moe);
        // the CPU experts' host thread on the pool's host core, as the solo decode loop has it (GUILD_HOST_AFFINITY=0
        // opts out there and here)
        struct Affinity {
            kernels::cpu::ThreadAffinity previous;
            bool pinned = false;
            ~Affinity() {
                if (pinned)
                    kernels::cpu::restore_thread_affinity(previous);
            }
        } affinity;
        const char* host_affinity = std::getenv("GUILD_HOST_AFFINITY");
        if (!host_affinity || std::atoi(host_affinity) != 0) {
            affinity.previous = kernels::cpu::pin_current_thread_if_allowed(m->pool->host_core());
            affinity.pinned = true;
        }
        if (windows == 0) {
            start = Clock::now();
            pool0 = m->drive.cpu_ms;
            wait0 = m->ver->ms_wait;
            miss0 = m->drive.d.multi_misses;
        }
        core::DFlashDrafter* d = dflash_sets ? m->dflash.get() : nullptr;
        MainContext back{d};
        // each active slot's rows: its token, then its drafts while few slots decode
        int nrows[kMax] = {};
        int32_t drafts[kMax][kMax] = {};
        const bool drafting = d != nullptr && n_act <= draft_slots && draft_rows > n_act;
        // a whole block, as the solo path drafts it (its first r - 1 drafts are this slot's rows);
        // GUILD_BATCH_DRAFT_BLOCK=rows: a block of the slot's rows only (a cheaper pass, other drafts: the block's
        // queries see one another)
        static const bool block_of_rows = [] {
            const char* v = std::getenv("GUILD_BATCH_DRAFT_BLOCK");
            return v != nullptr && std::string(v) == "rows";
        }();
        core::DFlashDrafter::SlotDraft jobs[kMax];
        int n_jobs = 0;
        const auto td0 = Clock::now();
        for (int i = 0; i < n_act; ++i) {
            Slot& sl = slots[(size_t) act[i]];
            int r = drafting ? draft_rows / n_act + (i < draft_rows % n_act ? 1 : 0) : 1;
            if (r > 1)
                r = (int) std::min<int64_t>({(int64_t) r, (int64_t) d->block(), m->config.max_context - sl.p});
            if (r > 1) {
                d->use_context(1 + act[i]);
                static const bool trace = std::getenv("GUILD_BATCH_SYNC_DEBUG") != nullptr;
                if (trace)
                    std::fprintf(stderr, "guild batch: slot %d drafts %d at %lld from set %d (context to %lld)\n",
                                 act[i], r - 1, (long long) sl.p, d->context_in_use(), (long long) d->context_end());
                if (d->context_end() != sl.p) {
                    r = 1; // its context fell behind (a stale set): no drafts for it this window
                } else if (shared_head) {
                    core::DFlashDrafter::SlotDraft& job = jobs[n_jobs++];
                    job.set = 1 + act[i];
                    job.anchor = sl.x;
                    job.start = sl.p;
                    job.B = block_of_rows ? r : d->block();
                    job.n = r - 1;
                    job.drafts = drafts[i];
                } else {
                    float probs[kMax] = {};
                    int32_t block[kMax] = {};
                    if (!d->draft(sl.x, sl.p, block_of_rows ? r : d->block(), block, probs, m->dflash_stream, err)) {
                        m->ready->store(false);
                        return false;
                    }
                    for (int j = 0; j < r - 1; ++j)
                        drafts[i][j] = block[j];
                }
            }
            nrows[i] = std::max(1, r);
        }
        if (n_jobs > 0 && !d->draft_slots(jobs, n_jobs, m->dflash_stream, err)) {
            m->ready->store(false);
            return false;
        }
        draft_ms += ms_between(td0, Clock::now());
        if (!phase_ok("drafts", err))
            return false;
        int rows_of[kMax] = {};
        int32_t tok[kMax] = {}, out[kMax] = {};
        int64_t pos[kMax] = {};
        int r0[kMax] = {};
        for (int i = 0; i < n_act; ++i) {
            const Slot& sl = slots[(size_t) act[i]];
            r0[i] = S;
            for (int j = 0; j < nrows[i] && S < kMax; ++j, ++S) {
                rows_of[S] = act[i];
                tok[S] = j == 0 ? sl.x : drafts[i][j - 1];
                pos[S] = sl.p + j;
            }
            nrows[i] = S - r0[i];
        }
        try {
            m->drive.d.layers = 0;
            m->drive.d.experts = 0;
            const auto w0 = Clock::now();
            if (!m->ver->run_slot_rows(rows_of, S, tok, pos, drive_pool_multi, &m->drive, out, err)) {
                m->ready->store(false);
                return false;
            }
            run_ms += ms_between(w0, Clock::now());
            if (!phase_ok("window", err))
                return false;
        } catch (const std::exception& e) {
            err = std::string("batch window failed: ") + e.what();
            m->ready->store(false);
            return false;
        }
        ++windows;
        rows += S;
        // each slot keeps its token and the drafts its own picks confirm, up to an end
        const auto& eos = m->config.eos_token_ids;
        auto is_eos = [&](int32_t y) { return std::find(eos.begin(), eos.end(), (int64_t) y) != eos.end(); };
        int keep[kMax] = {};
        for (int i = 0; i < n_act; ++i) {
            Slot& sl = slots[(size_t) act[i]];
            int a = 0;
            while (a < nrows[i] - 1 && tok[r0[i] + a + 1] == out[r0[i] + a])
                ++a;
            for (int j = 0; j <= a; ++j)
                if (is_eos(out[r0[i] + j]) || sl.produced + j + 1 >= sl.max_new) {
                    a = j;
                    break;
                }
            keep[i] = a + 1;
            sl.drafts_offered += nrows[i] - 1;
            sl.drafts_accepted += a;
            offered += nrows[i] - 1;
            accepted += a;
        }
        const auto c0 = Clock::now();
        const bool any_drafts = S > n_act;
        if (!(any_drafts ? m->ver->commit_slot_rows(keep, err) : m->ver->commit_slots(err))) {
            m->ready->store(false);
            return false;
        }
        commit_ms += ms_between(c0, Clock::now());
        if (!phase_ok("commit", err))
            return false;
        // every slot's drafter context takes the rows it kept (their features, from this window)
        if (d != nullptr) {
            const size_t fstride = d->target_layers().size() * (size_t) m->g.n_embd;
            for (int i = 0; i < n_act; ++i) {
                d->use_context(1 + act[i]);
                if (d->context_end() != slots[(size_t) act[i]].p) // stale (its conversation came in without one):
                    d->reset(slots[(size_t) act[i]].p);           // it starts here
                if (!d->add_context(m->ver->features() + (size_t) r0[i] * fstride, fstride, keep[i],
                                    slots[(size_t) act[i]].p, m->dflash_stream, err)) {
                    m->ready->store(false);
                    return false;
                }
            }
            if (!phase_ok("drafter contexts", err))
                return false;
            if (cudaStreamSynchronize(m->dflash_stream) != cudaSuccess) { // the next window rewrites the features
                err = "batch: the drafter's stream failed";
                m->ready->store(false);
                return false;
            }
        }
        for (int i = 0; i < n_act; ++i) {
            const int b = act[i];
            Slot& sl = slots[(size_t) b];
            const char* fin = nullptr;
            for (int j = 0; j < keep[i] && fin == nullptr; ++j) {
                const int32_t y = out[r0[i] + j];
                sl.ids.push_back(tok[r0[i] + j]); // the window fed it: the slot's sessions hold it now
                ++sl.produced;
                ++tokens;
                if (on_token)
                    on_token(Token{b, y});
                fin = is_eos(y)                              ? "stop"
                      : sl.produced >= sl.max_new            ? "length"
                      : sl.p + j + 2 > m->config.max_context ? "length"
                      : (sl.stop && j + 1 == keep[i])        ? "cancel"
                                                             : nullptr;
            }
            sl.x = out[r0[i] + keep[i] - 1];
            sl.p += keep[i];
            if (fin != nullptr) {
                sl.active = false;
                sl.stop = false;
                if (on_done)
                    on_done(Done{b, sl.produced, fin, ms_between(sl.t0, Clock::now())});
            }
        }
        return true;
    }
};

GuildBatch::GuildBatch(GuildSession* session) : impl_(std::make_unique<Impl>()) {
    impl_->session = session;
    impl_->m = session && session->model() ? session->model()->impl() : nullptr;
    if (impl_->m) {
        GuildModel::Impl& m = *impl_->m;
        impl_->slots.resize(m.slot_ss.size());
        impl_->note = m.batch_note;
        impl_->dflash_sets = m.dflash != nullptr && m.dflash->contexts() >= (int) m.slot_ss.size() + 1;
        auto env_int = [](const char* k, int dflt) {
            const char* v = std::getenv(k);
            return v != nullptr ? std::atoi(v) : dflt;
        };
        impl_->draft_slots = env_int("GUILD_BATCH_DRAFT_SLOTS", 3);
        impl_->draft_rows = std::min(env_int("GUILD_BATCH_ROWS", kernels::kVerifyMaxT), kernels::kVerifyMaxT);
        impl_->shared_head = env_int("GUILD_BATCH_DFLASH_SHARED_HEAD", 1) != 0;
    }
}

GuildBatch::~GuildBatch() = default;

int GuildBatch::slots() const {
    return (int) impl_->slots.size();
}
const std::string& GuildBatch::note() const {
    return impl_->note;
}
bool GuildBatch::slot_active(int b) const {
    return b >= 0 && b < slots() && impl_->slots[(size_t) b].active;
}
bool GuildBatch::any_active() const {
    return active_count() > 0;
}
int GuildBatch::active_count() const {
    int n = 0;
    for (const auto& s : impl_->slots)
        n += s.active ? 1 : 0;
    return n;
}
const std::vector<int32_t>& GuildBatch::held(int b) const {
    static const std::vector<int32_t> none;
    return b >= 0 && b < slots() ? impl_->slots[(size_t) b].ids : none;
}

int GuildBatch::restore_for(const std::vector<int32_t>& prompt, std::string& err) {
    Impl& b = *impl_;
    if (b.slots.empty() || prompt.size() < 2 || !b.ready())
        return -1;
    GuildModel::Impl& m = *b.m;
    const size_t max_reusable = prompt.size() - 1; // the last token is read by the first window
    auto prefix_of_prompt = [&](const std::vector<int32_t>& v) {
        return !v.empty() && v.size() <= max_reusable && std::equal(v.begin(), v.end(), prompt.begin());
    };
    // what the session reuses by itself (session.cpp reuse_prefix): its live state, else the longest checkpoint
    size_t own = 0;
    if (std::getenv("GUILD_NO_KV_CACHE") == nullptr || std::atoi(std::getenv("GUILD_NO_KV_CACHE")) == 0) {
        if (prefix_of_prompt(m.live_committed_tokens))
            own = m.live_committed_tokens.size();
        for (const auto& cp : m.prefix_checkpoints)
            if (cp.tokens.size() > own && prefix_of_prompt(cp.tokens))
                own = cp.tokens.size();
    } else {
        return -1;
    }
    int best = -1;
    size_t best_n = own;
    for (int s = 0; s < (int) b.slots.size(); ++s) {
        const auto& sl = b.slots[(size_t) s];
        if (!sl.active && sl.ids.size() > best_n && prefix_of_prompt(sl.ids)) {
            best = s;
            best_n = sl.ids.size();
        }
    }
    if (best < 0)
        return -1;
    std::lock_guard<std::mutex> lock(m.generation_mutex);
    core::ScopedModelExecution exec_scope(m.exec_state);
    const auto t0 = Clock::now();
    auto& sl = b.slots[(size_t) best];
    if (!b.copy(*m.slot_ss[(size_t) best], m.ss, (int64_t) sl.ids.size(), err) ||
        !b.copy_dflash(best, (int64_t) sl.ids.size(), false, err)) {
        m.ready->store(false);
        return -1;
    }
    // the drafter's context is the slot's now: the session keeps it for these tokens (generate's dflash_seq rule)
    if (b.dflash_sets)
        m.dflash_seq = sl.ids;
    // The session's state is the slot's conversation now: its live tokens, and only the checkpoints that are its
    // prefixes stay (a checkpoint's K/V cells are the live ones, and those before its end now hold this conversation).
    m.live_committed_tokens = sl.ids;
    for (auto it = m.prefix_checkpoints.begin(); it != m.prefix_checkpoints.end();) {
        const bool keep =
            it->tokens.size() <= sl.ids.size() && std::equal(it->tokens.begin(), it->tokens.end(), sl.ids.begin());
        if (keep) {
            ++it;
        } else {
            m.recycle_checkpoint(*it);
            it = m.prefix_checkpoints.erase(it);
        }
    }
    std::fprintf(stderr, "guild batch: slot %d's %zu tokens back into the session (copied in %.1f ms)\n", best,
                 sl.ids.size(), ms_between(t0, Clock::now()));
    return best;
}

bool GuildBatch::admit(int b, const std::vector<int32_t>& prompt, int32_t first, int64_t max_new,
                       const GenerationRequest& sampling, std::string& err) {
    Impl& bi = *impl_;
    if (b < 0 || b >= slots() || bi.slots[(size_t) b].active) {
        err = "batch: no such free slot";
        return false;
    }
    if (!bi.ready()) {
        err = "model is not ready or an earlier inference failed; reload the model";
        return false;
    }
    GuildModel::Impl& m = *bi.m;
    if ((int64_t) prompt.size() + 1 > m.config.max_context) {
        err = "batch: the prompt fills the context";
        return false;
    }
    std::lock_guard<std::mutex> lock(m.generation_mutex);
    core::ScopedModelExecution exec_scope(m.exec_state);
    auto& sl = bi.slots[(size_t) b];
    sl = Impl::Slot{};
    const auto t0 = Clock::now();
    if (!bi.copy(m.ss, *m.slot_ss[(size_t) b], (int64_t) prompt.size(), err) ||
        !bi.copy_dflash(b, (int64_t) prompt.size(), true, err)) {
        m.ready->store(false);
        return false;
    }
    kernels::SamplerParams sp;
    sp.temperature = sampling.temperature;
    sp.top_p = sampling.top_p;
    sp.min_p = sampling.min_p;
    sp.top_k = sampling.top_k;
    sp.seed = sampling.seed;
    sp.greedy = sampling.temperature <= 0.0f;
    m.ver->set_slot_sampling(b, sp);
    sl.active = true;
    sl.x = first;
    sl.p = (int64_t) prompt.size();
    sl.produced = 1; // the first token, from the admission's window
    sl.max_new = max_new;
    sl.t0 = Clock::now();
    sl.ids = prompt;
    std::fprintf(stderr, "guild batch: slot %d takes %zu tokens (copied in %.1f ms)\n", b, prompt.size(),
                 ms_between(t0, Clock::now()));
    return true;
}

void GuildBatch::stop(int b) {
    if (b >= 0 && b < slots() && impl_->slots[(size_t) b].active)
        impl_->slots[(size_t) b].stop = true;
}

bool GuildBatch::park_inside(int b, const std::vector<int32_t>& prefix, std::string& err) {
    Impl& bi = *impl_;
    if (b < 0 || b >= slots() || bi.slots[(size_t) b].active || prefix.empty()) {
        err = "batch: no such idle slot to park the prompt in";
        return false;
    }
    if (!bi.ready()) {
        err = "model is not ready or an earlier inference failed; reload the model";
        return false;
    }
    GuildModel::Impl& m = *bi.m;
    auto& sl = bi.slots[(size_t) b];
    sl = Impl::Slot{};
    const auto t0 = Clock::now();
    if (!bi.copy(m.ss, *m.slot_ss[(size_t) b], (int64_t) prefix.size(), err) ||
        !bi.copy_dflash(b, (int64_t) prefix.size(), true, err)) {
        m.ready->store(false);
        return false;
    }
    sl.ids = prefix;
    std::fprintf(stderr,
                 "guild batch: the prompt read gives way at %zu tokens; slot %d holds them (copied in %.1f ms)\n",
                 prefix.size(), b, ms_between(t0, Clock::now()));
    return true;
}

bool GuildBatch::step(const std::function<void(const Token&)>& on_token,
                      const std::function<void(const Done&)>& on_done, std::string& err) {
    if (!any_active())
        return true;
    std::lock_guard<std::mutex> lock(impl_->m->generation_mutex);
    return impl_->step(on_token, on_done, err);
}

bool GuildBatch::step_inside(const std::function<void(const Token&)>& on_token,
                             const std::function<void(const Done&)>& on_done, std::string& err) {
    if (!any_active())
        return true;
    return impl_->step(on_token, on_done, err);
}

std::string GuildBatch::take_report() {
    Impl& b = *impl_;
    if (b.windows == 0 || b.m == nullptr)
        return {};
    const double w = (double) b.windows, wall = ms_between(b.start, Clock::now());
    const double L = (double) std::max<int64_t>(1, b.m->g.n_layers);
    char line[640];
    std::snprintf(
        line, sizeof line,
        "guild batch: %lld windows, %.2f rows and %.2f tokens a window, %.2f ms a window (drafts %.2f, run %.2f = "
        "GPU-reach wait %.2f + CPU experts %.2f; commit %.2f), drafts accepted %lld of %lld, %.2f CPU experts a "
        "layer-window; %.1f tokens/s over %.0f ms",
        (long long) b.windows, b.rows / w, b.tokens / w, (b.draft_ms + b.run_ms + b.commit_ms) / w, b.draft_ms / w,
        b.run_ms / w, (b.m->ver->ms_wait - b.wait0) / w, (b.m->drive.cpu_ms - b.pool0) / w, b.commit_ms / w,
        (long long) b.accepted, (long long) b.offered, (b.m->drive.d.multi_misses - b.miss0) / (w * L),
        1000.0 * b.tokens / std::max(wall, 1e-9), wall);
    b.windows = b.rows = b.tokens = b.offered = b.accepted = 0;
    b.run_ms = b.commit_ms = b.draft_ms = 0;
    std::string out = line;
    // GUILD_VERIFY_PROFILE=1 (one-group windows: GUILD_BATCH_SPLIT=0): the GPU's stages of these windows
    if (const std::string prof = b.m->ver->profile_report(); !prof.empty())
        out += "\nguild batch GPU stages (ms a window):" + prof;
    return out;
}

} // namespace guild::runtime

#else // CPU-only build: no batch slots

namespace guild::runtime {

struct GuildBatch::Impl {
    std::string note = "batch slots need the GPU runtime";
};

GuildBatch::GuildBatch(GuildSession*) : impl_(std::make_unique<Impl>()) {
}
GuildBatch::~GuildBatch() = default;
int GuildBatch::slots() const {
    return 0;
}
const std::string& GuildBatch::note() const {
    return impl_->note;
}
bool GuildBatch::slot_active(int) const {
    return false;
}
bool GuildBatch::any_active() const {
    return false;
}
int GuildBatch::active_count() const {
    return 0;
}
const std::vector<int32_t>& GuildBatch::held(int) const {
    static const std::vector<int32_t> none;
    return none;
}
int GuildBatch::restore_for(const std::vector<int32_t>&, std::string&) {
    return -1;
}
bool GuildBatch::admit(int, const std::vector<int32_t>&, int32_t, int64_t, const GenerationRequest&, std::string& err) {
    err = impl_->note;
    return false;
}
void GuildBatch::stop(int) {
}
bool GuildBatch::park_inside(int, const std::vector<int32_t>&, std::string& err) {
    err = impl_->note;
    return false;
}
bool GuildBatch::step(const std::function<void(const Token&)>&, const std::function<void(const Done&)>&, std::string&) {
    return true;
}
bool GuildBatch::step_inside(const std::function<void(const Token&)>&, const std::function<void(const Done&)>&,
                             std::string&) {
    return true;
}
std::string GuildBatch::take_report() {
    return {};
}

} // namespace guild::runtime

#endif
