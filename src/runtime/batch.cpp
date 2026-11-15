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
        int32_t x = 0;   ///< the token the next window feeds (not yet in the slot's state)
        int64_t p = 0;   ///< its position
        int64_t produced = 0, max_new = 0;
        Clock::time_point t0;
        /// what the slot's sessions hold: the prompt, then every token a window fed it.  Kept when the request ends,
        /// so its conversation's next turn (or the same request on the solo path) continues from it.
        std::vector<int32_t> ids;
    };
    std::vector<Slot> slots;
    std::string note;
    // the windows since the slots were last all idle (take_report)
    int64_t windows = 0, rows = 0;
    double run_ms = 0, commit_ms = 0, pool0 = 0, wait0 = 0;
    int64_t miss0 = 0;
    Clock::time_point start;

    bool ready() const {
        return m != nullptr && m->ready && m->ready->load();
    }

    // `from` -> `to`, the first n tokens' state.  Nothing may be writing either: the main session's last commit may
    // still be in flight (the solo path does not wait for it), and the copies run on the legacy stream.
    bool copy(const core::SessionState& from, core::SessionState& to, int64_t n, std::string& err) {
        if (!m->ver->wait_commit(err))
            return false;
        if (cudaDeviceSynchronize() != cudaSuccess) {
            err = "batch: the device failed before a slot copy";
            return false;
        }
        for (int64_t j = 0; j < from.qsa_alloc; ++j)
            if (!core::conversation_kv_copy(from.qsa_states[from.qsa_ord0 + j], to.qsa_states[to.qsa_ord0 + j], m->g,
                                            n, true, err))
                return false;
        if (!core::conversation_state_copy(from, to, m->g, (size_t) n, err))
            return false;
        if (cudaDeviceSynchronize() != cudaSuccess) {
            err = "batch: a slot copy failed";
            return false;
        }
        return true;
    }

    bool step(const std::function<void(const Token&)>& on_token, const std::function<void(const Done&)>& on_done,
              std::string& err) {
        constexpr int kMax = kernels::kVerifyMaxT;
        int S = 0;
        int rows_of[kMax] = {};
        int32_t tok[kMax] = {}, out[kMax] = {};
        int64_t pos[kMax] = {};
        for (int b = 0; b < (int) slots.size() && S < kMax; ++b)
            if (slots[(size_t) b].active) {
                rows_of[S] = b;
                tok[S] = slots[(size_t) b].x;
                pos[S] = slots[(size_t) b].p;
                ++S;
            }
        if (S == 0)
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
        try {
            m->drive.d.layers = 0;
            m->drive.d.experts = 0;
            const auto w0 = Clock::now();
            if (!m->ver->run_slot_rows(rows_of, S, tok, pos, drive_pool_multi, &m->drive, out, err)) {
                m->ready->store(false);
                return false;
            }
            const auto w1 = Clock::now();
            if (!m->ver->commit_slots(err)) {
                m->ready->store(false);
                return false;
            }
            run_ms += ms_between(w0, w1);
            commit_ms += ms_between(w1, Clock::now());
        } catch (const std::exception& e) {
            err = std::string("batch window failed: ") + e.what();
            m->ready->store(false);
            return false;
        }
        ++windows;
        rows += S;
        const auto& eos = m->config.eos_token_ids;
        for (int t = 0; t < S; ++t) {
            const int b = rows_of[t];
            Slot& sl = slots[(size_t) b];
            const int32_t y = out[t];
            sl.ids.push_back(sl.x); // the window fed it: the slot's sessions hold it now
            ++sl.produced;
            if (on_token)
                on_token(Token{b, y});
            const bool is_eos = std::find(eos.begin(), eos.end(), (int64_t) y) != eos.end();
            const char* fin = is_eos                           ? "stop"
                              : sl.stop                        ? "cancel"
                              : sl.produced >= sl.max_new      ? "length"
                              : sl.p + 2 > m->config.max_context ? "length"
                                                               : nullptr;
            if (fin != nullptr) {
                sl.active = false;
                sl.stop = false;
                if (on_done)
                    on_done(Done{b, sl.produced, fin, ms_between(sl.t0, Clock::now())});
            } else {
                sl.x = y;
                sl.p += 1;
            }
        }
        return true;
    }
};

GuildBatch::GuildBatch(GuildSession* session) : impl_(std::make_unique<Impl>()) {
    impl_->session = session;
    impl_->m = session && session->model() ? session->model()->impl() : nullptr;
    if (impl_->m) {
        impl_->slots.resize(impl_->m->slot_ss.size());
        impl_->note = impl_->m->batch_note;
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
    if (!b.copy(*m.slot_ss[(size_t) best], m.ss, (int64_t) sl.ids.size(), err)) {
        m.ready->store(false);
        return -1;
    }
    // The session's state is the slot's conversation now: its live tokens, and only the checkpoints that are its
    // prefixes stay (a checkpoint's K/V cells are the live ones, and those before its end now hold this conversation).
    m.live_committed_tokens = sl.ids;
    for (auto it = m.prefix_checkpoints.begin(); it != m.prefix_checkpoints.end();) {
        const bool keep = it->tokens.size() <= sl.ids.size() &&
                          std::equal(it->tokens.begin(), it->tokens.end(), sl.ids.begin());
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
    if (!bi.copy(m.ss, *m.slot_ss[(size_t) b], (int64_t) prompt.size(), err)) {
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
    if (!bi.copy(m.ss, *m.slot_ss[(size_t) b], (int64_t) prefix.size(), err)) {
        m.ready->store(false);
        return false;
    }
    sl.ids = prefix;
    std::fprintf(stderr, "guild batch: the prompt read gives way at %zu tokens; slot %d holds them (copied in %.1f ms)\n",
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
    char line[512];
    std::snprintf(line, sizeof line,
                  "guild batch: %lld windows, %.2f rows a window, %.2f ms a window (run %.2f = GPU-reach wait %.2f + CPU "
                  "experts %.2f; commit %.2f), %.2f CPU experts a layer-window; %.1f rows/s over %.0f ms",
                  (long long) b.windows, b.rows / w, (b.run_ms + b.commit_ms) / w, b.run_ms / w,
                  (b.m->ver->ms_wait - b.wait0) / w, (b.m->drive.cpu_ms - b.pool0) / w, b.commit_ms / w,
                  (b.m->drive.d.multi_misses - b.miss0) / (w * L), 1000.0 * b.rows / std::max(wall, 1e-9), wall);
    b.windows = b.rows = 0;
    b.run_ms = b.commit_ms = 0;
    return line;
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
bool GuildBatch::admit(int, const std::vector<int32_t>&, int32_t, int64_t, const GenerationRequest&,
                       std::string& err) {
    err = impl_->note;
    return false;
}
void GuildBatch::stop(int) {
}
bool GuildBatch::park_inside(int, const std::vector<int32_t>&, std::string& err) {
    err = impl_->note;
    return false;
}
bool GuildBatch::step(const std::function<void(const Token&)>&, const std::function<void(const Done&)>&,
                      std::string&) {
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
