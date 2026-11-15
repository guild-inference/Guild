#pragma once

// guild::runtime::GuildBatch - several conversations decoded together (guild-generate --batch N, the server's
// "parallel": N).
//
// A request is read through the session's usual path - prompt cache, prompt engine, its first token from a verify
// window - and its state is then copied into one of the model's SLOT SESSIONS (carved at load like the main session:
// same context, same K/V mode).  From there every batch window carries one token of each active slot, so the dense
// weights, the shared expert, the head and every routed expert two conversations share are read once per window for
// all of them (core/verify.hpp run_slot_rows).  A row's arithmetic is the single-token window's, so with greedy
// decoding a conversation produces in a slot the tokens it produces alone (with GUILD_IQ_MT_MIN=1, as for the solo
// path's own exactness tests).
//
// A slot that finishes keeps what it holds (the prompt and every token a window fed it): the next turn of that
// conversation, or the same request going back to the solo path, continues from it instead of reading the history
// again (`restore_for`).  Sampled slots draw row by row with the solo window's Philox(seed, position); penalties are
// not applied in batch windows.  Thread use: one thread drives a GuildBatch (the serve loop); `step` takes the
// model's generation lock, `step_inside` runs from GenerationRequest::between_chunks, which holds it already.

#include "guild/runtime/types.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace guild::runtime {

class GuildModel;
class GuildSession;

class GuildBatch {
public:
    /// `session` reads the prompts (its model's slot sessions are the slots).
    explicit GuildBatch(GuildSession* session);
    ~GuildBatch();
    GuildBatch(const GuildBatch&) = delete;
    GuildBatch& operator=(const GuildBatch&) = delete;

    /// The slots the model carved (0: batching is off; `note()` says why when it was asked for).
    int slots() const;
    const std::string& note() const;
    bool slot_active(int b) const;
    bool any_active() const;
    int active_count() const;
    /// The tokens slot `b`'s sessions hold (an idle slot keeps its last conversation; empty: nothing reusable).
    const std::vector<int32_t>& held(int b) const;

    /// Before a request is read through the session: when an idle slot holds a longer start of `prompt` than the
    /// session itself can reuse (its live state or a prompt-cache checkpoint), that slot's state is copied into the
    /// session, so only the rest is read.  Returns the slot used (-1: none).  `max_reusable`: at most this many
    /// tokens are taken (the prompt's last token is always read by a window).
    int restore_for(const std::vector<int32_t>& prompt, std::string& err);

    /// After `session` read a request's prompt and produced its first token `first` (GuildSession::generate with
    /// max_new_tokens = 1, into_slot): its state moves into slot `b`, which decodes the rest in batch windows.
    /// `prompt` is what the session holds (the whole prompt), `max_new` the request's limit (the first token
    /// counted).  False with `err` when the copy failed (the slot then holds nothing).
    bool admit(int b, const std::vector<int32_t>& prompt, int32_t first, int64_t max_new,
               const GenerationRequest& sampling, std::string& err);

    /// Ends slot `b` at its next window (finish "cancel"); nothing when it is idle.
    void stop(int b);

    /// From GenerationRequest::between_chunks only (the generating thread holds the model): the session's first
    /// `prefix.size()` tokens - the part of a prompt read so far - into the idle slot `b`, which then holds them, so the
    /// same request, sent again, reads on from there (restore_for) with the same chunks.  The server's BYIELD: a long
    /// prompt read gives way to a short request at a chunk boundary.  False with `err` when the copy failed.
    bool park_inside(int b, const std::vector<int32_t>& prefix, std::string& err);

    struct Token {
        int slot;
        int32_t id;
    };
    struct Done {
        int slot;
        int64_t produced;
        const char* finish; ///< "stop" (EOS), "length", "cancel"
        double ms;          ///< since its admission
    };
    /// One batch window over the active slots (each gets one token); the slots that end are reported after their
    /// last token.  Takes the model's generation lock.  False: an error (`err`); the model is then not ready.
    bool step(const std::function<void(const Token&)>& on_token, const std::function<void(const Done&)>& on_done,
              std::string& err);
    /// The same, from inside a generation (GenerationRequest::between_chunks: the generating thread holds the lock).
    bool step_inside(const std::function<void(const Token&)>& on_token,
                     const std::function<void(const Done&)>& on_done, std::string& err);

    /// The timing of the windows since the slots were last all idle, one line ("" when no window ran); resets it.
    std::string take_report();

    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
};

} // namespace guild::runtime
