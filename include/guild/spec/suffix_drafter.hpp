// include/guild/spec/suffix_drafter.hpp - plan v0.3 P6: the suffix-lookup drafter (no weights, no GPU).
//
// Proposes the tokens that followed the longest earlier occurrence of the sequence's current suffix: when the
// output quotes its input (edits, refactors, repeated code) the continuation is usually exact, and a verify pass
// accepts many tokens at once. It needs no model and costs microseconds.
//
// Index: every trigram of the history (prompt + accepted output) maps to its WAYS most recent end positions in a
// fixed-size open-addressing table, so memory is bounded (~20 bytes per history token) and appends are O(1).
// A proposal checks those candidates, extends each match backwards up to `max_match`, and takes the longest
// (most recent on ties). Matches shorter than `min_match` propose nothing.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace guild::spec {

class SuffixDrafter {
public:
    static constexpr int WAYS = 4;
    static constexpr size_t kMaxProbe = 16;

    explicit SuffixDrafter(int min_match = 3, int max_match = 32, size_t capacity_tokens = 1u << 19);

    void reset();
    void set_max_history(size_t max_tokens);
    size_t max_history() const {
        return max_hist_;
    }

    /// Add tokens to the history (the prompt, then every accepted token).
    void append(const int32_t* tokens, size_t n);
    void append(int32_t token) {
        append(&token, 1);
    }

    /// Write up to `max_k` proposed next tokens to `out`; returns how many (0 = no match of at least min_match).
    int propose(int max_k, int32_t* out);

    /// Length of the match behind the last proposal (0 if none).
    int last_match() const {
        return last_match_;
    }
    /// Number of candidate matches that agreed on the immediate continuation token of the last proposal.
    int last_freq() const {
        return last_freq_;
    }
    size_t size() const {
        return hist_.size();
    }
    size_t memory_bytes() const {
        return hist_.capacity() * sizeof(int32_t) + table_.capacity() * sizeof(Slot);
    }

private:
    struct Slot {
        uint64_t key = 0;        // n-gram hash (0 = empty)
        uint32_t pos[WAYS] = {}; // end positions, most recent first
        uint8_t n = 0;
    };
    Slot* find_slot(uint64_t key, bool insert);
    uint64_t key_at(size_t end) const;
    uint64_t key2_at(size_t end) const;
    void index_position(size_t end);
    void compact_history();

    int min_match_, max_match_;
    size_t max_hist_ = 0;
    std::vector<int32_t> hist_;
    std::vector<Slot> table_;
    size_t mask_ = 0;
    int last_match_ = 0;
    int last_freq_ = 0;
};

} // namespace guild::spec
