// include/guild/spec/retrieval_drafter.hpp - Phase 6.6: session-local retrieval-based speculative drafter.
//
// Extends suffix lookup to a two-tier session-local token datastore inspired by REST (He et al., NAACL 2024):
//   1. `corpus_`: session-local persistent token store across turns or ingested reference documents, strictly
//      isolated per session (never shared across sessions or users) and bounded by `max_corpus_tokens`.
//   2. `stream_`: the active generation request's prompt + committed output tokens, bounded by `max_stream_tokens`.
//
// Both tiers are indexed incrementally in O(1) time per token using bounded-probe multi-way open-addressing tables
// over 3-gram and 4-gram anchors (plus 2-gram anchors when `min_match == 2`). During `propose()`, up to
// `kMaxCandidates` matching spans are gathered from `stream_` and `corpus_`, extended backwards up to `max_match`,
// and combined via length-weighted continuation voting:
//
//   weight(span c) = (match_len_c)^2 * source_boost_c * recency_decay_c
//
// At each draft step j, candidate spans that matched proposed tokens 0..j-1 vote on token j. The highest-weight
// continuation token wins, and proposals stop early if empirical continuation confidence falls below `min_confidence`.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace guild::spec {

class RetrievalDrafter {
public:
    static constexpr int WAYS = 8;
    static constexpr size_t kMaxProbe = 16;
    static constexpr int kMaxCandidates = 16;

    explicit RetrievalDrafter(int min_match = 3, int max_match = 48, size_t max_stream_tokens = 1u << 16,
                              size_t max_corpus_tokens = 1u << 16, float min_confidence = 0.35f);

    /// Clear only the active request's token stream while retaining the session-local retrieval corpus.
    void reset_stream();

    /// Clear only the session-local retrieval corpus (explicit deletion/retention control).
    void clear_corpus();

    /// Clear both the active stream and the session-local retrieval corpus.
    void reset_all();

    /// Configure memory bounds.
    void set_max_stream_tokens(size_t max_tokens);
    void set_max_corpus_tokens(size_t max_tokens);
    void set_min_confidence(float min_conf) {
        min_confidence_ = min_conf;
    }

    /// Append tokens to the active request stream (prompt + committed tokens).
    void append_stream(const int32_t* tokens, size_t n);
    void append_stream(int32_t token) {
        append_stream(&token, 1);
    }

    /// Ingest tokens into the session-local retrieval corpus (e.g. completed turns or reference context).
    void ingest_corpus(const int32_t* tokens, size_t n);

    /// Commit the current active stream into the session-local corpus (called at turn completion or before reset).
    void archive_stream_to_corpus();

    /// Propose up to `max_k` tokens into `out`. Returns number of proposed tokens (0 if no qualifying match).
    int propose(int max_k, int32_t* out);

    int last_match() const {
        return last_match_;
    }
    int last_candidates() const {
        return last_candidates_;
    }
    float last_confidence() const {
        return last_confidence_;
    }
    bool last_from_corpus() const {
        return last_from_corpus_;
    }
    size_t stream_size() const {
        return stream_.size();
    }
    size_t corpus_size() const {
        return corpus_.size();
    }
    size_t memory_bytes() const {
        return (stream_.capacity() + corpus_.capacity()) * sizeof(int32_t) +
               (stream_table_.capacity() + corpus_table_.capacity()) * sizeof(Slot);
    }

private:
    struct Slot {
        uint64_t key = 0;
        uint32_t pos[WAYS] = {};
        uint8_t n = 0;
    };

    struct CandidateSpan {
        const std::vector<int32_t>* seq = nullptr;
        size_t end_pos = 0;
        size_t limit_pos = 0;
        int match_len = 0;
        double weight = 0.0;
        bool from_corpus = false;
        bool active = true;
    };

    static uint64_t key2_at(const std::vector<int32_t>& seq, size_t end);
    static uint64_t key3_at(const std::vector<int32_t>& seq, size_t end);
    static uint64_t key4_at(const std::vector<int32_t>& seq, size_t end);

    static Slot* find_slot(std::vector<Slot>& table, size_t mask, uint64_t key, bool insert);
    static const Slot* find_slot_const(const std::vector<Slot>& table, size_t mask, uint64_t key);
    void index_token(std::vector<int32_t>& seq, std::vector<Slot>& table, size_t mask, size_t end);
    void compact_tier(std::vector<int32_t>& seq, std::vector<Slot>& table, size_t mask, size_t max_tokens);

    int min_match_;
    int max_match_;
    size_t max_stream_tokens_;
    size_t max_corpus_tokens_;
    float min_confidence_;

    std::vector<int32_t> stream_;
    std::vector<Slot> stream_table_;
    size_t stream_mask_ = 0;

    std::vector<int32_t> corpus_;
    std::vector<Slot> corpus_table_;
    size_t corpus_mask_ = 0;

    int last_match_ = 0;
    int last_candidates_ = 0;
    float last_confidence_ = 0.0f;
    bool last_from_corpus_ = false;
};

} // namespace guild::spec
