// src/spec/retrieval_drafter.cpp - see include/guild/spec/retrieval_drafter.hpp.
#include "guild/spec/retrieval_drafter.hpp"

#include <algorithm>
#include <cmath>

namespace guild::spec {

namespace {
uint64_t mix(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdull;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ull;
    return x ^ (x >> 33);
}

size_t next_pow2_slots(size_t max_tokens) {
    size_t cap = 64;
    const size_t target = std::clamp<size_t>(max_tokens * 2, 64, 1u << 20);
    while (cap < target)
        cap <<= 1;
    return cap;
}
} // namespace

RetrievalDrafter::RetrievalDrafter(int min_match, int max_match, size_t max_stream_tokens, size_t max_corpus_tokens,
                                   float min_confidence)
    : min_match_(std::max(2, min_match)),
      max_match_(std::max(min_match_, max_match)),
      max_stream_tokens_(std::max<size_t>(max_stream_tokens, 64)),
      max_corpus_tokens_(std::max<size_t>(max_corpus_tokens, 64)),
      min_confidence_(min_confidence) {
    const size_t stream_cap = next_pow2_slots(max_stream_tokens_);
    stream_table_.assign(stream_cap, Slot{});
    stream_mask_ = stream_cap - 1;
    stream_.reserve(std::min<size_t>(max_stream_tokens_, 1u << 18));

    const size_t corpus_cap = next_pow2_slots(max_corpus_tokens_);
    corpus_table_.assign(corpus_cap, Slot{});
    corpus_mask_ = corpus_cap - 1;
    corpus_.reserve(std::min<size_t>(max_corpus_tokens_, 1u << 18));
}

void RetrievalDrafter::reset_stream() {
    stream_.clear();
    std::fill(stream_table_.begin(), stream_table_.end(), Slot{});
    last_match_ = 0;
    last_candidates_ = 0;
    last_confidence_ = 0.0f;
    last_from_corpus_ = false;
}

void RetrievalDrafter::clear_corpus() {
    corpus_.clear();
    std::fill(corpus_table_.begin(), corpus_table_.end(), Slot{});
}

void RetrievalDrafter::reset_all() {
    reset_stream();
    clear_corpus();
}

void RetrievalDrafter::set_max_stream_tokens(size_t max_tokens) {
    max_stream_tokens_ = std::max<size_t>(max_tokens, 64);
    if (stream_.size() > max_stream_tokens_)
        compact_tier(stream_, stream_table_, stream_mask_, max_stream_tokens_);
}

void RetrievalDrafter::set_max_corpus_tokens(size_t max_tokens) {
    max_corpus_tokens_ = std::max<size_t>(max_tokens, 64);
    if (corpus_.size() > max_corpus_tokens_)
        compact_tier(corpus_, corpus_table_, corpus_mask_, max_corpus_tokens_);
}

uint64_t RetrievalDrafter::key2_at(const std::vector<int32_t>& seq, size_t end) {
    const uint64_t b = (uint32_t) seq[end - 1], c = (uint32_t) seq[end];
    const uint64_t h = mix(b * 0x9E3779B97F4A7C15ull ^ (c + 0x632BE59BD9B4E019ull));
    return (h & ~7ull) | 1ull; // tag 001: 2-gram
}

uint64_t RetrievalDrafter::key3_at(const std::vector<int32_t>& seq, size_t end) {
    const uint64_t a = (uint32_t) seq[end - 2], b = (uint32_t) seq[end - 1], c = (uint32_t) seq[end];
    const uint64_t h = mix(a * 0x9E3779B97F4A7C15ull ^ mix(b + 0x632BE59BD9B4E019ull) ^ (c << 1));
    return (h & ~7ull) | 3ull; // tag 011: 3-gram
}

uint64_t RetrievalDrafter::key4_at(const std::vector<int32_t>& seq, size_t end) {
    const uint64_t d0 = (uint32_t) seq[end - 3], a = (uint32_t) seq[end - 2];
    const uint64_t b = (uint32_t) seq[end - 1], c = (uint32_t) seq[end];
    const uint64_t h =
        mix(d0 * 0xBF58476D1CE4E5B9ull ^ mix(a * 0x9E3779B97F4A7C15ull) ^ mix(b + 0x632BE59BD9B4E019ull) ^ (c << 1));
    return (h & ~7ull) | 5ull; // tag 101: 4-gram
}

RetrievalDrafter::Slot* RetrievalDrafter::find_slot(std::vector<Slot>& table, size_t mask, uint64_t key, bool insert) {
    const size_t max_p = std::min(kMaxProbe, mask + 1);
    size_t oldest_idx = key & mask;
    uint32_t oldest_pos = UINT32_MAX;
    for (size_t i = key & mask, probes = 0; probes < max_p; i = (i + 1) & mask, ++probes) {
        Slot& s = table[i];
        if (s.key == key)
            return &s;
        if (s.key == 0) {
            if (!insert)
                return nullptr;
            s.key = key;
            return &s;
        }
        if (insert && s.pos[0] < oldest_pos) {
            oldest_pos = s.pos[0];
            oldest_idx = i;
        }
    }
    if (!insert)
        return nullptr;
    Slot& ev = table[oldest_idx];
    ev = Slot{};
    ev.key = key;
    return &ev;
}

const RetrievalDrafter::Slot* RetrievalDrafter::find_slot_const(const std::vector<Slot>& table, size_t mask,
                                                                uint64_t key) {
    const size_t max_p = std::min(kMaxProbe, mask + 1);
    for (size_t i = key & mask, probes = 0; probes < max_p; i = (i + 1) & mask, ++probes) {
        const Slot& s = table[i];
        if (s.key == key)
            return &s;
        if (s.key == 0)
            return nullptr;
    }
    return nullptr;
}

void RetrievalDrafter::index_token(std::vector<int32_t>& seq, std::vector<Slot>& table, size_t mask, size_t end) {
    auto push_pos = [&](uint64_t key) {
        Slot* s = find_slot(table, mask, key, true);
        if (s == nullptr)
            return;
        for (int w = WAYS - 1; w > 0; --w)
            s->pos[w] = s->pos[w - 1];
        s->pos[0] = (uint32_t) end;
        if (s->n < WAYS)
            ++s->n;
    };
    if (end >= 3)
        push_pos(key4_at(seq, end));
    if (end >= 2)
        push_pos(key3_at(seq, end));
    if (min_match_ == 2 && end >= 1)
        push_pos(key2_at(seq, end));
}

void RetrievalDrafter::compact_tier(std::vector<int32_t>& seq, std::vector<Slot>& table, size_t mask,
                                    size_t max_tokens) {
    const size_t keep = std::max<size_t>(max_tokens / 2, 32);
    if (seq.size() <= keep)
        return;
    const size_t drop = seq.size() - keep;
    std::move(seq.begin() + (ptrdiff_t) drop, seq.end(), seq.begin());
    seq.resize(keep);
    std::fill(table.begin(), table.end(), Slot{});
    for (size_t idx = 0; idx < seq.size(); ++idx)
        index_token(seq, table, mask, idx);
}

void RetrievalDrafter::append_stream(const int32_t* tokens, size_t n) {
    if (tokens == nullptr || n == 0)
        return;
    for (size_t i = 0; i < n; ++i) {
        if (stream_.size() >= max_stream_tokens_)
            compact_tier(stream_, stream_table_, stream_mask_, max_stream_tokens_);
        stream_.push_back(tokens[i]);
        index_token(stream_, stream_table_, stream_mask_, stream_.size() - 1);
    }
}

void RetrievalDrafter::ingest_corpus(const int32_t* tokens, size_t n) {
    if (tokens == nullptr || n == 0)
        return;
    for (size_t i = 0; i < n; ++i) {
        if (corpus_.size() >= max_corpus_tokens_)
            compact_tier(corpus_, corpus_table_, corpus_mask_, max_corpus_tokens_);
        corpus_.push_back(tokens[i]);
        index_token(corpus_, corpus_table_, corpus_mask_, corpus_.size() - 1);
    }
}

void RetrievalDrafter::archive_stream_to_corpus() {
    if (!stream_.empty())
        ingest_corpus(stream_.data(), stream_.size());
}

int RetrievalDrafter::propose(int max_k, int32_t* out) {
    last_match_ = 0;
    last_candidates_ = 0;
    last_confidence_ = 0.0f;
    last_from_corpus_ = false;

    const size_t n = stream_.size();
    if (n < (size_t) min_match_ || max_k <= 0 || out == nullptr)
        return 0;
    const size_t cur = n - 1;

    CandidateSpan spans[kMaxCandidates];
    int n_spans = 0;

    auto gather_from_slot = [&](const Slot* s, const std::vector<int32_t>& seq, size_t limit_pos, bool from_corpus) {
        if (s == nullptr || s->n == 0)
            return;
        for (int w = 0; w < s->n && n_spans < kMaxCandidates; ++w) {
            const size_t p = s->pos[w];
            if (p >= limit_pos)
                continue;
            // Avoid duplicate candidate spans from 4-gram and 3-gram slots pointing to the same end position.
            bool dup = false;
            for (int c = 0; c < n_spans; ++c) {
                if (spans[c].seq == &seq && spans[c].end_pos == p) {
                    dup = true;
                    break;
                }
            }
            if (dup)
                continue;

            int len = 0;
            const int max_back = (int) std::min<size_t>({(size_t) max_match_, p + 1, cur + 1});
            while (len < max_back && seq[p - (size_t) len] == stream_[cur - (size_t) len])
                ++len;
            if (len < min_match_)
                continue;

            const double source_boost = from_corpus ? 0.90 : 1.00;
            const double recency = 1.0 / (1.0 + 0.08 * (double) w);
            CandidateSpan& cs = spans[n_spans++];
            cs.seq = &seq;
            cs.end_pos = p;
            cs.limit_pos = limit_pos;
            cs.match_len = len;
            cs.weight = (double) (len * len) * source_boost * recency;
            cs.from_corpus = from_corpus;
            cs.active = true;
        }
    };

    // 1. Check active stream (prompt + generated tokens so far)
    if (n >= (size_t) (min_match_ + 1)) {
        if (cur >= 3)
            gather_from_slot(find_slot_const(stream_table_, stream_mask_, key4_at(stream_, cur)), stream_, cur, false);
        if (cur >= 2 && n_spans < kMaxCandidates)
            gather_from_slot(find_slot_const(stream_table_, stream_mask_, key3_at(stream_, cur)), stream_, cur, false);
        if (min_match_ == 2 && cur >= 1 && n_spans == 0)
            gather_from_slot(find_slot_const(stream_table_, stream_mask_, key2_at(stream_, cur)), stream_, cur, false);
    }

    // 2. Check session-local corpus datastore
    if (!corpus_.empty() && corpus_.size() > (size_t) min_match_) {
        const size_t corpus_limit = corpus_.size() - 1;
        if (cur >= 3 && n_spans < kMaxCandidates)
            gather_from_slot(find_slot_const(corpus_table_, corpus_mask_, key4_at(stream_, cur)), corpus_, corpus_limit,
                             true);
        if (cur >= 2 && n_spans < kMaxCandidates)
            gather_from_slot(find_slot_const(corpus_table_, corpus_mask_, key3_at(stream_, cur)), corpus_, corpus_limit,
                             true);
        if (min_match_ == 2 && cur >= 1 && n_spans == 0)
            gather_from_slot(find_slot_const(corpus_table_, corpus_mask_, key2_at(stream_, cur)), corpus_, corpus_limit,
                             true);
    }

    if (n_spans == 0)
        return 0;

    int best_match = 0;
    for (int c = 0; c < n_spans; ++c)
        best_match = std::max(best_match, spans[c].match_len);
    last_match_ = best_match;
    last_candidates_ = n_spans;

    int k = 0;
    float first_conf = 0.0f;
    bool first_corpus = false;

    for (int step = 0; step < max_k; ++step) {
        int32_t cand_tok[kMaxCandidates] = {};
        double cand_w[kMaxCandidates] = {};
        bool cand_corp[kMaxCandidates] = {};
        int n_uniq = 0;
        double total_w = 0.0;

        for (int c = 0; c < n_spans; ++c) {
            if (!spans[c].active)
                continue;
            const size_t pos = spans[c].end_pos + 1 + (size_t) step;
            if (pos > spans[c].limit_pos) {
                spans[c].active = false;
                continue;
            }
            const int32_t tok = (*spans[c].seq)[pos];
            total_w += spans[c].weight;
            int found = -1;
            for (int u = 0; u < n_uniq; ++u) {
                if (cand_tok[u] == tok) {
                    found = u;
                    break;
                }
            }
            if (found >= 0) {
                cand_w[found] += spans[c].weight;
            } else {
                cand_tok[n_uniq] = tok;
                cand_w[n_uniq] = spans[c].weight;
                cand_corp[n_uniq] = spans[c].from_corpus;
                ++n_uniq;
            }
        }

        if (n_uniq == 0 || total_w <= 0.0)
            break;

        int best_u = 0;
        for (int u = 1; u < n_uniq; ++u) {
            if (cand_w[u] > cand_w[best_u])
                best_u = u;
        }

        const float conf = (float) (cand_w[best_u] / total_w);
        if (step > 0 && conf < min_confidence_)
            break;

        if (step == 0) {
            first_conf = conf;
            first_corpus = cand_corp[best_u];
        }

        const int32_t chosen = cand_tok[best_u];
        out[k++] = chosen;

        // Prune candidate spans that diverged at this continuation step.
        for (int c = 0; c < n_spans; ++c) {
            if (!spans[c].active)
                continue;
            const size_t pos = spans[c].end_pos + 1 + (size_t) step;
            if (pos > spans[c].limit_pos || (*spans[c].seq)[pos] != chosen)
                spans[c].active = false;
        }
    }

    if (k == 0) {
        last_match_ = 0;
        last_candidates_ = 0;
        return 0;
    }
    last_confidence_ = first_conf;
    last_from_corpus_ = first_corpus;
    return k;
}

} // namespace guild::spec
