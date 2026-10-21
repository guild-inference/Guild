// src/spec/suffix_drafter.cpp - see include/guild/spec/suffix_drafter.hpp.
#include "guild/spec/suffix_drafter.hpp"

#include <algorithm>

namespace guild::spec {

namespace {
uint64_t mix(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdull;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ull;
    return x ^ (x >> 33);
}
} // namespace

SuffixDrafter::SuffixDrafter(int min_match, int max_match, size_t capacity_tokens)
    : min_match_(std::max(2, min_match)),
      max_match_(std::max(min_match_, max_match)),
      max_hist_(std::max<size_t>(capacity_tokens * 8, 256)) {
    size_t cap = 1;
    const size_t want_slots = std::max<size_t>(capacity_tokens * 2, 64);
    while (cap < want_slots)
        cap <<= 1; // load factor <= 0.5 at the nominal capacity
    table_.assign(cap, Slot{});
    mask_ = cap - 1;
    hist_.reserve(std::min<size_t>(capacity_tokens, 1u << 20));
}

void SuffixDrafter::set_max_history(size_t max_tokens) {
    max_hist_ = std::max<size_t>(max_tokens, 64);
    if (hist_.size() > max_hist_)
        compact_history();
}

void SuffixDrafter::reset() {
    hist_.clear();
    std::fill(table_.begin(), table_.end(), Slot{});
    last_match_ = 0;
    last_freq_ = 0;
}

uint64_t SuffixDrafter::key_at(size_t end) const {
    const uint64_t a = (uint32_t) hist_[end - 2], b = (uint32_t) hist_[end - 1], c = (uint32_t) hist_[end];
    const uint64_t h = mix(a * 0x9E3779B97F4A7C15ull ^ mix(b + 0x632BE59BD9B4E019ull) ^ (c << 1));
    return (h & ~3ull) | 1ull; // tag 01: trigram, never 0
}

uint64_t SuffixDrafter::key2_at(size_t end) const {
    const uint64_t b = (uint32_t) hist_[end - 1], c = (uint32_t) hist_[end];
    const uint64_t h = mix(b * 0x9E3779B97F4A7C15ull ^ (c + 0x632BE59BD9B4E019ull));
    return (h & ~3ull) | 3ull; // tag 11: bigram, never 0
}

SuffixDrafter::Slot* SuffixDrafter::find_slot(uint64_t key, bool insert) {
    const size_t max_p = std::min(kMaxProbe, mask_ + 1);
    size_t oldest_idx = key & mask_;
    uint32_t oldest_pos = UINT32_MAX;
    for (size_t i = key & mask_, probes = 0; probes < max_p; i = (i + 1) & mask_, ++probes) {
        Slot& s = table_[i];
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
    Slot& ev = table_[oldest_idx];
    ev = Slot{};
    ev.key = key;
    return &ev;
}

void SuffixDrafter::index_position(size_t end) {
    if (end >= 2) {
        Slot* s = find_slot(key_at(end), true);
        if (s != nullptr) {
            for (int w = WAYS - 1; w > 0; --w)
                s->pos[w] = s->pos[w - 1];
            s->pos[0] = (uint32_t) end;
            if (s->n < WAYS)
                ++s->n;
        }
    }
    if (min_match_ == 2 && end >= 1) {
        Slot* s2 = find_slot(key2_at(end), true);
        if (s2 != nullptr) {
            for (int w = WAYS - 1; w > 0; --w)
                s2->pos[w] = s2->pos[w - 1];
            s2->pos[0] = (uint32_t) end;
            if (s2->n < WAYS)
                ++s2->n;
        }
    }
}

void SuffixDrafter::compact_history() {
    const size_t keep = std::max<size_t>(max_hist_ / 2, 32);
    if (hist_.size() <= keep)
        return;
    const size_t drop = hist_.size() - keep;
    std::move(hist_.begin() + (ptrdiff_t) drop, hist_.end(), hist_.begin());
    hist_.resize(keep);
    std::fill(table_.begin(), table_.end(), Slot{});
    for (size_t idx = 0; idx < hist_.size(); ++idx)
        index_position(idx);
}

void SuffixDrafter::append(const int32_t* tokens, size_t n) {
    if (tokens == nullptr || n == 0)
        return;
    for (size_t i = 0; i < n; ++i) {
        if (hist_.size() >= max_hist_)
            compact_history();
        hist_.push_back(tokens[i]);
        index_position(hist_.size() - 1);
    }
}

int SuffixDrafter::propose(int max_k, int32_t* out) {
    last_match_ = 0;
    last_freq_ = 0;
    const size_t n = hist_.size();
    if (n < (size_t) (min_match_ + 1) || max_k <= 0 || out == nullptr)
        return 0;
    const size_t cur = n - 1;

    auto eval_slot = [&](const Slot* s, size_t& best_end, int& best_len, int& best_freq) {
        if (s == nullptr || s->n == 0)
            return;
        int cand_len[WAYS] = {};
        size_t cand_end[WAYS] = {};
        int valid = 0;
        for (int w = 0; w < s->n; ++w) {
            const size_t p = s->pos[w];
            if (p >= cur)
                continue; // the current suffix itself
            int len = 0;
            while (len < max_match_ && len <= (int) p && hist_[p - len] == hist_[cur - len])
                ++len;
            if (len >= min_match_) {
                cand_len[valid] = len;
                cand_end[valid] = p;
                ++valid;
            }
        }
        for (int i = 0; i < valid; ++i) {
            const int len = cand_len[i];
            const size_t p = cand_end[i];
            const int32_t next_tok = hist_[p + 1];
            int freq = 0;
            for (int j = 0; j < valid; ++j) {
                if (hist_[cand_end[j] + 1] == next_tok)
                    ++freq;
            }
            // Prefer longer match; on equal match length, prefer higher continuation frequency, then most recent
            // (smaller i).
            if (len > best_len || (len == best_len && freq > best_freq)) {
                best_len = len;
                best_freq = freq;
                best_end = p;
            }
        }
    };

    size_t best_end = 0;
    int best_len = 0;
    int best_freq = 0;
    if (cur >= 2)
        eval_slot(find_slot(key_at(cur), false), best_end, best_len, best_freq);
    if (best_len < min_match_ && min_match_ == 2 && cur >= 1)
        eval_slot(find_slot(key2_at(cur), false), best_end, best_len, best_freq);

    if (best_len < min_match_)
        return 0;
    last_match_ = best_len;
    last_freq_ = best_freq;
    int k = 0;
    // The continuation may run into the current suffix (periodic text); reading history up to `cur` is valid.
    for (size_t q = best_end + 1; q <= cur && k < max_k; ++q)
        out[k++] = hist_[q];
    return k;
}

} // namespace guild::spec
