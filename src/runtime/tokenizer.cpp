#include "guild/runtime/tokenizer.hpp"
#include "guild/runtime/unicode_tables.hpp"
#include "guild/server/json.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace guild::runtime {

namespace {

// UTF-8 helper functions
uint32_t decode_utf8_codepoint(const char* s, size_t len, size_t& bytes_read) {
    if (len == 0) {
        bytes_read = 0;
        return 0;
    }
    const unsigned char c0 = static_cast<unsigned char>(s[0]);
    if (c0 < 0x80) {
        bytes_read = 1;
        return c0;
    }
    if ((c0 & 0xE0) == 0xC0 && len >= 2) {
        const unsigned char c1 = static_cast<unsigned char>(s[1]);
        if ((c1 & 0xC0) == 0x80) {
            bytes_read = 2;
            return ((c0 & 0x1F) << 6) | (c1 & 0x3F);
        }
    } else if ((c0 & 0xF0) == 0xE0 && len >= 3) {
        const unsigned char c1 = static_cast<unsigned char>(s[1]);
        const unsigned char c2 = static_cast<unsigned char>(s[2]);
        if ((c1 & 0xC0) == 0x80 && (c2 & 0xC0) == 0x80) {
            bytes_read = 3;
            return ((c0 & 0x0F) << 12) | ((c1 & 0x3F) << 6) | (c2 & 0x3F);
        }
    } else if ((c0 & 0xF8) == 0xF0 && len >= 4) {
        const unsigned char c1 = static_cast<unsigned char>(s[1]);
        const unsigned char c2 = static_cast<unsigned char>(s[2]);
        const unsigned char c3 = static_cast<unsigned char>(s[3]);
        if ((c1 & 0xC0) == 0x80 && (c2 & 0xC0) == 0x80 && (c3 & 0xC0) == 0x80) {
            bytes_read = 4;
            return ((c0 & 0x07) << 18) | ((c1 & 0x3F) << 12) | ((c2 & 0x3F) << 6) | (c3 & 0x3F);
        }
    }
    bytes_read = 1;
    return c0;
}

} // namespace

// ============================================================================
// IncrementalDecoder
// ============================================================================

IncrementalDecoder::IncrementalDecoder(const Tokenizer* tokenizer, bool skip_special_tokens)
    : tokenizer_(tokenizer), skip_special_tokens_(skip_special_tokens) {}

IncrementalDecoder::~IncrementalDecoder() = default;

void IncrementalDecoder::reset() {
    pending_.clear();
}

std::string IncrementalDecoder::add(int32_t token_id) {
    if (!tokenizer_) return "";

    if (skip_special_tokens_ && tokenizer_->is_special(token_id)) {
        return "";
    }

    std::string raw_bytes = tokenizer_->decode_raw_bytes(token_id);
    pending_.append(raw_bytes);

    // Scan for complete UTF-8 sequences from start of pending_
    size_t cursor = 0;
    const size_t n = pending_.size();
    const char* data = pending_.data();

    while (cursor < n) {
        const unsigned char c0 = static_cast<unsigned char>(data[cursor]);
        size_t expected_len = 1;
        if (c0 < 0x80) {
            expected_len = 1;
        } else if ((c0 & 0xE0) == 0xC0) {
            expected_len = 2;
        } else if ((c0 & 0xF0) == 0xE0) {
            expected_len = 3;
        } else if ((c0 & 0xF8) == 0xF0) {
            expected_len = 4;
        } else {
            // Invalid lead byte - emit replacement char and advance 1 byte
            cursor += 1;
            continue;
        }

        if (cursor + expected_len > n) {
            // Incomplete UTF-8 sequence at the end of buffer: wait for more bytes
            break;
        }

        // Validate continuation bytes
        bool valid = true;
        for (size_t k = 1; k < expected_len; ++k) {
            if ((static_cast<unsigned char>(data[cursor + k]) & 0xC0) != 0x80) {
                valid = false;
                break;
            }
        }

        if (valid) {
            cursor += expected_len;
        } else {
            // Invalid continuation, advance 1 byte
            cursor += 1;
        }
    }

    if (cursor > 0) {
        std::string emitted = pending_.substr(0, cursor);
        pending_.erase(0, cursor);
        return emitted;
    }

    return "";
}

std::string IncrementalDecoder::flush() {
    if (pending_.empty()) return "";
    std::string res;
    // For any remaining incomplete bytes, emit U+FFFD replacement
    size_t cursor = 0;
    while (cursor < pending_.size()) {
        const unsigned char c = static_cast<unsigned char>(pending_[cursor]);
        if (c < 0x80) {
            res.push_back(static_cast<char>(c));
            cursor++;
        } else {
            // Incomplete sequence at EOF -> emit U+FFFD
            res += "\xEF\xBF\xBD";
            cursor++;
        }
    }
    pending_.clear();
    return res;
}

// ============================================================================
// Tokenizer
// ============================================================================

Tokenizer::Tokenizer() {
    init_byte_encoder();
}

Tokenizer::~Tokenizer() = default;

void Tokenizer::init_byte_encoder() {
    std::vector<int> bs;
    for (int b = 0x21; b <= 0x7E; ++b) bs.push_back(b);
    for (int b = 0xA1; b <= 0xAC; ++b) bs.push_back(b);
    for (int b = 0xAE; b <= 0xFF; ++b) bs.push_back(b);
    std::vector<int> cs = bs;
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        bool found = false;
        for (int x : bs) {
            if (x == b) {
                found = true;
                break;
            }
        }
        if (!found) {
            bs.push_back(b);
            cs.push_back(256 + n);
            n++;
        }
    }

    unicode_to_byte_.assign(512, 0);
    byte_to_unicode_.assign(256, "");
    for (size_t i = 0; i < bs.size(); ++i) {
        uint8_t b = static_cast<uint8_t>(bs[i]);
        uint32_t c = static_cast<uint32_t>(cs[i]);
        if (c < static_cast<uint32_t>(unicode_to_byte_.size())) {
            unicode_to_byte_[c] = b;
        }
        std::string s;
        if (c < 128) {
            s.push_back(static_cast<char>(c));
        } else if (c < 2048) {
            s.push_back(static_cast<char>(0xC0 | (c >> 6)));
            s.push_back(static_cast<char>(0x80 | (c & 0x3F)));
        }
        byte_to_unicode_[b] = s;
    }
}

bool Tokenizer::load(const std::string& tokenizer_dir, std::string& err_msg) {
    loaded_ = false;
    err_msg.clear();
    vocab_tokens_.clear();
    token_to_id_.clear();
    merges_.clear();
    special_tokens_.clear();
    is_special_map_.clear();

    if (tokenizer_dir.empty()) {
        err_msg = "empty tokenizer directory";
        return false;
    }

    std::filesystem::path dir(tokenizer_dir);
    if (!std::filesystem::is_directory(dir)) {
        err_msg = "tokenizer directory does not exist: " + tokenizer_dir;
        return false;
    }

    // 1. Verify and parse tokenizer.json
    std::filesystem::path tok_info_path = dir / "tokenizer.json";
    std::ifstream tif(tok_info_path);
    if (!tif.is_open()) {
        err_msg = "missing tokenizer metadata: " + tok_info_path.string();
        return false;
    }

    std::string tok_info_content((std::istreambuf_iterator<char>(tif)), std::istreambuf_iterator<char>());
    server::json::JsonValue tok_info;
    std::string parse_err;
    if (!server::json::JsonValue::parse(tok_info_content, tok_info, parse_err) || !tok_info.is_object()) {
        err_msg = "malformed tokenizer.json: " + parse_err;
        return false;
    }

    std::string model_type = tok_info["model"].as_string();
    if (model_type != "gpt2") {
        err_msg = "unsupported tokenizer model type: " + model_type + " (expected 'gpt2')";
        return false;
    }

    std::string pre_type = tok_info["pre"].as_string();
    if (pre_type != "qwen35") {
        err_msg = "unsupported pre-tokenization rule: " + pre_type + " (expected 'qwen35')";
        return false;
    }

    int64_t expected_vocab_size = tok_info["vocab_size"].as_int(-1);
    if (expected_vocab_size <= 0) {
        err_msg = "invalid or missing vocab_size in tokenizer.json";
        return false;
    }

    int64_t expected_merges = tok_info["n_merges"].as_int(-1);
    if (expected_merges <= 0) {
        err_msg = "invalid or missing n_merges in tokenizer.json";
        return false;
    }

    if (tok_info.contains("special_ids") && tok_info["special_ids"].is_object()) {
        const auto& s_ids = tok_info["special_ids"];
        if (s_ids.contains("tokenizer.ggml.eos_token_id")) {
            eos_token_id_ = static_cast<int32_t>(s_ids["tokenizer.ggml.eos_token_id"].as_int(248046));
        }
    }

    // 2. Load vocab.json
    std::filesystem::path vocab_path = dir / "vocab.json";
    std::ifstream vf(vocab_path);
    if (!vf.is_open()) {
        err_msg = "failed to open " + vocab_path.string();
        return false;
    }

    std::string vocab_content((std::istreambuf_iterator<char>(vf)), std::istreambuf_iterator<char>());
    server::json::JsonValue vocab_root;
    if (!server::json::JsonValue::parse(vocab_content, vocab_root, parse_err) || !vocab_root.is_object()) {
        err_msg = "malformed vocab.json: " + parse_err;
        return false;
    }

    if (static_cast<int64_t>(vocab_root.obj_val.size()) != expected_vocab_size) {
        err_msg = "vocab.json size (" + std::to_string(vocab_root.obj_val.size()) +
                  ") does not match expected vocab_size (" + std::to_string(expected_vocab_size) + ")";
        return false;
    }

    vocab_tokens_.resize(static_cast<size_t>(expected_vocab_size));
    for (const auto& kv : vocab_root.obj_val) {
        int64_t id = kv.second.as_int(-1);
        if (id < 0 || id >= expected_vocab_size) {
            err_msg = "token id out of range in vocab.json: " + std::to_string(id);
            return false;
        }
        vocab_tokens_[static_cast<size_t>(id)] = kv.first;
        token_to_id_[kv.first] = static_cast<int32_t>(id);
    }

    // 3. Load merges.txt
    std::filesystem::path merges_path = dir / "merges.txt";
    std::ifstream mf(merges_path);
    if (!mf.is_open()) {
        err_msg = "failed to open " + merges_path.string();
        return false;
    }

    std::string line;
    int32_t rank = 0;
    while (std::getline(mf, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        size_t sp = line.find(' ');
        if (sp == std::string::npos) continue;
        std::string first = line.substr(0, sp);
        std::string second = line.substr(sp + 1);
        merges_[std::make_pair(std::move(first), std::move(second))] = rank++;
    }

    if (static_cast<int64_t>(rank) != expected_merges) {
        err_msg = "merges.txt count (" + std::to_string(rank) +
                  ") does not match expected n_merges (" + std::to_string(expected_merges) + ")";
        return false;
    }

    // 4. Load token_type.json
    std::filesystem::path type_path = dir / "token_type.json";
    std::ifstream tf(type_path);
    if (!tf.is_open()) {
        err_msg = "failed to open " + type_path.string();
        return false;
    }

    std::string type_content((std::istreambuf_iterator<char>(tf)), std::istreambuf_iterator<char>());
    server::json::JsonValue type_root;
    if (!server::json::JsonValue::parse(type_content, type_root, parse_err) || !type_root.is_array()) {
        err_msg = "malformed token_type.json: " + parse_err;
        return false;
    }

    for (size_t id = 0; id < type_root.arr_val.size() && id < vocab_tokens_.size(); ++id) {
        int64_t ttype = type_root.arr_val[id].as_int(1);
        // Type 3 = Control, Type 4 = User Defined
        if (ttype == 3 || ttype == 4) {
            const std::string& tok_str = vocab_tokens_[id];
            special_tokens_[tok_str] = static_cast<int32_t>(id);
            is_special_map_[static_cast<int32_t>(id)] = true;
        }
    }

    // 5. Load chat_template.jinja
    std::string chat_err;
    if (!chat_template_.load(tokenizer_dir, chat_err)) {
        err_msg = "failed to load chat template: " + chat_err;
        return false;
    }

    // Set standard EOS token IDs
    eos_token_ids_.clear();
    eos_token_ids_.push_back(eos_token_id_);
    auto it_endoftext = token_to_id_.find("<|endoftext|>");
    if (it_endoftext != token_to_id_.end() && it_endoftext->second != eos_token_id_) {
        eos_token_ids_.push_back(it_endoftext->second);
    }

    loaded_ = true;
    return true;
}

bool Tokenizer::is_special(int32_t token_id) const {
    auto it = is_special_map_.find(token_id);
    return it != is_special_map_.end() && it->second;
}

std::vector<std::string> Tokenizer::pre_tokenize_qwen(const std::string& text) const {
    std::vector<std::string> tokens;
    const size_t n = text.size();
    size_t i = 0;

    while (i < n) {
        // Branch 1: Contractions (e.g. 's, 't, 're, 've, 'm, 'll, 'd)
        if (text[i] == '\'') {
            bool matched = false;
            if (i + 1 < n) {
                char c1 = text[i + 1];
                if (c1 == 's' || c1 == 'S' || c1 == 't' || c1 == 'T' ||
                    c1 == 'm' || c1 == 'M' || c1 == 'd' || c1 == 'D') {
                    tokens.push_back(text.substr(i, 2));
                    i += 2;
                    matched = true;
                }
            }
            if (matched) continue;
            if (i + 2 < n) {
                char c1 = text[i + 1];
                char c2 = text[i + 2];
                if ((c1 == 'r' || c1 == 'R') && (c2 == 'e' || c2 == 'E')) {
                    tokens.push_back(text.substr(i, 3));
                    i += 3;
                    matched = true;
                } else if ((c1 == 'v' || c1 == 'V') && (c2 == 'e' || c2 == 'E')) {
                    tokens.push_back(text.substr(i, 3));
                    i += 3;
                    matched = true;
                } else if ((c1 == 'l' || c1 == 'L') && (c2 == 'l' || c2 == 'L')) {
                    tokens.push_back(text.substr(i, 3));
                    i += 3;
                    matched = true;
                }
            }
            if (matched) continue;
        }

        size_t b_read = 0;
        uint32_t cp = decode_utf8_codepoint(text.data() + i, n - i, b_read);

        // Branch 2: [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
        if (text[i] != '\r' && text[i] != '\n' && !unicode::is_letter(cp) && !unicode::is_number(cp)) {
            size_t next_read = 0;
            if (i + b_read < n) {
                uint32_t next_cp = decode_utf8_codepoint(text.data() + i + b_read, n - (i + b_read), next_read);
                if (unicode::is_letter(next_cp) || unicode::is_mark(next_cp)) {
                    size_t start = i;
                    i += b_read;
                    while (i < n) {
                        size_t cr = 0;
                        uint32_t c = decode_utf8_codepoint(text.data() + i, n - i, cr);
                        if (!unicode::is_letter(c) && !unicode::is_mark(c)) break;
                        i += cr;
                    }
                    tokens.push_back(text.substr(start, i - start));
                    continue;
                }
            }
        }
        if (unicode::is_letter(cp) || unicode::is_mark(cp)) {
            size_t start = i;
            i += b_read;
            while (i < n) {
                size_t cr = 0;
                uint32_t c = decode_utf8_codepoint(text.data() + i, n - i, cr);
                if (!unicode::is_letter(c) && !unicode::is_mark(c)) break;
                i += cr;
            }
            tokens.push_back(text.substr(start, i - start));
            continue;
        }

        // Branch 3: \p{N}
        if (unicode::is_number(cp)) {
            tokens.push_back(text.substr(i, b_read));
            i += b_read;
            continue;
        }

        // Branch 4:  ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
        if (text[i] == ' ') {
            size_t next_read = 0;
            if (i + 1 < n) {
                uint32_t next_cp = decode_utf8_codepoint(text.data() + i + 1, n - (i + 1), next_read);
                if (!unicode::is_whitespace(next_cp) && !unicode::is_letter(next_cp) &&
                    !unicode::is_mark(next_cp) && !unicode::is_number(next_cp)) {
                    size_t start = i;
                    i += 1;
                    while (i < n) {
                        size_t cr = 0;
                        uint32_t c = decode_utf8_codepoint(text.data() + i, n - i, cr);
                        if (unicode::is_whitespace(c) || unicode::is_letter(c) ||
                            unicode::is_mark(c) || unicode::is_number(c)) break;
                        i += cr;
                    }
                    while (i < n && (text[i] == '\r' || text[i] == '\n')) {
                        i += 1;
                    }
                    tokens.push_back(text.substr(start, i - start));
                    continue;
                }
            }
        } else if (!unicode::is_whitespace(cp) && !unicode::is_letter(cp) &&
                   !unicode::is_mark(cp) && !unicode::is_number(cp)) {
            size_t start = i;
            i += b_read;
            while (i < n) {
                size_t cr = 0;
                uint32_t c = decode_utf8_codepoint(text.data() + i, n - i, cr);
                if (unicode::is_whitespace(c) || unicode::is_letter(c) ||
                    unicode::is_mark(c) || unicode::is_number(c)) break;
                i += cr;
            }
            while (i < n && (text[i] == '\r' || text[i] == '\n')) {
                i += 1;
            }
            tokens.push_back(text.substr(start, i - start));
            continue;
        }

        // Branch 5: \s*[\r\n]+
        if (unicode::is_whitespace(cp)) {
            size_t k = i;
            size_t last_nl = std::string::npos;
            while (k < n) {
                size_t cr = 0;
                uint32_t c = decode_utf8_codepoint(text.data() + k, n - k, cr);
                if (!unicode::is_whitespace(c)) break;
                if (text[k] == '\r' || text[k] == '\n') {
                    last_nl = k;
                }
                k += cr;
            }
            if (last_nl != std::string::npos) {
                tokens.push_back(text.substr(i, (last_nl + 1) - i));
                i = last_nl + 1;
                continue;
            }
        }

        // Branch 6: \s+(?!\S)
        if (unicode::is_whitespace(cp)) {
            size_t k = i;
            while (k < n) {
                size_t cr = 0;
                uint32_t c = decode_utf8_codepoint(text.data() + k, n - k, cr);
                if (!unicode::is_whitespace(c)) break;
                k += cr;
            }
            if (k == n) {
                tokens.push_back(text.substr(i, k - i));
                i = k;
                continue;
            }
            if (k - i >= 2) {
                tokens.push_back(text.substr(i, (k - 1) - i));
                i = k - 1;
                continue;
            }
        }

        // Branch 7: \s+
        if (unicode::is_whitespace(cp)) {
            size_t start = i;
            while (i < n) {
                size_t cr = 0;
                uint32_t c = decode_utf8_codepoint(text.data() + i, n - i, cr);
                if (!unicode::is_whitespace(c)) break;
                i += cr;
            }
            tokens.push_back(text.substr(start, i - start));
            continue;
        }

        // Fallback: 1 codepoint
        tokens.push_back(text.substr(i, b_read));
        i += b_read;
    }

    return tokens;
}

std::vector<int32_t> Tokenizer::bpe_merge_word(const std::string& word) const {
    if (word.empty()) return {};

    // Initial sequence of byte-encoded characters
    std::vector<std::string> word_tokens;
    word_tokens.reserve(word.size());
    for (unsigned char b : word) {
        if (b < byte_to_unicode_.size()) {
            word_tokens.push_back(byte_to_unicode_[b]);
        } else {
            word_tokens.push_back(std::string(1, static_cast<char>(b)));
        }
    }

    // Iteratively merge the pair with lowest rank
    while (word_tokens.size() > 1) {
        int32_t min_rank = -1;
        std::pair<std::string, std::string> best_pair;

        for (size_t i = 0; i < word_tokens.size() - 1; ++i) {
            auto pair = std::make_pair(word_tokens[i], word_tokens[i + 1]);
            auto it = merges_.find(pair);
            if (it != merges_.end()) {
                if (min_rank == -1 || it->second < min_rank) {
                    min_rank = it->second;
                    best_pair = std::move(pair);
                }
            }
        }

        if (min_rank == -1) break;

        std::vector<std::string> next_tokens;
        next_tokens.reserve(word_tokens.size());
        size_t i = 0;
        while (i < word_tokens.size()) {
            if (i < word_tokens.size() - 1 && word_tokens[i] == best_pair.first && word_tokens[i + 1] == best_pair.second) {
                next_tokens.push_back(best_pair.first + best_pair.second);
                i += 2;
            } else {
                next_tokens.push_back(word_tokens[i]);
                i += 1;
            }
        }
        word_tokens = std::move(next_tokens);
    }

    // Map merged tokens to IDs
    std::vector<int32_t> ids;
    ids.reserve(word_tokens.size());
    for (const auto& t : word_tokens) {
        auto it = token_to_id_.find(t);
        if (it != token_to_id_.end()) {
            ids.push_back(it->second);
        }
    }
    return ids;
}

std::vector<int32_t> Tokenizer::tokenize(const std::string& text) const {
    std::vector<int32_t> ids;
    std::string err;
    encode(text, ids, false, err);
    return ids;
}

bool Tokenizer::encode(const std::string& text,
                       std::vector<int32_t>& ids,
                       bool allow_special,
                       std::string& err_msg) const {
    ids.clear();
    err_msg.clear();

    if (text.empty()) return true;

    if (!loaded_) {
        err_msg = "tokenizer is not loaded";
        return false;
    }

    size_t pos = 0;
    const size_t n = text.size();

    while (pos < n) {
        if (allow_special && !special_tokens_.empty()) {
            size_t next_pos = std::string::npos;
            size_t spec_len = 0;
            int32_t spec_id = -1;

            for (const auto& sp : special_tokens_) {
                size_t f = text.find(sp.first, pos);
                if (f != std::string::npos) {
                    if (next_pos == std::string::npos || f < next_pos ||
                        (f == next_pos && sp.first.size() > spec_len)) {
                        next_pos = f;
                        spec_len = sp.first.size();
                        spec_id = sp.second;
                    }
                }
            }

            if (next_pos != std::string::npos) {
                if (next_pos > pos) {
                    std::string segment = text.substr(pos, next_pos - pos);
                    std::vector<std::string> words = pre_tokenize_qwen(segment);
                    for (const auto& w : words) {
                        std::vector<int32_t> w_ids = bpe_merge_word(w);
                        ids.insert(ids.end(), w_ids.begin(), w_ids.end());
                    }
                }
                ids.push_back(spec_id);
                pos = next_pos + spec_len;
                continue;
            }
        }

        // No more special tokens or allow_special == false
        std::string segment = text.substr(pos);
        std::vector<std::string> words = pre_tokenize_qwen(segment);
        for (const auto& w : words) {
            std::vector<int32_t> w_ids = bpe_merge_word(w);
            ids.insert(ids.end(), w_ids.begin(), w_ids.end());
        }
        break;
    }

    return true;
}

std::string Tokenizer::decode_raw_bytes(int32_t token_id) const {
    if (token_id < 0 || static_cast<size_t>(token_id) >= vocab_tokens_.size()) {
        return "";
    }
    const std::string& s = vocab_tokens_[static_cast<size_t>(token_id)];
    if (s.empty()) return "";

    // Fast path: pure ASCII without byte encoding characters
    bool all_ascii = true;
    for (unsigned char c : s) {
        if (c >= 0x80) {
            all_ascii = false;
            break;
        }
    }
    if (all_ascii) return s;

    // Convert byte-level Unicode characters back to raw bytes
    std::string out;
    size_t i = 0;
    while (i < s.size()) {
        unsigned char c = static_cast<unsigned char>(s[i++]);
        uint32_t codepoint = c;
        if ((c & 0xE0) == 0xC0 && i < s.size()) {
            codepoint = ((c & 0x1F) << 6) | (static_cast<unsigned char>(s[i++]) & 0x3F);
        } else if ((c & 0xF0) == 0xE0 && i + 1 < s.size()) {
            codepoint = ((c & 0x0F) << 12) |
                        ((static_cast<unsigned char>(s[i]) & 0x3F) << 6) |
                        (static_cast<unsigned char>(s[i + 1]) & 0x3F);
            i += 2;
        }

        if (codepoint < unicode_to_byte_.size() && unicode_to_byte_[codepoint] != 0) {
            out.push_back(static_cast<char>(unicode_to_byte_[codepoint]));
        } else {
            out.push_back(static_cast<char>(c));
        }
    }
    return out;
}

std::string Tokenizer::decode(int32_t token_id, bool skip_special_tokens) const {
    if (token_id < 0 || static_cast<size_t>(token_id) >= vocab_tokens_.size()) {
        return "";
    }
    if (skip_special_tokens && is_special(token_id)) {
        return "";
    }
    return decode_raw_bytes(token_id);
}

std::string Tokenizer::decode(const std::vector<int32_t>& tokens, bool skip_special_tokens) const {
    std::string res;
    for (int32_t t : tokens) {
        res += decode(t, skip_special_tokens);
    }
    return res;
}

std::unique_ptr<IncrementalDecoder> Tokenizer::create_incremental_decoder(bool skip_special_tokens) const {
    return std::make_unique<IncrementalDecoder>(this, skip_special_tokens);
}

} // namespace guild::runtime
