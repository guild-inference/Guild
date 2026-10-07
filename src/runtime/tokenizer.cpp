#include "guild/runtime/tokenizer.hpp"
#include "guild/server/json.hpp"

#include <fstream>
#include <sstream>
#include <iostream>

namespace guild::runtime {

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
    if (tokenizer_dir.empty()) {
        err_msg = "empty tokenizer directory";
        return false;
    }

    std::string vocab_path = tokenizer_dir + "/vocab.json";
    std::ifstream vf(vocab_path);
    if (!vf.is_open()) {
        err_msg = "failed to open " + vocab_path;
        return false;
    }

    std::string content((std::istreambuf_iterator<char>(vf)), std::istreambuf_iterator<char>());
    server::json::JsonValue root;
    std::string parse_err;
    if (!server::json::JsonValue::parse(content, root, parse_err) || !root.is_object()) {
        err_msg = "malformed vocab.json: " + parse_err;
        return false;
    }

    vocab_tokens_.clear();
    token_to_id_.clear();
    vocab_tokens_.resize(root.obj_val.size() + 2048);

    for (const auto& kv : root.obj_val) {
        int64_t id = kv.second.as_int(-1);
        if (id >= 0) {
            if (static_cast<size_t>(id) >= vocab_tokens_.size()) {
                vocab_tokens_.resize(id + 2048);
            }
            vocab_tokens_[id] = kv.first;
            token_to_id_[kv.first] = static_cast<int32_t>(id);
            if (kv.first.rfind("<|", 0) == 0 && kv.first.find("|>") != std::string::npos) {
                special_tokens_[kv.first] = static_cast<int32_t>(id);
            } else if (kv.first == "<think>" || kv.first == "</think>") {
                special_tokens_[kv.first] = static_cast<int32_t>(id);
            }
        }
    }

    eos_token_ids_.clear();
    auto it_im_end = token_to_id_.find("<|im_end|>");
    if (it_im_end != token_to_id_.end()) {
        eos_token_id_ = it_im_end->second;
        eos_token_ids_.push_back(it_im_end->second);
    }
    auto it_endoftext = token_to_id_.find("<|endoftext|>");
    if (it_endoftext != token_to_id_.end()) {
        eos_token_ids_.push_back(it_endoftext->second);
    }

    loaded_ = true;
    return true;
}

std::string Tokenizer::decode(int32_t token_id) const {
    if (token_id < 0 || static_cast<size_t>(token_id) >= vocab_tokens_.size()) {
        return "";
    }
    const std::string& s = vocab_tokens_[token_id];
    if (s.empty()) return "";

    bool all_ascii = true;
    for (unsigned char c : s) {
        if (c >= 0x80) {
            all_ascii = false;
            break;
        }
    }
    if (all_ascii) return s;

    std::string out;
    size_t i = 0;
    while (i < s.size()) {
        unsigned char c = s[i++];
        uint32_t codepoint = c;
        if ((c & 0xE0) == 0xC0 && i < s.size()) {
            codepoint = ((c & 0x1F) << 6) | (s[i++] & 0x3F);
        } else if ((c & 0xF0) == 0xE0 && i + 1 < s.size()) {
            codepoint = ((c & 0x0F) << 12) | ((s[i] & 0x3F) << 6) | (s[i + 1] & 0x3F);
            i += 2;
        }

        if (codepoint < unicode_to_byte_.size() && unicode_to_byte_[codepoint] != 0) {
            out += static_cast<char>(unicode_to_byte_[codepoint]);
        } else {
            out += static_cast<char>(c);
        }
    }
    return out;
}

std::string Tokenizer::decode(const std::vector<int32_t>& tokens) const {
    std::string res;
    for (int32_t t : tokens) {
        res += decode(t);
    }
    return res;
}

std::vector<int32_t> Tokenizer::tokenize(const std::string& text) const {
    std::vector<int32_t> ids;
    if (text.empty()) return ids;

    if (!loaded_) {
        ids.reserve(text.size());
        for (unsigned char c : text) {
            ids.push_back(static_cast<int32_t>(c));
        }
        return ids;
    }

    auto tokenize_span = [&](const std::string& span) {
        if (span.empty()) return;
        // Map raw bytes to byte-level BPE unicode characters
        std::string mapped;
        mapped.reserve(span.size() * 2);
        for (unsigned char b : span) {
            if (b < byte_to_unicode_.size()) {
                mapped += byte_to_unicode_[b];
            } else {
                mapped += static_cast<char>(b);
            }
        }

        size_t pos = 0;
        while (pos < mapped.size()) {
            bool matched = false;
            size_t max_len = std::min(mapped.size() - pos, size_t(128));
            for (size_t len = max_len; len > 0; --len) {
                std::string sub = mapped.substr(pos, len);
                auto it = token_to_id_.find(sub);
                if (it != token_to_id_.end()) {
                    ids.push_back(it->second);
                    pos += len;
                    matched = true;
                    break;
                }
            }
            if (!matched) {
                // Advance by 1 UTF-8 character
                unsigned char c = static_cast<unsigned char>(mapped[pos]);
                size_t step = 1;
                if ((c & 0xE0) == 0xC0) step = 2;
                else if ((c & 0xF0) == 0xE0) step = 3;
                else if ((c & 0xF8) == 0xF0) step = 4;
                pos += std::min(step, mapped.size() - pos);
            }
        }
    };

    // Find special tokens in text
    size_t pos = 0;
    while (pos < text.size()) {
        size_t next_spec = std::string::npos;
        size_t spec_len = 0;
        int32_t spec_id = -1;

        for (const auto& sp : special_tokens_) {
            size_t f = text.find(sp.first, pos);
            if (f != std::string::npos) {
                if (next_spec == std::string::npos || f < next_spec || (f == next_spec && sp.first.size() > spec_len)) {
                    next_spec = f;
                    spec_len = sp.first.size();
                    spec_id = sp.second;
                }
            }
        }

        if (next_spec != std::string::npos) {
            if (next_spec > pos) {
                tokenize_span(text.substr(pos, next_spec - pos));
            }
            ids.push_back(spec_id);
            pos = next_spec + spec_len;
        } else {
            tokenize_span(text.substr(pos));
            break;
        }
    }

    return ids;
}

} // namespace guild::runtime
