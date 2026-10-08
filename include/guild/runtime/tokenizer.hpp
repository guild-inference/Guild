#pragma once

#include "guild/runtime/chat_template.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace guild::runtime {

class Tokenizer;

class IncrementalDecoder {
public:
    IncrementalDecoder(const Tokenizer* tokenizer, bool skip_special_tokens = true);
    ~IncrementalDecoder();

    // Appends a token and returns any newly completed valid UTF-8 text
    std::string add(int32_t token_id);

    // Flushes any remaining bytes in the buffer
    std::string flush();

    // Resets buffer state
    void reset();

private:
    const Tokenizer* tokenizer_ = nullptr;
    bool skip_special_tokens_ = true;
    std::string pending_;
};

class Tokenizer {
public:
    Tokenizer();
    ~Tokenizer();

    // Loads vocab.json, merges.txt, tokenizer.json, token_type.json, and chat_template.jinja
    bool load(const std::string& tokenizer_dir, std::string& err_msg);

    bool is_loaded() const { return loaded_; }

    // Tokenizes raw text into token IDs (defaults to allow_special = false)
    std::vector<int32_t> tokenize(const std::string& text) const;

    // Full encoding with control/special token handling and error reporting
    bool encode(const std::string& text,
                std::vector<int32_t>& ids,
                bool allow_special,
                std::string& err_msg) const;

    // Decodes single token ID into UTF-8 text string
    std::string decode(int32_t token_id, bool skip_special_tokens = true) const;

    // Decodes sequence of token IDs into text string
    std::string decode(const std::vector<int32_t>& tokens, bool skip_special_tokens = true) const;

    // Creates an incremental stateful decoder for streaming SSE
    std::unique_ptr<IncrementalDecoder> create_incremental_decoder(bool skip_special_tokens = true) const;

    const ChatTemplate& chat_template() const { return chat_template_; }

    int32_t eos_token_id() const { return eos_token_id_; }
    const std::vector<int32_t>& eos_token_ids() const { return eos_token_ids_; }
    size_t vocab_size() const { return vocab_tokens_.size(); }

    bool is_special(int32_t token_id) const;

    // Returns raw decoded bytes for a token without UTF-8 validation
    std::string decode_raw_bytes(int32_t token_id) const;

private:
    bool loaded_ = false;
    int32_t eos_token_id_ = 248046;
    std::vector<int32_t> eos_token_ids_ = {248046, 248044};
    std::vector<std::string> vocab_tokens_;
    std::unordered_map<std::string, int32_t> token_to_id_;
    std::vector<uint8_t> unicode_to_byte_;
    std::vector<std::string> byte_to_unicode_;
    std::unordered_map<std::string, int32_t> special_tokens_;
    std::unordered_map<int32_t, bool> is_special_map_;

    // Pair merges: (first, second) -> rank
    struct PairHash {
        size_t operator()(const std::pair<std::string, std::string>& p) const noexcept {
            size_t h1 = std::hash<std::string>{}(p.first);
            size_t h2 = std::hash<std::string>{}(p.second);
            return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
        }
    };
    std::unordered_map<std::pair<std::string, std::string>, int32_t, PairHash> merges_;

    ChatTemplate chat_template_;

    void init_byte_encoder();
    std::vector<std::string> pre_tokenize_qwen(const std::string& text) const;
    std::vector<int32_t> bpe_merge_word(const std::string& word) const;
};

} // namespace guild::runtime
