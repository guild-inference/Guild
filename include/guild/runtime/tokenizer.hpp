#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>

namespace guild::runtime {

class Tokenizer {
public:
    Tokenizer();
    ~Tokenizer();

    // Loads vocab.json and merges.txt from tokenizer_dir
    bool load(const std::string& tokenizer_dir, std::string& err_msg);

    bool is_loaded() const { return loaded_; }

    // Tokenizes raw text into token IDs
    std::vector<int32_t> tokenize(const std::string& text) const;

    // Decodes token ID into UTF-8 text string
    std::string decode(int32_t token_id) const;

    // Decodes sequence of token IDs into text string
    std::string decode(const std::vector<int32_t>& tokens) const;

    int32_t eos_token_id() const { return eos_token_id_; }
    const std::vector<int32_t>& eos_token_ids() const { return eos_token_ids_; }
    size_t vocab_size() const { return vocab_tokens_.size(); }

private:
    bool loaded_ = false;
    int32_t eos_token_id_ = 151643;
    std::vector<int32_t> eos_token_ids_ = {151643, 151645, 151644};
    std::vector<std::string> vocab_tokens_;
    std::unordered_map<std::string, int32_t> token_to_id_;
    std::vector<uint8_t> unicode_to_byte_;
    std::vector<std::string> byte_to_unicode_;
    std::unordered_map<std::string, int32_t> special_tokens_;

    void init_byte_encoder();
};

} // namespace guild::runtime
