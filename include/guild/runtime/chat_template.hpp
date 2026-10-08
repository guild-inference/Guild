#pragma once

#include <string>
#include <vector>

namespace guild::runtime {

struct ChatMessage {
    std::string role;
    std::string content;
};

enum class TemplateType {
    Unknown,
    Qwen,
    Ornith
};

class ChatTemplate {
public:
    ChatTemplate();
    ~ChatTemplate();

    // Loads chat_template.jinja from tokenizer_dir or file path
    bool load(const std::string& path_or_dir, std::string& err_msg);

    bool is_loaded() const { return loaded_; }
    TemplateType type() const { return type_; }
    const std::string& source() const { return source_; }

    // Renders messages according to the loaded template
    bool render(const std::vector<ChatMessage>& messages,
                bool add_generation_prompt,
                std::string& rendered,
                std::string& err_msg) const;

private:
    bool loaded_ = false;
    TemplateType type_ = TemplateType::Unknown;
    std::string source_;

    bool render_qwen(const std::vector<ChatMessage>& messages,
                     bool add_generation_prompt,
                     std::string& rendered,
                     std::string& err_msg) const;

    bool render_ornith(const std::vector<ChatMessage>& messages,
                      bool add_generation_prompt,
                      std::string& rendered,
                      std::string& err_msg) const;
};

} // namespace guild::runtime
