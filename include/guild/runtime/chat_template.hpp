#pragma once

#include <string>
#include <vector>

namespace guild::runtime {

struct ToolCallFunction {
    std::string name;
    std::string arguments_json = "{}";
};

struct ToolCall {
    std::string id;
    std::string type = "function";
    ToolCallFunction function;
};

struct ToolDefinition {
    std::string type = "function"; // "function" or "custom"
    std::string name;              // flat name seen by model (e.g. "get_weather" or "ns.tool")
    std::string original_name;     // unqualified tool name when inside a namespace
    std::string namespace_name;    // optional namespace name
    std::string description;
    std::string parameters_json = "{}";
    std::string raw_json;
};

struct ChatMessage {
    std::string role;
    std::string content;
    std::string reasoning_content;
    std::vector<ToolCall> tool_calls;
    std::string tool_call_id;
    std::string name;
    std::string raw_model_output;

    ChatMessage() = default;
    ChatMessage(std::string r, std::string c) : role(std::move(r)), content(std::move(c)) {
    }
};

struct ChatTemplateOptions {
    bool add_generation_prompt = true;
    bool enable_thinking = true;
    std::string reasoning_effort = "xhigh";
    std::vector<ToolDefinition> tools;
    std::string tool_choice = "auto"; // "auto", "none", "required", or function name
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
    bool load_from_string(const std::string& content, std::string& err_msg);

    bool is_loaded() const {
        return loaded_;
    }
    TemplateType type() const {
        return type_;
    }
    const std::string& source() const {
        return source_;
    }

    // Renders messages according to the loaded template
    bool render(const std::vector<ChatMessage>& messages, bool add_generation_prompt, std::string& rendered,
                std::string& err_msg) const;

    // Renders messages with full tool definitions, tool_choice, and reasoning controls
    bool render(const std::vector<ChatMessage>& messages, const ChatTemplateOptions& options, std::string& rendered,
                std::string& err_msg) const;

private:
    bool loaded_ = false;
    TemplateType type_ = TemplateType::Unknown;
    std::string source_;

    bool render_qwen(const std::vector<ChatMessage>& messages, const ChatTemplateOptions& options,
                     std::string& rendered, std::string& err_msg) const;

    bool render_ornith(const std::vector<ChatMessage>& messages, const ChatTemplateOptions& options,
                       std::string& rendered, std::string& err_msg) const;
};

} // namespace guild::runtime
