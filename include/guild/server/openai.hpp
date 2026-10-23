#pragma once

#include "guild/runtime/chat_template.hpp"
#include "guild/server/json.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace guild::server {

using ChatMessage = guild::runtime::ChatMessage;
using ToolCall = guild::runtime::ToolCall;
using ToolCallFunction = guild::runtime::ToolCallFunction;
using ToolDefinition = guild::runtime::ToolDefinition;

struct ChatCompletionRequest {
    std::string model;
    std::vector<ChatMessage> messages;
    std::vector<ToolDefinition> tools;
    std::string tool_choice = "auto"; // "auto", "none", "required", or specific function name
    bool parallel_tool_calls = true;
    bool enable_thinking = true;
    std::string reasoning_effort = "xhigh";
    std::optional<int> reasoning_budget_tokens;
    std::optional<float> temperature;
    std::optional<float> top_p;
    std::optional<int> top_k;
    std::optional<int> max_tokens;
    bool stream = false;
    bool stream_include_usage = true;
    std::vector<std::string> stop;
    std::optional<int> seed;
    std::string response_format_type = "text";
};

struct CompletionRequest {
    std::string model;
    std::string prompt;
    std::optional<float> temperature;
    std::optional<float> top_p;
    std::optional<int> max_tokens;
    bool stream = false;
    std::vector<std::string> stop;
    std::optional<int> seed;
};

struct ResponsesRequest {
    ChatCompletionRequest chat;
    std::string instructions;
    bool store = false;
    bool include_encrypted_reasoning = false;
};

enum class ParsedEventKind {
    Reasoning,
    Content,
    ToolCallStart,
    ToolCallArgs,
    ToolCallComplete
};

struct ParsedOutputEvent {
    ParsedEventKind kind = ParsedEventKind::Content;
    std::string text;
    int tool_index = 0;
    ToolCall call;
};

class OpenAiOutputParser {
public:
    OpenAiOutputParser(bool prompt_opened_think, const std::vector<ToolDefinition>& tools = {},
                       bool stream_tools = true);

    std::vector<ParsedOutputEvent> feed(const std::string& delta, bool is_generated_token = true);
    std::vector<ParsedOutputEvent> finish();

    const std::string& reasoning_content() const {
        return reasoning_acc_;
    }
    const std::string& content() const {
        return content_acc_;
    }
    const std::string& raw_output() const {
        return raw_acc_;
    }
    const std::vector<ToolCall>& tool_calls() const {
        return completed_calls_;
    }
    bool has_tool_calls() const {
        return !completed_calls_.empty();
    }
    int reasoning_tokens() const {
        return reasoning_tokens_;
    }
    bool in_reasoning() const {
        return state_ == State::Reasoning || state_ == State::DetectStart;
    }

    static ToolCall parse_tool_call_body(const std::string& body,
                                         const std::unordered_map<std::string, ToolDefinition>& schemas,
                                         const std::string& call_id = "");

private:
    enum class State {
        DetectStart,
        Reasoning,
        Content,
        ToolCall
    };

    enum class ScanState {
        Name,
        Between,
        StringVal,
        RawVal,
        Done,
        JsonFallback
    };

    State state_ = State::Content;
    bool prompt_opened_think_ = false;
    bool stripped_initial_think_ = false;
    bool strip_leading_newlines_ = false;
    bool stream_tools_ = true;

    std::string buf_;
    std::string raw_acc_;
    std::string reasoning_acc_;
    std::string content_acc_;
    std::vector<ToolCall> completed_calls_;
    int reasoning_tokens_ = 0;
    int next_tool_index_ = 0;

    std::unordered_map<std::string, ToolDefinition> schemas_;

    // Incremental tool call streaming state
    size_t scan_pos_ = 0;
    ScanState scan_state_ = ScanState::Name;
    bool scan_announced_ = false;
    int scan_tool_index_ = 0;
    ToolCall scan_call_;
    bool scan_first_param_ = true;
    bool scan_val_started_ = false;
    std::unordered_map<std::string, std::string> scan_param_types_;

    void reset_tool_scan();
    void scan_tool_stream(const std::string& call_buf, bool is_complete, std::vector<ParsedOutputEvent>& out);
    void close_unfinished_tool_scan(const std::string& call_buf, std::vector<ParsedOutputEvent>& out);
};

class OpenAiParser {
public:
    static bool parse_chat_request(const std::string& body, ChatCompletionRequest& req, std::string& err_type,
                                   std::string& err_param, std::string& err_code, std::string& err_msg);

    static bool parse_completion_request(const std::string& body, CompletionRequest& req, std::string& err_type,
                                         std::string& err_param, std::string& err_code, std::string& err_msg);

    static bool parse_responses_request(const std::string& body, ResponsesRequest& req, std::string& err_type,
                                        std::string& err_param, std::string& err_code, std::string& err_msg);
};

class OpenAiFormatter {
public:
    static std::string generate_id(const std::string& prefix = "chatcmpl-");

    static std::string format_error(const std::string& message, const std::string& type = "invalid_request_error",
                                    const std::string& param = "", const std::string& code = "invalid_request");

    static std::string format_chat_completion(const std::string& id, const std::string& model, int64_t created,
                                              const std::string& content, const std::string& finish_reason,
                                              int prompt_tokens, int completion_tokens,
                                              const std::string& reasoning_content = "",
                                              const std::vector<ToolCall>& tool_calls = {}, int reasoning_tokens = 0,
                                              int cached_tokens = 0);

    static std::string format_chat_chunk(const std::string& id, const std::string& model, int64_t created,
                                         const std::string& delta_role, const std::string& delta_content,
                                         const std::string& finish_reason, const std::string& delta_reasoning = "");

    static std::string format_chat_tool_call_chunk(const std::string& id, const std::string& model, int64_t created,
                                                   int tool_index, const std::string& call_id,
                                                   const std::string& function_name,
                                                   const std::string& arguments_delta);

    static std::string format_chat_usage_chunk(const std::string& id, const std::string& model, int64_t created,
                                               const std::string& finish_reason, int prompt_tokens,
                                               int completion_tokens, int reasoning_tokens = 0, int cached_tokens = 0);

    static std::string format_responses_object(const std::string& id, const std::string& model, int64_t created,
                                               const std::string& content, const std::string& reasoning_content,
                                               const std::vector<ToolCall>& tool_calls,
                                               const std::string& finish_reason, int prompt_tokens,
                                               int completion_tokens, int reasoning_tokens = 0, int cached_tokens = 0);

    static std::string format_completion(const std::string& id, const std::string& model, int64_t created,
                                         const std::string& text, const std::string& finish_reason, int prompt_tokens,
                                         int completion_tokens);

    static std::string format_completion_chunk(const std::string& id, const std::string& model, int64_t created,
                                               const std::string& text, const std::string& finish_reason);

    static std::string format_models(const std::vector<std::string>& model_names);
    static std::string format_model(const std::string& model_name);

    static std::string format_health(const std::string& model_name, int64_t max_context, bool ready);

    static std::string render_chatml(const std::vector<ChatMessage>& messages,
                                     const runtime::ChatTemplateOptions& options = {});
};

} // namespace guild::server
