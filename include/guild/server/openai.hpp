#pragma once

#include "guild/server/json.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace guild::server {

struct ChatMessage {
    std::string role;
    std::string content;
};

struct ChatCompletionRequest {
    std::string model;
    std::vector<ChatMessage> messages;
    std::optional<float> temperature;
    std::optional<float> top_p;
    std::optional<int> max_tokens;
    bool stream = false;
    std::vector<std::string> stop;
    std::optional<int> seed;
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

class OpenAiParser {
public:
    static bool parse_chat_request(const std::string& body,
                                   ChatCompletionRequest& req,
                                   std::string& err_type,
                                   std::string& err_param,
                                   std::string& err_code,
                                   std::string& err_msg);

    static bool parse_completion_request(const std::string& body,
                                         CompletionRequest& req,
                                         std::string& err_type,
                                         std::string& err_param,
                                         std::string& err_code,
                                         std::string& err_msg);
};

class OpenAiFormatter {
public:
    static std::string generate_id(const std::string& prefix = "chatcmpl-");

    static std::string format_error(const std::string& message,
                                    const std::string& type = "invalid_request_error",
                                    const std::string& param = "",
                                    const std::string& code = "invalid_request");

    static std::string format_chat_completion(const std::string& id,
                                              const std::string& model,
                                              int64_t created,
                                              const std::string& content,
                                              const std::string& finish_reason,
                                              int prompt_tokens,
                                              int completion_tokens);

    static std::string format_chat_chunk(const std::string& id,
                                         const std::string& model,
                                         int64_t created,
                                         const std::string& delta_role,
                                         const std::string& delta_content,
                                         const std::string& finish_reason);

    static std::string format_completion(const std::string& id,
                                         const std::string& model,
                                         int64_t created,
                                         const std::string& text,
                                         const std::string& finish_reason,
                                         int prompt_tokens,
                                         int completion_tokens);

    static std::string format_completion_chunk(const std::string& id,
                                               const std::string& model,
                                               int64_t created,
                                               const std::string& text,
                                               const std::string& finish_reason);

    static std::string format_models(const std::vector<std::string>& model_names);

    static std::string format_health(const std::string& model_name, int64_t max_context, bool ready);

    static std::string render_chatml(const std::vector<ChatMessage>& messages);
};

} // namespace guild::server
