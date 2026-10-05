#include "guild/server/openai.hpp"
#include "guild/server/json.hpp"

#include <chrono>
#include <random>
#include <sstream>

namespace guild::server {

std::string OpenAiFormatter::generate_id(const std::string& prefix) {
    static thread_local std::mt19937_64 rng(std::random_device{}());
    static const char hex_chars[] = "0123456789abcdef";
    std::string id = prefix;
    for (int i = 0; i < 24; ++i) {
        id += hex_chars[rng() % 16];
    }
    return id;
}

bool OpenAiParser::parse_chat_request(const std::string& body,
                                      ChatCompletionRequest& req,
                                      std::string& err_type,
                                      std::string& err_param,
                                      std::string& err_code,
                                      std::string& err_msg) {
    json::JsonValue val;
    std::string parse_err;
    if (!json::JsonValue::parse(body, val, parse_err)) {
        err_type = "invalid_request_error";
        err_param = "";
        err_code = "parse_error";
        err_msg = "Failed to parse JSON body: " + parse_err;
        return false;
    }

    if (!val.is_object()) {
        err_type = "invalid_request_error";
        err_param = "";
        err_code = "invalid_type";
        err_msg = "Expected JSON object at root";
        return false;
    }

    static const std::vector<std::string> allowed_keys = {
        "model", "messages", "temperature", "top_p", "max_tokens",
        "max_completion_tokens", "stream", "stop", "seed"
    };

    std::string unsupported_key;
    if (!val.check_unsupported_keys(allowed_keys, unsupported_key)) {
        err_type = "invalid_request_error";
        err_param = unsupported_key;
        err_code = "unsupported_parameter";
        err_msg = "Unsupported parameter: '" + unsupported_key +
                  "'. Supported parameters: model, messages, temperature, top_p, max_tokens, max_completion_tokens, stream, stop, seed.";
        return false;
    }

    if (val.contains("model")) {
        if (!val["model"].is_string()) {
            err_type = "invalid_request_error";
            err_param = "model";
            err_code = "invalid_type";
            err_msg = "Field 'model' must be a string";
            return false;
        }
        req.model = val["model"].as_string();
    }

    if (!val.contains("messages") || !val["messages"].is_array()) {
        err_type = "invalid_request_error";
        err_param = "messages";
        err_code = "missing_required_field";
        err_msg = "Field 'messages' is required and must be an array";
        return false;
    }

    const auto& msgs = val["messages"];
    if (msgs.size() == 0) {
        err_type = "invalid_request_error";
        err_param = "messages";
        err_code = "empty_array";
        err_msg = "Array 'messages' cannot be empty";
        return false;
    }

    req.messages.clear();
    for (size_t i = 0; i < msgs.size(); ++i) {
        const auto& m = msgs[i];
        if (!m.is_object() || !m.contains("role") || !m.contains("content")) {
            err_type = "invalid_request_error";
            err_param = "messages";
            err_code = "invalid_type";
            err_msg = "Each item in 'messages' must be an object with 'role' and 'content'";
            return false;
        }
        ChatMessage cm;
        cm.role = m["role"].as_string();
        if (m["content"].is_string()) {
            cm.content = m["content"].as_string();
        } else if (m["content"].is_array()) {
            // Combine text parts if structured array
            for (size_t p = 0; p < m["content"].size(); ++p) {
                if (m["content"][p].is_object() && m["content"][p].contains("text")) {
                    cm.content += m["content"][p]["text"].as_string();
                }
            }
        }
        req.messages.push_back(std::move(cm));
    }

    if (val.contains("temperature")) {
        if (!val["temperature"].is_number()) {
            err_type = "invalid_request_error";
            err_param = "temperature";
            err_code = "invalid_type";
            err_msg = "Field 'temperature' must be a number";
            return false;
        }
        req.temperature = static_cast<float>(val["temperature"].as_double());
    }

    if (val.contains("top_p")) {
        if (!val["top_p"].is_number()) {
            err_type = "invalid_request_error";
            err_param = "top_p";
            err_code = "invalid_type";
            err_msg = "Field 'top_p' must be a number";
            return false;
        }
        req.top_p = static_cast<float>(val["top_p"].as_double());
    }

    if (val.contains("max_completion_tokens")) {
        if (!val["max_completion_tokens"].is_number()) {
            err_type = "invalid_request_error";
            err_param = "max_completion_tokens";
            err_code = "invalid_type";
            err_msg = "Field 'max_completion_tokens' must be an integer";
            return false;
        }
        req.max_tokens = static_cast<int>(val["max_completion_tokens"].as_int());
    } else if (val.contains("max_tokens")) {
        if (!val["max_tokens"].is_number()) {
            err_type = "invalid_request_error";
            err_param = "max_tokens";
            err_code = "invalid_type";
            err_msg = "Field 'max_tokens' must be an integer";
            return false;
        }
        req.max_tokens = static_cast<int>(val["max_tokens"].as_int());
    }

    if (val.contains("stream")) {
        if (!val["stream"].is_bool()) {
            err_type = "invalid_request_error";
            err_param = "stream";
            err_code = "invalid_type";
            err_msg = "Field 'stream' must be a boolean";
            return false;
        }
        req.stream = val["stream"].as_bool();
    }

    if (val.contains("stop")) {
        if (val["stop"].is_string()) {
            req.stop.push_back(val["stop"].as_string());
        } else if (val["stop"].is_array()) {
            for (size_t s = 0; s < val["stop"].size(); ++s) {
                if (val["stop"][s].is_string()) {
                    req.stop.push_back(val["stop"][s].as_string());
                }
            }
        }
    }

    if (val.contains("seed")) {
        if (val["seed"].is_number()) {
            req.seed = static_cast<int>(val["seed"].as_int());
        }
    }

    return true;
}

bool OpenAiParser::parse_completion_request(const std::string& body,
                                            CompletionRequest& req,
                                            std::string& err_type,
                                            std::string& err_param,
                                            std::string& err_code,
                                            std::string& err_msg) {
    json::JsonValue val;
    std::string parse_err;
    if (!json::JsonValue::parse(body, val, parse_err)) {
        err_type = "invalid_request_error";
        err_param = "";
        err_code = "parse_error";
        err_msg = "Failed to parse JSON body: " + parse_err;
        return false;
    }

    if (!val.is_object()) {
        err_type = "invalid_request_error";
        err_param = "";
        err_code = "invalid_type";
        err_msg = "Expected JSON object at root";
        return false;
    }

    static const std::vector<std::string> allowed_keys = {
        "model", "prompt", "temperature", "top_p", "max_tokens", "stream", "stop", "seed"
    };

    std::string unsupported_key;
    if (!val.check_unsupported_keys(allowed_keys, unsupported_key)) {
        err_type = "invalid_request_error";
        err_param = unsupported_key;
        err_code = "unsupported_parameter";
        err_msg = "Unsupported parameter: '" + unsupported_key +
                  "'. Supported parameters: model, prompt, temperature, top_p, max_tokens, stream, stop, seed.";
        return false;
    }

    if (val.contains("model")) {
        req.model = val["model"].as_string();
    }

    if (!val.contains("prompt")) {
        err_type = "invalid_request_error";
        err_param = "prompt";
        err_code = "missing_required_field";
        err_msg = "Field 'prompt' is required";
        return false;
    }

    if (val["prompt"].is_string()) {
        req.prompt = val["prompt"].as_string();
    } else if (val["prompt"].is_array()) {
        for (size_t i = 0; i < val["prompt"].size(); ++i) {
            if (val["prompt"][i].is_string()) {
                req.prompt += val["prompt"][i].as_string();
            }
        }
    } else {
        err_type = "invalid_request_error";
        err_param = "prompt";
        err_code = "invalid_type";
        err_msg = "Field 'prompt' must be a string or array of strings";
        return false;
    }

    if (val.contains("temperature") && val["temperature"].is_number()) {
        req.temperature = static_cast<float>(val["temperature"].as_double());
    }
    if (val.contains("top_p") && val["top_p"].is_number()) {
        req.top_p = static_cast<float>(val["top_p"].as_double());
    }
    if (val.contains("max_tokens") && val["max_tokens"].is_number()) {
        req.max_tokens = static_cast<int>(val["max_tokens"].as_int());
    }
    if (val.contains("stream") && val["stream"].is_bool()) {
        req.stream = val["stream"].as_bool();
    }
    if (val.contains("stop")) {
        if (val["stop"].is_string()) {
            req.stop.push_back(val["stop"].as_string());
        } else if (val["stop"].is_array()) {
            for (size_t s = 0; s < val["stop"].size(); ++s) {
                if (val["stop"][s].is_string()) {
                    req.stop.push_back(val["stop"][s].as_string());
                }
            }
        }
    }
    if (val.contains("seed") && val["seed"].is_number()) {
        req.seed = static_cast<int>(val["seed"].as_int());
    }

    return true;
}

std::string OpenAiFormatter::format_error(const std::string& message,
                                          const std::string& type,
                                          const std::string& param,
                                          const std::string& code) {
    std::ostringstream ss;
    ss << "{\n"
       << "  \"error\": {\n"
       << "    \"message\": \"" << json::JsonValue::escape_string(message) << "\",\n"
       << "    \"type\": \"" << json::JsonValue::escape_string(type) << "\",\n";
    if (param.empty()) {
        ss << "    \"param\": null,\n";
    } else {
        ss << "    \"param\": \"" << json::JsonValue::escape_string(param) << "\",\n";
    }
    ss << "    \"code\": \"" << json::JsonValue::escape_string(code) << "\"\n"
       << "  }\n"
       << "}";
    return ss.str();
}

std::string OpenAiFormatter::format_chat_completion(const std::string& id,
                                                    const std::string& model,
                                                    int64_t created,
                                                    const std::string& content,
                                                    const std::string& finish_reason,
                                                    int prompt_tokens,
                                                    int completion_tokens) {
    std::ostringstream ss;
    ss << "{\n"
       << "  \"id\": \"" << json::JsonValue::escape_string(id) << "\",\n"
       << "  \"object\": \"chat.completion\",\n"
       << "  \"created\": " << created << ",\n"
       << "  \"model\": \"" << json::JsonValue::escape_string(model) << "\",\n"
       << "  \"choices\": [\n"
       << "    {\n"
       << "      \"index\": 0,\n"
       << "      \"message\": {\n"
       << "        \"role\": \"assistant\",\n"
       << "        \"content\": \"" << json::JsonValue::escape_string(content) << "\"\n"
       << "      },\n"
       << "      \"finish_reason\": \"" << json::JsonValue::escape_string(finish_reason) << "\"\n"
       << "    }\n"
       << "  ],\n"
       << "  \"usage\": {\n"
       << "    \"prompt_tokens\": " << prompt_tokens << ",\n"
       << "    \"completion_tokens\": " << completion_tokens << ",\n"
       << "    \"total_tokens\": " << (prompt_tokens + completion_tokens) << "\n"
       << "  }\n"
       << "}";
    return ss.str();
}

std::string OpenAiFormatter::format_chat_chunk(const std::string& id,
                                               const std::string& model,
                                               int64_t created,
                                               const std::string& delta_role,
                                               const std::string& delta_content,
                                               const std::string& finish_reason) {
    std::ostringstream ss;
    ss << "{\"id\":\"" << json::JsonValue::escape_string(id)
       << "\",\"object\":\"chat.completion.chunk\",\"created\":" << created
       << ",\"model\":\"" << json::JsonValue::escape_string(model)
       << "\",\"choices\":[{\"index\":0,\"delta\":{";

    bool has_field = false;
    if (!delta_role.empty()) {
        ss << "\"role\":\"" << json::JsonValue::escape_string(delta_role) << "\"";
        has_field = true;
    }
    if (!delta_content.empty()) {
        if (has_field) ss << ",";
        ss << "\"content\":\"" << json::JsonValue::escape_string(delta_content) << "\"";
    }
    ss << "},\"finish_reason\":";
    if (finish_reason.empty()) {
        ss << "null";
    } else {
        ss << "\"" << json::JsonValue::escape_string(finish_reason) << "\"";
    }
    ss << "}]}";
    return ss.str();
}

std::string OpenAiFormatter::format_completion(const std::string& id,
                                               const std::string& model,
                                               int64_t created,
                                               const std::string& text,
                                               const std::string& finish_reason,
                                               int prompt_tokens,
                                               int completion_tokens) {
    std::ostringstream ss;
    ss << "{\n"
       << "  \"id\": \"" << json::JsonValue::escape_string(id) << "\",\n"
       << "  \"object\": \"text_completion\",\n"
       << "  \"created\": " << created << ",\n"
       << "  \"model\": \"" << json::JsonValue::escape_string(model) << "\",\n"
       << "  \"choices\": [\n"
       << "    {\n"
       << "      \"text\": \"" << json::JsonValue::escape_string(text) << "\",\n"
       << "      \"index\": 0,\n"
       << "      \"logprobs\": null,\n"
       << "      \"finish_reason\": \"" << json::JsonValue::escape_string(finish_reason) << "\"\n"
       << "    }\n"
       << "  ],\n"
       << "  \"usage\": {\n"
       << "    \"prompt_tokens\": " << prompt_tokens << ",\n"
       << "    \"completion_tokens\": " << completion_tokens << ",\n"
       << "    \"total_tokens\": " << (prompt_tokens + completion_tokens) << "\n"
       << "  }\n"
       << "}";
    return ss.str();
}

std::string OpenAiFormatter::format_completion_chunk(const std::string& id,
                                                     const std::string& model,
                                                     int64_t created,
                                                     const std::string& text,
                                                     const std::string& finish_reason) {
    std::ostringstream ss;
    ss << "{\"id\":\"" << json::JsonValue::escape_string(id)
       << "\",\"object\":\"text_completion\",\"created\":" << created
       << ",\"model\":\"" << json::JsonValue::escape_string(model)
       << "\",\"choices\":[{\"text\":\"" << json::JsonValue::escape_string(text)
       << "\",\"index\":0,\"logprobs\":null,\"finish_reason\":";
    if (finish_reason.empty()) {
        ss << "null";
    } else {
        ss << "\"" << json::JsonValue::escape_string(finish_reason) << "\"";
    }
    ss << "}]}";
    return ss.str();
}

std::string OpenAiFormatter::format_models(const std::vector<std::string>& model_names) {
    std::ostringstream ss;
    ss << "{\n"
       << "  \"object\": \"list\",\n"
       << "  \"data\": [\n";
    for (size_t i = 0; i < model_names.size(); ++i) {
        if (i > 0) ss << ",\n";
        ss << "    {\n"
           << "      \"id\": \"" << json::JsonValue::escape_string(model_names[i]) << "\",\n"
           << "      \"object\": \"model\",\n"
           << "      \"created\": 1728100000,\n"
           << "      \"owned_by\": \"guild\"\n"
           << "    }";
    }
    ss << "\n  ]\n}";
    return ss.str();
}

std::string OpenAiFormatter::format_health(const std::string& model_name, int64_t max_context, bool ready) {
    std::ostringstream ss;
    ss << "{\n"
       << "  \"status\": \"" << (ready ? "ok" : "loading") << "\",\n"
       << "  \"model\": \"" << json::JsonValue::escape_string(model_name) << "\",\n"
       << "  \"max_context\": " << max_context << ",\n"
       << "  \"service\": \"guild\"\n"
       << "}";
    return ss.str();
}

std::string OpenAiFormatter::render_chatml(const std::vector<ChatMessage>& messages) {
    std::string out;
    for (const auto& msg : messages) {
        out += "<|im_start|>" + msg.role + "\n" + msg.content + "<|im_end|>\n";
    }
    out += "<|im_start|>assistant\n";
    return out;
}

} // namespace guild::server
