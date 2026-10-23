#include "guild/runtime/chat_template.hpp"
#include "guild/server/json.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <unordered_map>

namespace guild::runtime {

namespace {

std::string trim_string(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\r\n");
    if (first == std::string::npos)
        return "";
    size_t last = str.find_last_not_of(" \t\r\n");
    return str.substr(first, (last - first + 1));
}

// Control token identifiers represented as constants
constexpr const char* kImStart = "<|im_start|>";
constexpr const char* kImEnd = "<|im_end|>";
constexpr const char* kThinkStart = "<think>";
constexpr const char* kThinkEnd = "</think>";

constexpr const char* kToolCallInstructions =
    "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
    "<tool_call>\n"
    "<function=example_function_name>\n"
    "<parameter=example_parameter_1>\n"
    "value_1\n"
    "</parameter>\n"
    "<parameter=example_parameter_2>\n"
    "This is the value for the second parameter\n"
    "that can span\n"
    "multiple lines\n"
    "</parameter>\n"
    "</function>\n"
    "</tool_call>\n\n"
    "<IMPORTANT>\n"
    "Reminder:\n"
    "- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within "
    "<tool_call></tool_call> XML tags\n"
    "- Required parameters MUST be specified\n"
    "- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT "
    "after\n"
    "- If there is no function call available, answer the question like normal with your current knowledge and do not "
    "tell the user about function calls\n"
    "</IMPORTANT>";

std::string serialize_tool_definition(const ToolDefinition& tool) {
    if (!tool.raw_json.empty()) {
        return tool.raw_json;
    }
    server::json::JsonValue obj(std::vector<std::pair<std::string, server::json::JsonValue>>{});
    obj["name"] = tool.name;
    obj["description"] = tool.description;
    server::json::JsonValue params;
    std::string perr;
    if (!tool.parameters_json.empty() && server::json::JsonValue::parse(tool.parameters_json, params, perr) &&
        params.is_object()) {
        obj["parameters"] = std::move(params);
    } else {
        server::json::JsonValue empty_params(std::vector<std::pair<std::string, server::json::JsonValue>>{});
        empty_params["type"] = "object";
        empty_params["properties"] =
            server::json::JsonValue(std::vector<std::pair<std::string, server::json::JsonValue>>{});
        obj["parameters"] = std::move(empty_params);
    }
    return obj.dump();
}

std::vector<ToolDefinition> filter_tools_by_choice(const std::vector<ToolDefinition>& tools,
                                                   const std::string& tool_choice) {
    if (tool_choice.empty() || tool_choice == "auto" || tool_choice == "required") {
        return tools;
    }
    if (tool_choice == "none") {
        return {};
    }
    std::vector<ToolDefinition> filtered;
    for (const auto& t : tools) {
        if (t.name == tool_choice) {
            filtered.push_back(t);
        }
    }
    return filtered.empty() ? tools : filtered;
}

std::vector<ChatMessage> order_tool_results(const std::vector<ChatMessage>& messages) {
    std::vector<ChatMessage> out = messages;
    size_t i = 0;
    while (i < out.size()) {
        if (out[i].role != "tool") {
            ++i;
            continue;
        }
        size_t j = i;
        while (j < out.size() && out[j].role == "tool") {
            ++j;
        }
        if (i > 0 && out[i - 1].role == "assistant" && !out[i - 1].tool_calls.empty()) {
            std::unordered_map<std::string, size_t> rank;
            const auto& calls = out[i - 1].tool_calls;
            for (size_t k = 0; k < calls.size(); ++k) {
                if (!calls[k].id.empty()) {
                    rank[calls[k].id] = k;
                }
            }
            if (!rank.empty()) {
                std::stable_sort(out.begin() + static_cast<std::ptrdiff_t>(i),
                                 out.begin() + static_cast<std::ptrdiff_t>(j),
                                 [&](const ChatMessage& a, const ChatMessage& b) {
                                     size_t ra = rank.count(a.tool_call_id) ? rank[a.tool_call_id] : calls.size();
                                     size_t rb = rank.count(b.tool_call_id) ? rank[b.tool_call_id] : calls.size();
                                     return ra < rb;
                                 });
            }
        }
        i = j;
    }
    return out;
}

bool render_turns(const std::vector<ChatMessage>& messages, size_t num_sys, std::ostringstream& out,
                  std::string& err_msg) {
    for (size_t i = num_sys; i < messages.size(); ++i) {
        const auto& msg = messages[i];
        const std::string content = trim_string(msg.content);

        if (msg.role == "user") {
            out << kImStart << "user\n" << content << kImEnd << "\n";
        } else if (msg.role == "assistant") {
            out << kImStart << "assistant\n";
            if (!msg.reasoning_content.empty()) {
                out << kThinkStart << "\n"
                    << trim_string(msg.reasoning_content) << "\n"
                    << kThinkEnd << "\n\n"
                    << content;
            } else {
                size_t t_pos = content.find(kThinkStart);
                if (t_pos != std::string::npos) {
                    out << content;
                } else {
                    out << kThinkStart << "\n\n" << kThinkEnd << "\n\n" << content;
                }
            }

            if (!msg.tool_calls.empty()) {
                for (size_t k = 0; k < msg.tool_calls.size(); ++k) {
                    const auto& tc = msg.tool_calls[k];
                    if (tc.function.name.empty()) {
                        err_msg = "Tool call is missing a function name.";
                        return false;
                    }
                    if (k == 0) {
                        if (!content.empty()) {
                            out << "\n\n<tool_call>\n<function=" << tc.function.name << ">\n";
                        } else {
                            out << "<tool_call>\n<function=" << tc.function.name << ">\n";
                        }
                    } else {
                        out << "\n<tool_call>\n<function=" << tc.function.name << ">\n";
                    }

                    if (!tc.function.arguments_json.empty()) {
                        server::json::JsonValue args_val;
                        std::string parse_err;
                        if (server::json::JsonValue::parse(tc.function.arguments_json, args_val, parse_err) &&
                            args_val.is_object()) {
                            for (const auto& kv : args_val.obj_val) {
                                out << "<parameter=" << kv.first << ">\n";
                                if (kv.second.is_string()) {
                                    out << kv.second.as_string();
                                } else {
                                    out << kv.second.dump();
                                }
                                out << "\n</parameter>\n";
                            }
                        } else {
                            err_msg = "Tool call arguments for function \"" + tc.function.name +
                                      "\" must be a valid JSON object.";
                            return false;
                        }
                    }
                    out << "</function>\n</tool_call>";
                }
            }
            out << kImEnd << "\n";
        } else if (msg.role == "tool") {
            if (i == num_sys || messages[i - 1].role != "tool") {
                out << kImStart << "user";
            }
            out << "\n<tool_response>\n" << content << "\n</tool_response>";
            if (i + 1 == messages.size() || messages[i + 1].role != "tool") {
                out << kImEnd << "\n";
            }
        } else {
            err_msg = "Unexpected message role: " + msg.role;
            return false;
        }
    }
    return true;
}

} // namespace

ChatTemplate::ChatTemplate() = default;
ChatTemplate::~ChatTemplate() = default;

bool ChatTemplate::load(const std::string& path_or_dir, std::string& err_msg) {
    loaded_ = false;
    type_ = TemplateType::Unknown;
    source_.clear();

    if (path_or_dir.empty()) {
        err_msg = "empty chat template path";
        return false;
    }

    std::filesystem::path p(path_or_dir);
    if (std::filesystem::is_directory(p)) {
        p = p / "chat_template.jinja";
    }

    std::ifstream f(p);
    if (!f.is_open()) {
        err_msg = "failed to open chat template file: " + p.string();
        return false;
    }

    std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (content.empty()) {
        err_msg = "chat template file is empty: " + p.string();
        return false;
    }

    return load_from_string(content, err_msg);
}

bool ChatTemplate::load_from_string(const std::string& content, std::string& err_msg) {
    loaded_ = false;
    type_ = TemplateType::Unknown;
    source_.clear();

    if (content.empty()) {
        err_msg = "chat template content is empty";
        return false;
    }

    source_ = content;

    // Detect template type from structure and identifiers
    if (source_.find("resolved_reasoning_effort") != std::string::npos ||
        source_.find("reasoning_instructions") != std::string::npos) {
        type_ = TemplateType::Qwen;
    } else if (source_.find("qwen") != std::string::npos ||
               (source_.find("im_start") != std::string::npos && source_.find("im_end") != std::string::npos &&
                source_.find("render_content") != std::string::npos)) {
        type_ = TemplateType::Ornith;
    } else {
        err_msg = "unsupported or unrecognized chat template format";
        return false;
    }

    loaded_ = true;
    return true;
}

bool ChatTemplate::render(const std::vector<ChatMessage>& messages, bool add_generation_prompt, std::string& rendered,
                          std::string& err_msg) const {
    ChatTemplateOptions opts;
    opts.add_generation_prompt = add_generation_prompt;
    return render(messages, opts, rendered, err_msg);
}

bool ChatTemplate::render(const std::vector<ChatMessage>& messages, const ChatTemplateOptions& options,
                          std::string& rendered, std::string& err_msg) const {
    rendered.clear();
    err_msg.clear();

    if (!loaded_) {
        err_msg = "chat template is not loaded";
        return false;
    }

    if (messages.empty()) {
        err_msg = "No messages provided.";
        return false;
    }

    switch (type_) {
    case TemplateType::Qwen: return render_qwen(messages, options, rendered, err_msg);
    case TemplateType::Ornith: return render_ornith(messages, options, rendered, err_msg);
    default: err_msg = "unsupported chat template type"; return false;
    }
}

bool ChatTemplate::render_qwen(const std::vector<ChatMessage>& raw_messages, const ChatTemplateOptions& options,
                               std::string& rendered, std::string& err_msg) const {
    std::vector<ChatMessage> messages = order_tool_results(raw_messages);

    // 1. Validate roles and locate system messages at the start
    size_t num_sys = 0;
    std::string merged_system;

    for (size_t i = 0; i < messages.size(); ++i) {
        const auto& role = messages[i].role;
        if (role == "system" || role == "developer") {
            if (i != num_sys) {
                err_msg = "System message must be at the beginning.";
                return false;
            }
            std::string content = trim_string(messages[i].content);
            if (!content.empty()) {
                if (!merged_system.empty())
                    merged_system += "\n";
                merged_system += content;
            }
            num_sys++;
        } else if (role == "user" || role == "assistant") {
            // Valid turn roles
        } else if (role == "tool") {
            if (i == num_sys) {
                err_msg = "Unexpected message role: tool";
                return false;
            }
        } else {
            err_msg = "Unexpected message role: " + role;
            return false;
        }
    }

    const bool thinking_enabled = options.enable_thinking && options.reasoning_effort != "none";
    std::string reasoning_instructions;
    if (thinking_enabled) {
        std::string effort = options.reasoning_effort.empty() ? "xhigh" : options.reasoning_effort;
        if (effort == "high")
            effort = "xhigh";
        if (effort == "xhigh") {
            reasoning_instructions =
                "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, "
                "consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final "
                "answer.";
        } else if (effort == "low") {
            reasoning_instructions =
                "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the "
                "conclusion without unnecessary elaboration.";
        } else if (effort == "medium") {
            reasoning_instructions = "";
        } else {
            err_msg = "Unexpected reasoning effort " + options.reasoning_effort +
                      ". Supported types are xhigh (default), medium, and low.";
            return false;
        }
    }

    const auto active_tools = filter_tools_by_choice(options.tools, options.tool_choice);

    std::ostringstream out;

    if (!active_tools.empty()) {
        out << kImStart << "system\n";
        if (!reasoning_instructions.empty()) {
            out << reasoning_instructions << "\n\n";
        }
        out << "# Tools\n\nYou have access to the following functions:\n\n<tools>";
        for (const auto& tool : active_tools) {
            out << "\n" << serialize_tool_definition(tool);
        }
        out << "\n</tools>" << kToolCallInstructions;
        if (!merged_system.empty()) {
            out << "\n\n" << merged_system;
        }
        if (options.tool_choice == "required") {
            out << "\n\nYou must call at least one function tool in your response.";
        } else if (!options.tool_choice.empty() && options.tool_choice != "auto" && options.tool_choice != "none") {
            out << "\n\nYou must call the function \"" << options.tool_choice << "\" in your response.";
        }
        out << kImEnd << "\n";
    } else if (!merged_system.empty() || !reasoning_instructions.empty()) {
        out << kImStart << "system\n";
        if (!reasoning_instructions.empty()) {
            out << reasoning_instructions;
            if (!merged_system.empty()) {
                out << "\n\n" << merged_system;
            }
        } else {
            out << merged_system;
        }
        out << kImEnd << "\n";
    }

    if (!render_turns(messages, num_sys, out, err_msg)) {
        return false;
    }

    if (options.add_generation_prompt) {
        out << kImStart << "assistant\n";
        if (!thinking_enabled) {
            out << kThinkStart << "\n\n" << kThinkEnd << "\n\n";
        } else {
            out << kThinkStart << "\n";
        }
    }

    rendered = out.str();
    return true;
}

bool ChatTemplate::render_ornith(const std::vector<ChatMessage>& raw_messages, const ChatTemplateOptions& options,
                                 std::string& rendered, std::string& err_msg) const {
    std::vector<ChatMessage> messages = order_tool_results(raw_messages);

    size_t num_sys = 0;
    std::string merged_system;

    for (size_t i = 0; i < messages.size(); ++i) {
        const auto& role = messages[i].role;
        if (role == "system" || role == "developer") {
            if (i != num_sys) {
                err_msg = "System message must be at the beginning.";
                return false;
            }
            std::string content = trim_string(messages[i].content);
            if (!content.empty()) {
                if (!merged_system.empty())
                    merged_system += "\n";
                merged_system += content;
            }
            num_sys++;
        } else if (role == "user" || role == "assistant") {
            // Valid turn roles
        } else if (role == "tool") {
            if (i == num_sys) {
                err_msg = "Unexpected message role: tool";
                return false;
            }
        } else {
            err_msg = "Unexpected message role: " + role;
            return false;
        }
    }

    const bool thinking_enabled = options.enable_thinking && options.reasoning_effort != "none";
    const auto active_tools = filter_tools_by_choice(options.tools, options.tool_choice);

    std::ostringstream out;

    if (!active_tools.empty()) {
        out << kImStart << "system\n"
            << "# Tools\n\nYou have access to the following functions:\n\n<tools>";
        for (const auto& tool : active_tools) {
            out << "\n" << serialize_tool_definition(tool);
        }
        out << "\n</tools>" << kToolCallInstructions;
        if (!merged_system.empty()) {
            out << "\n\n" << merged_system;
        }
        if (options.tool_choice == "required") {
            out << "\n\nYou must call at least one function tool in your response.";
        } else if (!options.tool_choice.empty() && options.tool_choice != "auto" && options.tool_choice != "none") {
            out << "\n\nYou must call the function \"" << options.tool_choice << "\" in your response.";
        }
        out << kImEnd << "\n";
    } else if (!merged_system.empty()) {
        out << kImStart << "system\n" << merged_system << kImEnd << "\n";
    }

    if (!render_turns(messages, num_sys, out, err_msg)) {
        return false;
    }

    if (options.add_generation_prompt) {
        out << kImStart << "assistant\n";
        if (!thinking_enabled) {
            out << kThinkStart << "\n\n" << kThinkEnd << "\n\n";
        } else {
            out << kThinkStart << "\n";
        }
    }

    rendered = out.str();
    return true;
}

} // namespace guild::runtime
