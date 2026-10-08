#include "guild/runtime/chat_template.hpp"

#include <filesystem>
#include <fstream>
#include <sstream>

namespace guild::runtime {

namespace {

std::string trim_string(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    size_t last = str.find_last_not_of(" \t\r\n");
    return str.substr(first, (last - first + 1));
}

// Control token identifiers represented as constants
constexpr const char* kImStart = "<|im_start|>";
constexpr const char* kImEnd = "<|im_end|>";
constexpr const char* kThinkStart = "<think>";
constexpr const char* kThinkEnd = "</think>";

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
        err_msg = "unsupported or unrecognized chat template format in " + p.string();
        return false;
    }

    loaded_ = true;
    return true;
}

bool ChatTemplate::render(const std::vector<ChatMessage>& messages,
                          bool add_generation_prompt,
                          std::string& rendered,
                          std::string& err_msg) const {
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
        case TemplateType::Qwen:
            return render_qwen(messages, add_generation_prompt, rendered, err_msg);
        case TemplateType::Ornith:
            return render_ornith(messages, add_generation_prompt, rendered, err_msg);
        default:
            err_msg = "unsupported chat template type";
            return false;
    }
}

bool ChatTemplate::render_qwen(const std::vector<ChatMessage>& messages,
                               bool add_generation_prompt,
                               std::string& rendered,
                               std::string& err_msg) const {
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
                if (!merged_system.empty()) merged_system += "\n";
                merged_system += content;
            }
            num_sys++;
        } else if (role == "user" || role == "assistant") {
            // Valid turn roles
        } else {
            err_msg = "Unexpected message role: " + role;
            return false;
        }
    }

    // Default Qwen reasoning instruction
    const std::string reasoning_instructions =
        "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, "
        "consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.";

    std::ostringstream out;

    // Render system message
    out << kImStart << "system\n";
    if (!reasoning_instructions.empty()) {
        out << reasoning_instructions;
        if (!merged_system.empty()) {
            out << "\n\n" << merged_system;
        }
    } else if (!merged_system.empty()) {
        out << merged_system;
    }
    out << kImEnd << "\n";

    // Render conversational turns
    for (size_t i = num_sys; i < messages.size(); ++i) {
        const auto& msg = messages[i];
        const std::string content = trim_string(msg.content);

        if (msg.role == "user") {
            out << kImStart << "user\n" << content << kImEnd << "\n";
        } else if (msg.role == "assistant") {
            out << kImStart << "assistant\n";
            // Check if thoughts are already in content
            size_t t_pos = content.find(kThinkStart);
            if (t_pos != std::string::npos) {
                out << content;
            } else {
                out << kThinkStart << "\n\n" << kThinkEnd << "\n\n" << content;
            }
            out << kImEnd << "\n";
        }
    }

    if (add_generation_prompt) {
        out << kImStart << "assistant\n" << kThinkStart << "\n";
    }

    rendered = out.str();
    return true;
}

bool ChatTemplate::render_ornith(const std::vector<ChatMessage>& messages,
                                 bool add_generation_prompt,
                                 std::string& rendered,
                                 std::string& err_msg) const {
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
                if (!merged_system.empty()) merged_system += "\n";
                merged_system += content;
            }
            num_sys++;
        } else if (role == "user" || role == "assistant") {
            // Valid turn roles
        } else {
            err_msg = "Unexpected message role: " + role;
            return false;
        }
    }

    std::ostringstream out;

    if (!merged_system.empty()) {
        out << kImStart << "system\n" << merged_system << kImEnd << "\n";
    }

    for (size_t i = num_sys; i < messages.size(); ++i) {
        const auto& msg = messages[i];
        const std::string content = trim_string(msg.content);

        if (msg.role == "user") {
            out << kImStart << "user\n" << content << kImEnd << "\n";
        } else if (msg.role == "assistant") {
            out << kImStart << "assistant\n";
            size_t t_pos = content.find(kThinkStart);
            if (t_pos != std::string::npos) {
                out << content;
            } else {
                out << kThinkStart << "\n\n" << kThinkEnd << "\n\n" << content;
            }
            out << kImEnd << "\n";
        }
    }

    if (add_generation_prompt) {
        out << kImStart << "assistant\n" << kThinkStart << "\n";
    }

    rendered = out.str();
    return true;
}

} // namespace guild::runtime
