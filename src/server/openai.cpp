#include "guild/server/openai.hpp"
#include "guild/server/json.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstring>
#include <random>
#include <sstream>

namespace guild::server {

namespace {

std::string trim_str(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\r\n");
    if (first == std::string::npos)
        return "";
    size_t last = str.find_last_not_of(" \t\r\n");
    return str.substr(first, (last - first + 1));
}

std::string to_lower_ascii(std::string s) {
    for (char& c : s) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return s;
}

bool apply_reasoning_effort_string(const std::string& raw_effort, bool& enable_thinking, std::string& resolved_effort,
                                   std::string& err_msg) {
    const std::string key = to_lower_ascii(trim_str(raw_effort));
    if (key.empty()) {
        return true;
    }
    if (key == "none" || key == "off" || key == "minimal" || key == "disabled" || key == "false") {
        enable_thinking = false;
        resolved_effort = "none";
        return true;
    }
    if (key == "low") {
        enable_thinking = true;
        resolved_effort = "low";
        return true;
    }
    if (key == "medium") {
        enable_thinking = true;
        resolved_effort = "medium";
        return true;
    }
    if (key == "high" || key == "xhigh" || key == "max" || key == "maximum") {
        enable_thinking = true;
        resolved_effort = "xhigh";
        return true;
    }
    err_msg = "Unknown reasoning_effort '" + raw_effort + "': expected none, low, medium, high, or xhigh";
    return false;
}

bool parse_tools_array(const json::JsonValue& tools_val, std::vector<ToolDefinition>& out_tools, std::string& err_type,
                       std::string& err_param, std::string& err_code, std::string& err_msg) {
    if (tools_val.is_null()) {
        return true;
    }
    if (!tools_val.is_array()) {
        err_type = "invalid_request_error";
        err_param = "tools";
        err_code = "invalid_type";
        err_msg = "Field 'tools' must be an array of tool objects";
        return false;
    }

    for (size_t i = 0; i < tools_val.size(); ++i) {
        const auto& t = tools_val[i];
        if (!t.is_object()) {
            err_type = "invalid_request_error";
            err_param = "tools";
            err_code = "invalid_type";
            err_msg = "Each item in 'tools' must be an object";
            return false;
        }

        const std::string t_type = t.contains("type") && t["type"].is_string() ? t["type"].as_string() : "function";
        // Skip hosted non-function tools (e.g. web_search) gracefully in Responses mode if not function
        if (t_type != "function" && t_type != "custom") {
            continue;
        }

        const json::JsonValue& fn = (t.contains("function") && t["function"].is_object()) ? t["function"] : t;
        if (!fn.contains("name") || !fn["name"].is_string() || trim_str(fn["name"].as_string()).empty()) {
            err_type = "invalid_request_error";
            err_param = "tools";
            err_code = "invalid_tool";
            err_msg = "tools[" + std::to_string(i) + "] is missing a valid function 'name'";
            return false;
        }

        ToolDefinition def;
        def.type = "function";
        def.name = trim_str(fn["name"].as_string());
        if (fn.contains("description") && fn["description"].is_string()) {
            def.description = fn["description"].as_string();
        }

        json::JsonValue canonical_fn(std::vector<std::pair<std::string, json::JsonValue>>{});
        canonical_fn["name"] = def.name;
        if (fn.contains("description") && fn["description"].is_string()) {
            canonical_fn["description"] = def.description;
        }
        if (fn.contains("parameters") && !fn["parameters"].is_null()) {
            if (!fn["parameters"].is_object()) {
                err_type = "invalid_request_error";
                err_param = "tools";
                err_code = "invalid_tool";
                err_msg = "tools[" + std::to_string(i) + "].function.parameters must be a JSON object";
                return false;
            }
            def.parameters_json = fn["parameters"].dump();
            canonical_fn["parameters"] = fn["parameters"];
        } else if (fn.contains("input_schema") && fn["input_schema"].is_object()) {
            def.parameters_json = fn["input_schema"].dump();
            canonical_fn["parameters"] = fn["input_schema"];
        } else {
            json::JsonValue empty_params(std::vector<std::pair<std::string, json::JsonValue>>{});
            empty_params["type"] = "object";
            empty_params["properties"] = json::JsonValue(std::vector<std::pair<std::string, json::JsonValue>>{});
            def.parameters_json = empty_params.dump();
            canonical_fn["parameters"] = std::move(empty_params);
        }
        def.raw_json = canonical_fn.dump();
        out_tools.push_back(std::move(def));
    }
    return true;
}

bool parse_tool_choice_field(const json::JsonValue& tc_val, const std::vector<ToolDefinition>& tools,
                             std::string& out_choice, std::string& err_type, std::string& err_param,
                             std::string& err_code, std::string& err_msg) {
    if (tc_val.is_null()) {
        out_choice = "auto";
        return true;
    }
    if (tc_val.is_string()) {
        const std::string s = tc_val.as_string();
        if (s != "auto" && s != "none" && s != "required") {
            err_type = "invalid_request_error";
            err_param = "tool_choice";
            err_code = "invalid_tool_choice";
            err_msg = "Invalid tool_choice '" + s + "': expected 'auto', 'none', 'required', or a function object";
            return false;
        }
        out_choice = s;
        return true;
    }
    if (tc_val.is_object()) {
        std::string fn_name;
        if (tc_val.contains("function") && tc_val["function"].is_object() && tc_val["function"].contains("name") &&
            tc_val["function"]["name"].is_string()) {
            fn_name = trim_str(tc_val["function"]["name"].as_string());
        } else if (tc_val.contains("name") && tc_val["name"].is_string()) {
            fn_name = trim_str(tc_val["name"].as_string());
        }
        if (fn_name.empty()) {
            err_type = "invalid_request_error";
            err_param = "tool_choice";
            err_code = "invalid_tool_choice";
            err_msg = "Object 'tool_choice' must specify a non-empty function 'name'";
            return false;
        }
        if (!tools.empty()) {
            bool found = false;
            for (const auto& t : tools) {
                if (t.name == fn_name) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                err_type = "invalid_request_error";
                err_param = "tool_choice";
                err_code = "invalid_tool_choice";
                err_msg = "tool_choice requested function '" + fn_name + "' which is not in 'tools'";
                return false;
            }
        }
        out_choice = fn_name;
        return true;
    }
    err_type = "invalid_request_error";
    err_param = "tool_choice";
    err_code = "invalid_type";
    err_msg = "Field 'tool_choice' must be a string or object";
    return false;
}

constexpr const char* kThinkStartTag = "<think>";
constexpr const char* kThinkEndTag = "</think>";
constexpr const char* kCallStartTag = "<tool_call>";
constexpr const char* kCallEndTag = "</tool_call>";
constexpr const char* kParamEndTag = "</parameter>";
constexpr const char* kFuncEndTag = "</function>";

size_t hold_partial_tag_suffix(const std::string& text, const std::vector<std::string>& tags) {
    size_t best = 0;
    for (const auto& tag : tags) {
        const size_t max_n = std::min(text.size(), tag.size() - 1);
        for (size_t n = 1; n <= max_n; ++n) {
            if (text.compare(text.size() - n, n, tag, 0, n) == 0) {
                best = std::max(best, n);
            }
        }
    }
    return best;
}

// Finds where a <parameter=...> value ends: the first </parameter> followed (after whitespace)
// by <parameter= or </function>. Returns >= 0 on match, -1 if none yet, -2 if candidate follower is still arriving.
int find_param_end(const std::string& text, bool final_pass = false) {
    const std::string param_end_tag = kParamEndTag;
    const std::string param_start_prefix = "<parameter=";
    const std::string func_end_tag = kFuncEndTag;

    size_t at = text.find(param_end_tag);
    while (at != std::string::npos) {
        std::string after = text.substr(at + param_end_tag.size());
        size_t first_non_ws = after.find_first_not_of(" \t\r\n");
        std::string stripped = (first_non_ws == std::string::npos) ? "" : after.substr(first_non_ws);

        if (stripped.rfind(param_start_prefix, 0) == 0 || stripped.rfind(func_end_tag, 0) == 0) {
            return static_cast<int>(at);
        }
        if (stripped.empty() || param_start_prefix.rfind(stripped, 0) == 0 || func_end_tag.rfind(stripped, 0) == 0) {
            return final_pass ? static_cast<int>(at) : -2;
        }
        at = text.find(param_end_tag, at + 1);
    }
    return -1;
}

// Finds where a <tool_call>...</tool_call> body ends. Returns index of </tool_call> or -1 if incomplete.
int find_call_end(const std::string& text) {
    size_t first_non_ws = text.find_first_not_of(" \t\r\n");
    if (first_non_ws == std::string::npos)
        return -1;
    std::string s = text.substr(first_non_ws);
    size_t pos = first_non_ws;

    const std::string func_prefix = "<function=";
    const std::string param_prefix = "<parameter=";
    const std::string func_end = kFuncEndTag;
    const std::string call_end = kCallEndTag;

    if (s.rfind(func_prefix, 0) != 0) {
        if (func_prefix.rfind(s, 0) == 0)
            return -1;
        size_t ce = text.find(call_end);
        return ce == std::string::npos ? -1 : static_cast<int>(ce);
    }

    size_t gt = text.find('>', pos);
    if (gt == std::string::npos)
        return -1;
    pos = gt + 1;

    while (true) {
        std::string rest = text.substr(pos);
        size_t non_ws = rest.find_first_not_of(" \t\r\n");
        std::string stripped = (non_ws == std::string::npos) ? "" : rest.substr(non_ws);
        pos += (non_ws == std::string::npos) ? rest.size() : non_ws;

        if (stripped.rfind(param_prefix, 0) == 0) {
            gt = text.find('>', pos);
            if (gt == std::string::npos)
                return -1;
            int end = find_param_end(text.substr(gt + 1), false);
            if (end < 0)
                return -1;
            pos = gt + 1 + static_cast<size_t>(end) + std::strlen(kParamEndTag);
        } else if (stripped.rfind(func_end, 0) == 0) {
            std::string after_func = text.substr(pos + func_end.size());
            size_t nw = after_func.find_first_not_of(" \t\r\n");
            std::string after_stripped = (nw == std::string::npos) ? "" : after_func.substr(nw);
            if (after_stripped.rfind(call_end, 0) == 0) {
                return static_cast<int>(pos + func_end.size() + nw);
            }
            if (after_stripped.empty() || call_end.rfind(after_stripped, 0) == 0) {
                return -1;
            }
            size_t ce = text.find(call_end, pos);
            return ce == std::string::npos ? -1 : static_cast<int>(ce);
        } else if (stripped.empty() || param_prefix.rfind(stripped, 0) == 0 || func_end.rfind(stripped, 0) == 0) {
            return -1;
        } else {
            size_t ce = text.find(call_end, pos);
            return ce == std::string::npos ? -1 : static_cast<int>(ce);
        }
    }
}

std::unordered_map<std::string, std::string>
extract_schema_param_types(const std::string& func_name,
                           const std::unordered_map<std::string, ToolDefinition>& schemas) {
    std::unordered_map<std::string, std::string> types;
    auto it = schemas.find(func_name);
    if (it == schemas.end() || it->second.parameters_json.empty())
        return types;
    json::JsonValue schema_val;
    std::string err;
    if (!json::JsonValue::parse(it->second.parameters_json, schema_val, err) || !schema_val.is_object())
        return types;
    if (!schema_val.contains("properties") || !schema_val["properties"].is_object())
        return types;
    for (const auto& kv : schema_val["properties"].obj_val) {
        if (kv.second.is_object() && kv.second.contains("type") && kv.second["type"].is_string()) {
            types[kv.first] = kv.second["type"].as_string();
        }
    }
    return types;
}

} // namespace

std::string OpenAiFormatter::generate_id(const std::string& prefix) {
    static thread_local std::mt19937_64 rng(std::random_device{}());
    static const char hex_chars[] = "0123456789abcdef";
    std::string id = prefix;
    for (int i = 0; i < 24; ++i) {
        id += hex_chars[rng() % 16];
    }
    return id;
}

// ============================================================================
// OpenAiOutputParser
// ============================================================================

OpenAiOutputParser::OpenAiOutputParser(bool prompt_opened_think, const std::vector<ToolDefinition>& tools,
                                       bool stream_tools)
    : state_(prompt_opened_think ? State::Reasoning : State::DetectStart),
      prompt_opened_think_(prompt_opened_think),
      stream_tools_(stream_tools) {
    for (const auto& t : tools) {
        if (!t.name.empty()) {
            schemas_[t.name] = t;
        }
    }
    reset_tool_scan();
}

void OpenAiOutputParser::reset_tool_scan() {
    scan_pos_ = 0;
    scan_state_ = ScanState::Name;
    scan_announced_ = false;
    scan_tool_index_ = 0;
    scan_call_ = ToolCall{};
    scan_first_param_ = true;
    scan_val_started_ = false;
    scan_param_types_.clear();
}

ToolCall OpenAiOutputParser::parse_tool_call_body(const std::string& raw_body,
                                                  const std::unordered_map<std::string, ToolDefinition>& schemas,
                                                  const std::string& existing_call_id) {
    ToolCall call;
    call.id = existing_call_id.empty() ? OpenAiFormatter::generate_id("call_") : existing_call_id;
    call.type = "function";

    std::string body = trim_str(raw_body);

    // Format A: <function=NAME>\n<parameter=P>\nVALUE\n</parameter>...</function>
    if (body.rfind("<function=", 0) == 0) {
        size_t gt = body.find('>');
        if (gt == std::string::npos) {
            call.function.name = body.substr(10);
            call.function.arguments_json = "{}";
            return call;
        }
        call.function.name = trim_str(body.substr(10, gt - 10));
        std::string rest = body.substr(gt + 1);
        const auto param_types = extract_schema_param_types(call.function.name, schemas);

        json::JsonValue args_obj(std::vector<std::pair<std::string, json::JsonValue>>{});
        const std::string param_prefix = "<parameter=";
        while (true) {
            size_t ppos = rest.find(param_prefix);
            if (ppos == std::string::npos)
                break;
            rest = rest.substr(ppos + param_prefix.size());
            size_t pgt = rest.find('>');
            if (pgt == std::string::npos)
                break;
            std::string pname = trim_str(rest.substr(0, pgt));
            rest = rest.substr(pgt + 1);

            int end = find_param_end(rest, true);
            std::string val_str = (end >= 0) ? rest.substr(0, static_cast<size_t>(end)) : rest;
            rest = (end >= 0) ? rest.substr(static_cast<size_t>(end) + std::strlen(kParamEndTag)) : "";

            if (!val_str.empty() && val_str.front() == '\n')
                val_str.erase(0, 1);
            else if (val_str.size() >= 2 && val_str[0] == '\r' && val_str[1] == '\n')
                val_str.erase(0, 2);

            if (!val_str.empty() && val_str.back() == '\n') {
                val_str.pop_back();
                if (!val_str.empty() && val_str.back() == '\r')
                    val_str.pop_back();
            }

            auto tit = param_types.find(pname);
            const std::string declared_type = (tit != param_types.end()) ? tit->second : "";
            if (declared_type == "string") {
                args_obj[pname] = val_str;
            } else {
                json::JsonValue parsed_val;
                std::string perr;
                if (json::JsonValue::parse(val_str, parsed_val, perr)) {
                    args_obj[pname] = std::move(parsed_val);
                } else {
                    args_obj[pname] = val_str;
                }
            }
        }
        call.function.arguments_json = args_obj.dump();
        return call;
    }

    // Format B: JSON tool call {"name": "...", "arguments": {...}}
    json::JsonValue jval;
    std::string jerr;
    if (json::JsonValue::parse(body, jval, jerr) && jval.is_object()) {
        const json::JsonValue& fn =
            (jval.contains("function") && jval["function"].is_object()) ? jval["function"] : jval;
        if (fn.contains("name") && fn["name"].is_string()) {
            call.function.name = fn["name"].as_string();
        }
        if (fn.contains("arguments")) {
            if (fn["arguments"].is_string()) {
                const std::string raw_args = trim_str(fn["arguments"].as_string());
                call.function.arguments_json = raw_args.empty() ? "{}" : raw_args;
            } else if (fn["arguments"].is_object()) {
                call.function.arguments_json = fn["arguments"].dump();
            } else {
                call.function.arguments_json = "{}";
            }
        } else if (fn.contains("parameters") && fn["parameters"].is_object()) {
            call.function.arguments_json = fn["parameters"].dump();
        }
        return call;
    }

    call.function.arguments_json = "{}";
    return call;
}

void OpenAiOutputParser::scan_tool_stream(const std::string& call_buf, bool is_complete,
                                          std::vector<ParsedOutputEvent>& out) {
    auto emit_args = [&](const std::string& s) {
        if (!s.empty()) {
            ParsedOutputEvent ev;
            ev.kind = ParsedEventKind::ToolCallArgs;
            ev.text = s;
            ev.tool_index = scan_tool_index_;
            ev.call = scan_call_;
            out.push_back(std::move(ev));
        }
    };

    while (true) {
        if (scan_pos_ > call_buf.size())
            return;
        std::string rest = call_buf.substr(scan_pos_);

        if (scan_state_ == ScanState::Name) {
            size_t non_ws = rest.find_first_not_of(" \t\r\n");
            if (non_ws == std::string::npos)
                return;
            std::string stripped = rest.substr(non_ws);
            if (stripped.rfind("{", 0) == 0) {
                scan_state_ = ScanState::JsonFallback;
                return;
            }
            size_t a = rest.find("<function=");
            size_t b = (a != std::string::npos) ? rest.find('>', a + 10) : std::string::npos;
            if (b == std::string::npos)
                return;

            std::string name = trim_str(rest.substr(a + 10, b - (a + 10)));
            scan_tool_index_ = next_tool_index_++;
            scan_call_.id = OpenAiFormatter::generate_id("call_");
            scan_call_.type = "function";
            scan_call_.function.name = name;
            scan_call_.function.arguments_json = "";
            scan_param_types_ = extract_schema_param_types(name, schemas_);
            scan_announced_ = true;

            ParsedOutputEvent start_ev;
            start_ev.kind = ParsedEventKind::ToolCallStart;
            start_ev.tool_index = scan_tool_index_;
            start_ev.call = scan_call_;
            out.push_back(std::move(start_ev));

            emit_args("{");
            scan_pos_ += b + 1;
            scan_state_ = ScanState::Between;
        } else if (scan_state_ == ScanState::Between) {
            size_t non_ws = rest.find_first_not_of(" \t\r\n");
            if (non_ws == std::string::npos) {
                scan_pos_ += rest.size();
                return;
            }
            scan_pos_ += non_ws;
            std::string stripped = rest.substr(non_ws);
            if (stripped.rfind("<parameter=", 0) == 0) {
                size_t b = stripped.find('>');
                if (b == std::string::npos)
                    return;
                std::string pname = trim_str(stripped.substr(11, b - 11));
                std::string prefix =
                    (scan_first_param_ ? "" : ",") + std::string("\"") + json::JsonValue::escape_string(pname) + "\":";
                emit_args(prefix);
                scan_first_param_ = false;
                scan_pos_ += b + 1;

                auto tit = scan_param_types_.find(pname);
                if (tit != scan_param_types_.end() && tit->second == "string") {
                    emit_args("\"");
                    scan_state_ = ScanState::StringVal;
                    scan_val_started_ = false;
                } else {
                    scan_state_ = ScanState::RawVal;
                }
            } else if (stripped.rfind(kFuncEndTag, 0) == 0) {
                emit_args("}");
                scan_pos_ += std::strlen(kFuncEndTag);
                scan_state_ = ScanState::Done;
            } else {
                return;
            }
        } else if (scan_state_ == ScanState::StringVal) {
            if (!scan_val_started_) {
                if (rest.empty())
                    return;
                if (rest[0] == '\n') {
                    scan_pos_ += 1;
                    rest.erase(0, 1);
                } else if (rest.size() >= 2 && rest[0] == '\r' && rest[1] == '\n') {
                    scan_pos_ += 2;
                    rest.erase(0, 2);
                }
                scan_val_started_ = true;
            }
            int end = find_param_end(rest, is_complete);
            if (end >= 0) {
                std::string value = rest.substr(0, static_cast<size_t>(end));
                if (!value.empty() && value.back() == '\n') {
                    value.pop_back();
                    if (!value.empty() && value.back() == '\r')
                        value.pop_back();
                }
                emit_args(json::JsonValue::escape_string(value) + "\"");
                scan_pos_ += static_cast<size_t>(end) + std::strlen(kParamEndTag);
                scan_state_ = ScanState::Between;
                continue;
            }
            size_t safe = 0;
            if (end == -2) {
                size_t p = rest.find(kParamEndTag);
                safe = (p == std::string::npos) ? 0 : p;
            } else {
                size_t hold = hold_partial_tag_suffix(rest, {kParamEndTag});
                safe = rest.size() > hold ? (rest.size() - hold) : 0;
            }
            if (safe > 0 && rest[safe - 1] == '\n')
                --safe;
            if (safe > 0 && rest[safe - 1] == '\r')
                --safe;
            if (safe > 0) {
                emit_args(json::JsonValue::escape_string(rest.substr(0, safe)));
                scan_pos_ += safe;
            }
            return;
        } else if (scan_state_ == ScanState::RawVal) {
            int end = find_param_end(rest, is_complete);
            if (end < 0)
                return;
            std::string value = rest.substr(0, static_cast<size_t>(end));
            if (!value.empty() && value.front() == '\n')
                value.erase(0, 1);
            if (!value.empty() && value.back() == '\n') {
                value.pop_back();
                if (!value.empty() && value.back() == '\r')
                    value.pop_back();
            }
            json::JsonValue parsed_val;
            std::string perr;
            if (json::JsonValue::parse(value, parsed_val, perr)) {
                emit_args(parsed_val.dump());
            } else {
                emit_args("\"" + json::JsonValue::escape_string(value) + "\"");
            }
            scan_pos_ += static_cast<size_t>(end) + std::strlen(kParamEndTag);
            scan_state_ = ScanState::Between;
        } else {
            return;
        }
    }
}

void OpenAiOutputParser::close_unfinished_tool_scan(const std::string& call_buf, std::vector<ParsedOutputEvent>& out) {
    if (!scan_announced_ || scan_state_ == ScanState::Done || scan_state_ == ScanState::JsonFallback)
        return;
    std::string tail;
    if (scan_state_ == ScanState::StringVal) {
        tail += "\"";
    } else if (scan_state_ == ScanState::RawVal) {
        std::string rem = scan_pos_ < call_buf.size() ? trim_str(call_buf.substr(scan_pos_)) : "";
        tail += "\"" + json::JsonValue::escape_string(rem) + "\"";
    }
    tail += "}";
    ParsedOutputEvent ev;
    ev.kind = ParsedEventKind::ToolCallArgs;
    ev.text = tail;
    ev.tool_index = scan_tool_index_;
    ev.call = scan_call_;
    out.push_back(std::move(ev));
    scan_state_ = ScanState::Done;
}

std::vector<ParsedOutputEvent> OpenAiOutputParser::feed(const std::string& delta, bool is_generated_token) {
    if (is_generated_token && !delta.empty()) {
        raw_acc_ += delta;
    }
    if (is_generated_token && !delta.empty() &&
        (state_ == State::Reasoning || (state_ == State::DetectStart && delta.find('<') != std::string::npos))) {
        ++reasoning_tokens_;
    }

    buf_ += delta;
    std::vector<ParsedOutputEvent> out;

    while (true) {
        if (state_ == State::DetectStart) {
            size_t non_ws = buf_.find_first_not_of(" \t\r\n");
            if (non_ws == std::string::npos) {
                return out;
            }
            std::string trimmed_start = buf_.substr(non_ws);
            const std::string think_open = kThinkStartTag;
            if (trimmed_start.rfind(think_open, 0) == 0) {
                size_t erase_len = non_ws + think_open.size();
                if (erase_len < buf_.size() && buf_[erase_len] == '\n')
                    ++erase_len;
                buf_.erase(0, erase_len);
                stripped_initial_think_ = true;
                state_ = State::Reasoning;
                continue;
            }
            if (think_open.rfind(trimmed_start, 0) == 0) {
                // Still a prefix of "<think>"
                return out;
            }
            state_ = State::Content;
            continue;
        }

        if (state_ == State::Reasoning) {
            if (!stripped_initial_think_) {
                const std::string think_open = kThinkStartTag;
                if (buf_.rfind(think_open, 0) == 0) {
                    size_t erase_len = think_open.size();
                    if (erase_len < buf_.size() && buf_[erase_len] == '\n')
                        ++erase_len;
                    buf_.erase(0, erase_len);
                    stripped_initial_think_ = true;
                } else if (think_open.rfind(buf_, 0) == 0) {
                    return out;
                } else {
                    stripped_initial_think_ = true;
                }
            }

            if (reasoning_acc_.empty() && !buf_.empty() && (buf_.front() == '\n' || buf_.front() == '\r')) {
                size_t first_non_nl = buf_.find_first_not_of("\r\n");
                if (first_non_nl == std::string::npos) {
                    buf_.clear();
                    return out;
                }
                buf_.erase(0, first_non_nl);
            }

            size_t end_pos = buf_.find(kThinkEndTag);
            if (end_pos == std::string::npos) {
                size_t keep = hold_partial_tag_suffix(buf_, {kThinkEndTag});
                if (buf_.size() > keep) {
                    std::string chunk = buf_.substr(0, buf_.size() - keep);
                    reasoning_acc_ += chunk;
                    out.push_back({ParsedEventKind::Reasoning, chunk, 0, {}});
                    buf_.erase(0, buf_.size() - keep);
                }
                return out;
            }

            if (end_pos > 0) {
                std::string chunk = buf_.substr(0, end_pos);
                reasoning_acc_ += chunk;
                out.push_back({ParsedEventKind::Reasoning, chunk, 0, {}});
            }
            buf_.erase(0, end_pos + std::strlen(kThinkEndTag));
            state_ = State::Content;
            strip_leading_newlines_ = true;
            continue;
        }

        if (state_ == State::Content) {
            if (strip_leading_newlines_) {
                size_t first_non_nl = buf_.find_first_not_of("\r\n");
                if (first_non_nl == std::string::npos) {
                    buf_.clear();
                    return out;
                }
                buf_.erase(0, first_non_nl);
                strip_leading_newlines_ = false;
            }

            // Also handle an implicit </think> if prompt_opened_think was false and the script omitted <think>
            if (reasoning_acc_.empty() && content_acc_.empty()) {
                size_t stray_think_end = buf_.find(kThinkEndTag);
                if (stray_think_end != std::string::npos) {
                    std::string rchunk = buf_.substr(0, stray_think_end);
                    if (!rchunk.empty()) {
                        reasoning_acc_ += rchunk;
                        out.push_back({ParsedEventKind::Reasoning, rchunk, 0, {}});
                    }
                    buf_.erase(0, stray_think_end + std::strlen(kThinkEndTag));
                    strip_leading_newlines_ = true;
                    continue;
                }
            }

            size_t call_pos = buf_.find(kCallStartTag);
            if (call_pos == std::string::npos) {
                size_t keep = hold_partial_tag_suffix(buf_, {kCallStartTag, kThinkEndTag});
                size_t emit_end = buf_.size() > keep ? (buf_.size() - keep) : 0;
                // Hold trailing newlines right before a potential <tool_call> tag
                while (emit_end > 0 && (buf_[emit_end - 1] == '\n' || buf_[emit_end - 1] == '\r')) {
                    --emit_end;
                }
                if (emit_end > 0) {
                    std::string chunk = buf_.substr(0, emit_end);
                    content_acc_ += chunk;
                    out.push_back({ParsedEventKind::Content, chunk, 0, {}});
                    buf_.erase(0, emit_end);
                }
                return out;
            }

            if (call_pos > 0) {
                std::string before = buf_.substr(0, call_pos);
                while (!before.empty() && (before.back() == '\n' || before.back() == '\r')) {
                    before.pop_back();
                }
                if (!before.empty()) {
                    content_acc_ += before;
                    out.push_back({ParsedEventKind::Content, before, 0, {}});
                }
            }
            buf_.erase(0, call_pos + std::strlen(kCallStartTag));
            state_ = State::ToolCall;
            reset_tool_scan();
            continue;
        }

        if (state_ == State::ToolCall) {
            int end_pos = find_call_end(buf_);
            if (stream_tools_) {
                if (end_pos >= 0) {
                    std::string body_only = buf_.substr(0, static_cast<size_t>(end_pos));
                    scan_tool_stream(body_only, true, out);
                    close_unfinished_tool_scan(body_only, out);
                } else {
                    scan_tool_stream(buf_, false, out);
                }
            }

            if (end_pos < 0) {
                return out;
            }

            std::string body = buf_.substr(0, static_cast<size_t>(end_pos));
            buf_.erase(0, static_cast<size_t>(end_pos) + std::strlen(kCallEndTag));

            ToolCall parsed = parse_tool_call_body(body, schemas_, scan_announced_ ? scan_call_.id : "");
            int idx = scan_announced_ ? scan_tool_index_ : next_tool_index_++;

            if (stream_tools_ && !scan_announced_) {
                ParsedOutputEvent start_ev;
                start_ev.kind = ParsedEventKind::ToolCallStart;
                start_ev.tool_index = idx;
                start_ev.call = parsed;
                out.push_back(std::move(start_ev));

                ParsedOutputEvent args_ev;
                args_ev.kind = ParsedEventKind::ToolCallArgs;
                args_ev.tool_index = idx;
                args_ev.text = parsed.function.arguments_json;
                args_ev.call = parsed;
                out.push_back(std::move(args_ev));
            }

            completed_calls_.push_back(parsed);
            ParsedOutputEvent done_ev;
            done_ev.kind = ParsedEventKind::ToolCallComplete;
            done_ev.tool_index = idx;
            done_ev.call = std::move(parsed);
            out.push_back(std::move(done_ev));

            reset_tool_scan();
            state_ = State::Content;
            strip_leading_newlines_ = true;
            continue;
        }
    }
}

std::vector<ParsedOutputEvent> OpenAiOutputParser::finish() {
    std::vector<ParsedOutputEvent> out;
    if (state_ == State::ToolCall && stream_tools_ && scan_announced_) {
        scan_tool_stream(buf_, true, out);
        if (scan_state_ == ScanState::Done) {
            ToolCall parsed = parse_tool_call_body(buf_, schemas_, scan_call_.id);
            completed_calls_.push_back(parsed);
            ParsedOutputEvent done_ev;
            done_ev.kind = ParsedEventKind::ToolCallComplete;
            done_ev.tool_index = scan_tool_index_;
            done_ev.call = std::move(parsed);
            out.push_back(std::move(done_ev));
        }
        buf_.clear();
        reset_tool_scan();
        return out;
    }

    if (!buf_.empty()) {
        if (state_ == State::Reasoning) {
            reasoning_acc_ += buf_;
            out.push_back({ParsedEventKind::Reasoning, buf_, 0, {}});
        } else {
            std::string text = (state_ == State::ToolCall) ? (std::string(kCallStartTag) + buf_) : buf_;
            if (strip_leading_newlines_) {
                size_t first_non_nl = text.find_first_not_of("\r\n");
                text = (first_non_nl == std::string::npos) ? "" : text.substr(first_non_nl);
            }
            if (!text.empty()) {
                content_acc_ += text;
                out.push_back({ParsedEventKind::Content, text, 0, {}});
            }
        }
        buf_.clear();
    }

    // Trim trailing whitespace on reasoning_acc_
    while (!reasoning_acc_.empty() && (reasoning_acc_.back() == '\n' || reasoning_acc_.back() == '\r')) {
        reasoning_acc_.pop_back();
    }
    return out;
}

// ============================================================================
// OpenAiParser
// ============================================================================

bool OpenAiParser::parse_chat_request(const std::string& body, ChatCompletionRequest& req, std::string& err_type,
                                      std::string& err_param, std::string& err_code, std::string& err_msg) {
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
        "model",
        "messages",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "reasoning_effort",
        "reasoning",
        "thinking",
        "enable_thinking",
        "reasoning_budget_tokens",
        "chat_template_kwargs",
        "temperature",
        "top_p",
        "top_k",
        "max_tokens",
        "max_completion_tokens",
        "stream",
        "stream_options",
        "stop",
        "seed",
        "response_format",
        "n",
        "presence_penalty",
        "frequency_penalty",
        "logit_bias",
        "logprobs",
        "top_logprobs",
        "user",
        "metadata",
        "store",
        "service_tier",
    };

    std::string unsupported_key;
    if (!val.check_unsupported_keys(allowed_keys, unsupported_key)) {
        err_type = "invalid_request_error";
        err_param = unsupported_key;
        err_code = "unsupported_parameter";
        err_msg = "Unsupported parameter: '" + unsupported_key + "'.";
        return false;
    }

    if (val.contains("model") && !val["model"].is_null()) {
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
    size_t leading_sys_end = 0;
    bool seen_non_sys = false;
    for (size_t i = 0; i < msgs.size(); ++i) {
        const auto& m = msgs[i];
        if (!m.is_object() || !m.contains("role") || !m["role"].is_string()) {
            err_type = "invalid_request_error";
            err_param = "messages";
            err_code = "invalid_type";
            err_msg = "Each item in 'messages' must be an object with a string 'role'";
            return false;
        }

        ChatMessage cm;
        cm.role = m["role"].as_string();
        if (cm.role == "developer") {
            cm.role = "system";
        } else if (cm.role == "function") {
            cm.role = "tool";
        }

        if (cm.role == "system") {
            if (seen_non_sys) {
                // Convert mid-conversation system/developer messages to user turns
                cm.role = "user";
            } else {
                leading_sys_end = i + 1;
            }
        } else {
            seen_non_sys = true;
        }
        (void) leading_sys_end;

        const bool has_content_key = m.contains("content");
        const bool has_tool_calls_key = m.contains("tool_calls") && !m["tool_calls"].is_null();
        const bool has_reasoning_key = (m.contains("reasoning_content") && m["reasoning_content"].is_string()) ||
                                       (m.contains("reasoning") && m["reasoning"].is_string());

        if (!has_content_key && !has_tool_calls_key && !has_reasoning_key && cm.role != "assistant") {
            err_type = "invalid_request_error";
            err_param = "messages";
            err_code = "invalid_type";
            err_msg = "Each item in 'messages' must contain 'content' or 'tool_calls'";
            return false;
        }

        if (has_content_key && !m["content"].is_null()) {
            if (m["content"].is_string()) {
                cm.content = m["content"].as_string();
            } else if (m["content"].is_array()) {
                for (size_t p = 0; p < m["content"].size(); ++p) {
                    const auto& part = m["content"][p];
                    if (part.is_object()) {
                        const std::string ptype =
                            part.contains("type") && part["type"].is_string() ? part["type"].as_string() : "text";
                        if (ptype == "thinking" || ptype == "reasoning_text") {
                            if (part.contains("thinking") && part["thinking"].is_string())
                                cm.reasoning_content += part["thinking"].as_string();
                            else if (part.contains("text") && part["text"].is_string())
                                cm.reasoning_content += part["text"].as_string();
                        } else if (part.contains("text") && part["text"].is_string()) {
                            cm.content += part["text"].as_string();
                        }
                    } else if (part.is_string()) {
                        cm.content += part.as_string();
                    }
                }
            } else if (cm.role == "tool" && m["content"].is_object()) {
                cm.content = m["content"].dump();
            } else {
                err_type = "invalid_request_error";
                err_param = "messages";
                err_code = "invalid_type";
                err_msg = "Message 'content' must be a string, array of content parts, or null";
                return false;
            }
        }

        if (m.contains("reasoning_content") && m["reasoning_content"].is_string()) {
            cm.reasoning_content = m["reasoning_content"].as_string();
        } else if (m.contains("reasoning") && m["reasoning"].is_string()) {
            cm.reasoning_content = m["reasoning"].as_string();
        }

        if (m.contains("tool_call_id") && m["tool_call_id"].is_string()) {
            cm.tool_call_id = m["tool_call_id"].as_string();
        }
        if (m.contains("name") && m["name"].is_string()) {
            cm.name = m["name"].as_string();
        }

        if (has_tool_calls_key) {
            if (!m["tool_calls"].is_array()) {
                err_type = "invalid_request_error";
                err_param = "messages";
                err_code = "invalid_type";
                err_msg = "Message 'tool_calls' must be an array";
                return false;
            }
            for (size_t tc_i = 0; tc_i < m["tool_calls"].size(); ++tc_i) {
                const auto& tc_val = m["tool_calls"][tc_i];
                if (!tc_val.is_object()) {
                    err_type = "invalid_request_error";
                    err_param = "messages";
                    err_code = "invalid_type";
                    err_msg = "Each item in 'tool_calls' must be an object";
                    return false;
                }
                const json::JsonValue& fn_val =
                    (tc_val.contains("function") && tc_val["function"].is_object()) ? tc_val["function"] : tc_val;
                if (!fn_val.contains("name") || !fn_val["name"].is_string() ||
                    trim_str(fn_val["name"].as_string()).empty()) {
                    err_type = "invalid_request_error";
                    err_param = "messages";
                    err_code = "invalid_tool_call";
                    err_msg = "Assistant tool_call is missing a valid function 'name'";
                    return false;
                }

                ToolCall tc;
                tc.id = (tc_val.contains("id") && tc_val["id"].is_string()) ? tc_val["id"].as_string()
                                                                            : OpenAiFormatter::generate_id("call_");
                tc.type = "function";
                tc.function.name = trim_str(fn_val["name"].as_string());

                if (fn_val.contains("arguments") && !fn_val["arguments"].is_null()) {
                    if (fn_val["arguments"].is_string()) {
                        std::string raw_args = trim_str(fn_val["arguments"].as_string());
                        if (raw_args.empty()) {
                            tc.function.arguments_json = "{}";
                        } else {
                            json::JsonValue parsed_args;
                            std::string aerr;
                            if (!json::JsonValue::parse(raw_args, parsed_args, aerr) || !parsed_args.is_object()) {
                                err_type = "invalid_request_error";
                                err_param = "messages";
                                err_code = "invalid_tool_call";
                                err_msg = "Tool call 'arguments' for function '" + tc.function.name +
                                          "' must be a valid JSON object string";
                                return false;
                            }
                            tc.function.arguments_json = parsed_args.dump();
                        }
                    } else if (fn_val["arguments"].is_object()) {
                        tc.function.arguments_json = fn_val["arguments"].dump();
                    } else {
                        err_type = "invalid_request_error";
                        err_param = "messages";
                        err_code = "invalid_tool_call";
                        err_msg = "Tool call 'arguments' must be a JSON string or object";
                        return false;
                    }
                } else {
                    tc.function.arguments_json = "{}";
                }
                cm.tool_calls.push_back(std::move(tc));
            }
        }

        req.messages.push_back(std::move(cm));
    }

    if (val.contains("tools")) {
        if (!parse_tools_array(val["tools"], req.tools, err_type, err_param, err_code, err_msg)) {
            return false;
        }
    }

    if (val.contains("tool_choice")) {
        if (!parse_tool_choice_field(val["tool_choice"], req.tools, req.tool_choice, err_type, err_param, err_code,
                                     err_msg)) {
            return false;
        }
    }

    if (val.contains("parallel_tool_calls") && !val["parallel_tool_calls"].is_null()) {
        if (!val["parallel_tool_calls"].is_bool()) {
            err_type = "invalid_request_error";
            err_param = "parallel_tool_calls";
            err_code = "invalid_type";
            err_msg = "Field 'parallel_tool_calls' must be a boolean";
            return false;
        }
        req.parallel_tool_calls = val["parallel_tool_calls"].as_bool();
    }

    // Reasoning / thinking controls
    if (val.contains("reasoning_effort") && !val["reasoning_effort"].is_null()) {
        if (val["reasoning_effort"].is_bool() && !val["reasoning_effort"].as_bool()) {
            req.enable_thinking = false;
            req.reasoning_effort = "none";
        } else if (val["reasoning_effort"].is_string()) {
            if (!apply_reasoning_effort_string(val["reasoning_effort"].as_string(), req.enable_thinking,
                                               req.reasoning_effort, err_msg)) {
                err_type = "invalid_request_error";
                err_param = "reasoning_effort";
                err_code = "invalid_reasoning_effort";
                return false;
            }
        } else {
            err_type = "invalid_request_error";
            err_param = "reasoning_effort";
            err_code = "invalid_type";
            err_msg = "Field 'reasoning_effort' must be a string";
            return false;
        }
    }

    if (val.contains("reasoning") && !val["reasoning"].is_null()) {
        if (!val["reasoning"].is_object()) {
            err_type = "invalid_request_error";
            err_param = "reasoning";
            err_code = "invalid_type";
            err_msg = "Field 'reasoning' must be an object";
            return false;
        }
        const auto& r_obj = val["reasoning"];
        if (r_obj.contains("effort") && r_obj["effort"].is_string()) {
            if (!apply_reasoning_effort_string(r_obj["effort"].as_string(), req.enable_thinking, req.reasoning_effort,
                                               err_msg)) {
                err_type = "invalid_request_error";
                err_param = "reasoning.effort";
                err_code = "invalid_reasoning_effort";
                return false;
            }
        }
    }

    if (val.contains("thinking") && !val["thinking"].is_null()) {
        if (val["thinking"].is_bool()) {
            req.enable_thinking = val["thinking"].as_bool();
            if (!req.enable_thinking)
                req.reasoning_effort = "none";
        } else if (val["thinking"].is_object()) {
            const auto& th = val["thinking"];
            if (th.contains("type") && th["type"].is_string()) {
                const std::string ttype = to_lower_ascii(th["type"].as_string());
                if (ttype == "disabled" || ttype == "none") {
                    req.enable_thinking = false;
                    req.reasoning_effort = "none";
                } else if (ttype == "enabled") {
                    req.enable_thinking = true;
                }
            }
            if (th.contains("budget_tokens") && th["budget_tokens"].is_number()) {
                int btok = static_cast<int>(th["budget_tokens"].as_int());
                req.reasoning_budget_tokens = btok;
                if (btok <= 0) {
                    req.enable_thinking = false;
                    req.reasoning_effort = "none";
                } else if (btok < 2048) {
                    req.reasoning_effort = "low";
                } else if (btok < 8192) {
                    req.reasoning_effort = "medium";
                } else {
                    req.reasoning_effort = "xhigh";
                }
            }
        }
    }

    if (val.contains("enable_thinking") && !val["enable_thinking"].is_null()) {
        if (!val["enable_thinking"].is_bool()) {
            err_type = "invalid_request_error";
            err_param = "enable_thinking";
            err_code = "invalid_type";
            err_msg = "Field 'enable_thinking' must be a boolean";
            return false;
        }
        req.enable_thinking = val["enable_thinking"].as_bool();
        if (!req.enable_thinking)
            req.reasoning_effort = "none";
    }

    if (val.contains("chat_template_kwargs") && val["chat_template_kwargs"].is_object()) {
        const auto& ctk = val["chat_template_kwargs"];
        if (ctk.contains("enable_thinking") && ctk["enable_thinking"].is_bool()) {
            req.enable_thinking = ctk["enable_thinking"].as_bool();
            if (!req.enable_thinking)
                req.reasoning_effort = "none";
        }
        if (ctk.contains("reasoning_effort") && ctk["reasoning_effort"].is_string() && req.enable_thinking) {
            if (!apply_reasoning_effort_string(ctk["reasoning_effort"].as_string(), req.enable_thinking,
                                               req.reasoning_effort, err_msg)) {
                err_type = "invalid_request_error";
                err_param = "chat_template_kwargs.reasoning_effort";
                err_code = "invalid_reasoning_effort";
                return false;
            }
        }
    }

    if (val.contains("reasoning_budget_tokens") && val["reasoning_budget_tokens"].is_number()) {
        req.reasoning_budget_tokens = static_cast<int>(val["reasoning_budget_tokens"].as_int());
    }

    if (val.contains("temperature") && !val["temperature"].is_null()) {
        if (!val["temperature"].is_number()) {
            err_type = "invalid_request_error";
            err_param = "temperature";
            err_code = "invalid_type";
            err_msg = "Field 'temperature' must be a number";
            return false;
        }
        req.temperature = static_cast<float>(val["temperature"].as_double());
    }

    if (val.contains("top_p") && !val["top_p"].is_null()) {
        if (!val["top_p"].is_number()) {
            err_type = "invalid_request_error";
            err_param = "top_p";
            err_code = "invalid_type";
            err_msg = "Field 'top_p' must be a number";
            return false;
        }
        req.top_p = static_cast<float>(val["top_p"].as_double());
    }

    if (val.contains("top_k") && !val["top_k"].is_null()) {
        if (!val["top_k"].is_number()) {
            err_type = "invalid_request_error";
            err_param = "top_k";
            err_code = "invalid_type";
            err_msg = "Field 'top_k' must be an integer";
            return false;
        }
        req.top_k = static_cast<int>(val["top_k"].as_int());
    }

    if (val.contains("max_completion_tokens") && !val["max_completion_tokens"].is_null()) {
        if (!val["max_completion_tokens"].is_number()) {
            err_type = "invalid_request_error";
            err_param = "max_completion_tokens";
            err_code = "invalid_type";
            err_msg = "Field 'max_completion_tokens' must be an integer";
            return false;
        }
        req.max_tokens = static_cast<int>(val["max_completion_tokens"].as_int());
    } else if (val.contains("max_tokens") && !val["max_tokens"].is_null()) {
        if (!val["max_tokens"].is_number()) {
            err_type = "invalid_request_error";
            err_param = "max_tokens";
            err_code = "invalid_type";
            err_msg = "Field 'max_tokens' must be an integer";
            return false;
        }
        req.max_tokens = static_cast<int>(val["max_tokens"].as_int());
    }

    if (val.contains("stream") && !val["stream"].is_null()) {
        if (!val["stream"].is_bool()) {
            err_type = "invalid_request_error";
            err_param = "stream";
            err_code = "invalid_type";
            err_msg = "Field 'stream' must be a boolean";
            return false;
        }
        req.stream = val["stream"].as_bool();
    }

    if (val.contains("stream_options") && val["stream_options"].is_object()) {
        const auto& so = val["stream_options"];
        if (so.contains("include_usage") && so["include_usage"].is_bool()) {
            req.stream_include_usage = so["include_usage"].as_bool();
        }
    }

    if (val.contains("stop") && !val["stop"].is_null()) {
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

    if (val.contains("response_format") && val["response_format"].is_object()) {
        const auto& rf = val["response_format"];
        if (rf.contains("type") && rf["type"].is_string()) {
            req.response_format_type = rf["type"].as_string();
        }
    }

    return true;
}

bool OpenAiParser::parse_responses_request(const std::string& body, ResponsesRequest& req, std::string& err_type,
                                           std::string& err_param, std::string& err_code, std::string& err_msg) {
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

    if (val.contains("model") && val["model"].is_string()) {
        req.chat.model = val["model"].as_string();
    }

    if (val.contains("instructions") && val["instructions"].is_string()) {
        req.instructions = val["instructions"].as_string();
        if (!req.instructions.empty()) {
            req.chat.messages.push_back({"system", req.instructions});
        }
    }

    if (!val.contains("input") || val["input"].is_null()) {
        err_type = "invalid_request_error";
        err_param = "input";
        err_code = "missing_required_parameter";
        err_msg = "Field 'input' is required";
        return false;
    }

    if (val["input"].is_string()) {
        req.chat.messages.push_back({"user", val["input"].as_string()});
    } else if (val["input"].is_array()) {
        std::string pending_reasoning;
        for (size_t i = 0; i < val["input"].size(); ++i) {
            const auto& item = val["input"][i];
            if (!item.is_object()) {
                err_type = "invalid_request_error";
                err_param = "input";
                err_code = "invalid_type";
                err_msg = "Each item in 'input' must be an object";
                return false;
            }
            const std::string kind = item.contains("type") && item["type"].is_string()
                                         ? item["type"].as_string()
                                         : (item.contains("role") && item["role"].is_string() ? "message" : "");

            if (kind == "message") {
                ChatMessage cm;
                cm.role = item.contains("role") && item["role"].is_string() ? item["role"].as_string() : "user";
                if (cm.role == "developer")
                    cm.role = "system";
                if (item.contains("content")) {
                    if (item["content"].is_string()) {
                        cm.content = item["content"].as_string();
                    } else if (item["content"].is_array()) {
                        for (size_t p = 0; p < item["content"].size(); ++p) {
                            const auto& part = item["content"][p];
                            if (part.is_object() && part.contains("text") && part["text"].is_string()) {
                                cm.content += part["text"].as_string();
                            }
                        }
                    }
                }
                if (cm.role == "assistant" && !pending_reasoning.empty()) {
                    cm.reasoning_content = pending_reasoning;
                    pending_reasoning.clear();
                }
                req.chat.messages.push_back(std::move(cm));
            } else if (kind == "reasoning") {
                if (item.contains("content") && item["content"].is_array()) {
                    for (size_t p = 0; p < item["content"].size(); ++p) {
                        const auto& part = item["content"][p];
                        if (part.is_object() && part.contains("text") && part["text"].is_string()) {
                            pending_reasoning += part["text"].as_string();
                        }
                    }
                }
            } else if (kind == "function_call" || kind == "custom_tool_call") {
                ToolCall tc;
                tc.id = item.contains("call_id") && item["call_id"].is_string() ? item["call_id"].as_string()
                                                                                : OpenAiFormatter::generate_id("call_");
                tc.type = "function";
                tc.function.name = item.contains("name") && item["name"].is_string() ? item["name"].as_string() : "";
                if (item.contains("arguments") && item["arguments"].is_string()) {
                    tc.function.arguments_json = item["arguments"].as_string();
                } else if (item.contains("arguments") && item["arguments"].is_object()) {
                    tc.function.arguments_json = item["arguments"].dump();
                } else {
                    tc.function.arguments_json = "{}";
                }
                if (!req.chat.messages.empty() && req.chat.messages.back().role == "assistant") {
                    req.chat.messages.back().tool_calls.push_back(std::move(tc));
                } else {
                    ChatMessage cm;
                    cm.role = "assistant";
                    cm.reasoning_content = pending_reasoning;
                    pending_reasoning.clear();
                    cm.tool_calls.push_back(std::move(tc));
                    req.chat.messages.push_back(std::move(cm));
                }
            } else if (kind == "function_call_output" || kind == "custom_tool_call_output") {
                ChatMessage cm;
                cm.role = "tool";
                if (item.contains("call_id") && item["call_id"].is_string()) {
                    cm.tool_call_id = item["call_id"].as_string();
                }
                if (item.contains("output")) {
                    if (item["output"].is_string())
                        cm.content = item["output"].as_string();
                    else
                        cm.content = item["output"].dump();
                }
                req.chat.messages.push_back(std::move(cm));
            }
        }
    } else {
        err_type = "invalid_request_error";
        err_param = "input";
        err_code = "invalid_type";
        err_msg = "Field 'input' must be a string or array of input items";
        return false;
    }

    if (val.contains("tools")) {
        if (!parse_tools_array(val["tools"], req.chat.tools, err_type, err_param, err_code, err_msg)) {
            return false;
        }
    }
    if (val.contains("tool_choice")) {
        if (!parse_tool_choice_field(val["tool_choice"], req.chat.tools, req.chat.tool_choice, err_type, err_param,
                                     err_code, err_msg)) {
            return false;
        }
    }
    if (val.contains("reasoning") && val["reasoning"].is_object()) {
        const auto& r = val["reasoning"];
        if (r.contains("effort") && r["effort"].is_string()) {
            if (!apply_reasoning_effort_string(r["effort"].as_string(), req.chat.enable_thinking,
                                               req.chat.reasoning_effort, err_msg)) {
                err_type = "invalid_request_error";
                err_param = "reasoning.effort";
                err_code = "invalid_reasoning_effort";
                return false;
            }
        }
    }
    if (val.contains("max_output_tokens") && val["max_output_tokens"].is_number()) {
        req.chat.max_tokens = static_cast<int>(val["max_output_tokens"].as_int());
    } else if (val.contains("max_tokens") && val["max_tokens"].is_number()) {
        req.chat.max_tokens = static_cast<int>(val["max_tokens"].as_int());
    }
    if (val.contains("temperature") && val["temperature"].is_number()) {
        req.chat.temperature = static_cast<float>(val["temperature"].as_double());
    }
    if (val.contains("top_p") && val["top_p"].is_number()) {
        req.chat.top_p = static_cast<float>(val["top_p"].as_double());
    }
    if (val.contains("stream") && val["stream"].is_bool()) {
        req.chat.stream = val["stream"].as_bool();
    }

    return true;
}

bool OpenAiParser::parse_completion_request(const std::string& body, CompletionRequest& req, std::string& err_type,
                                            std::string& err_param, std::string& err_code, std::string& err_msg) {
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

    static const std::vector<std::string> allowed_keys = {"model",      "prompt", "temperature", "top_p",
                                                          "max_tokens", "stream", "stop",        "seed"};

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

// ============================================================================
// OpenAiFormatter
// ============================================================================

std::string OpenAiFormatter::format_error(const std::string& message, const std::string& type, const std::string& param,
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

std::string OpenAiFormatter::format_chat_completion(const std::string& id, const std::string& model, int64_t created,
                                                    const std::string& content, const std::string& finish_reason,
                                                    int prompt_tokens, int completion_tokens,
                                                    const std::string& reasoning_content,
                                                    const std::vector<ToolCall>& tool_calls, int reasoning_tokens,
                                                    int cached_tokens) {
    const std::string effective_finish =
        (!tool_calls.empty() && finish_reason == "stop") ? "tool_calls" : finish_reason;

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
       << "        \"role\": \"assistant\",\n";

    if (content.empty() && !tool_calls.empty()) {
        ss << "        \"content\": null";
    } else {
        ss << "        \"content\": \"" << json::JsonValue::escape_string(content) << "\"";
    }

    if (!reasoning_content.empty()) {
        ss << ",\n        \"reasoning_content\": \"" << json::JsonValue::escape_string(reasoning_content) << "\"";
    }

    if (!tool_calls.empty()) {
        ss << ",\n        \"tool_calls\": [\n";
        for (size_t i = 0; i < tool_calls.size(); ++i) {
            const auto& tc = tool_calls[i];
            if (i > 0)
                ss << ",\n";
            ss << "          {\n"
               << "            \"id\": \"" << json::JsonValue::escape_string(tc.id) << "\",\n"
               << "            \"type\": \"" << json::JsonValue::escape_string(tc.type) << "\",\n"
               << "            \"function\": {\n"
               << "              \"name\": \"" << json::JsonValue::escape_string(tc.function.name) << "\",\n"
               << "              \"arguments\": \"" << json::JsonValue::escape_string(tc.function.arguments_json)
               << "\"\n"
               << "            }\n"
               << "          }";
        }
        ss << "\n        ]";
    }

    ss << "\n      },\n"
       << "      \"finish_reason\": \"" << json::JsonValue::escape_string(effective_finish) << "\"\n"
       << "    }\n"
       << "  ],\n"
       << "  \"usage\": {\n"
       << "    \"prompt_tokens\": " << prompt_tokens << ",\n"
       << "    \"completion_tokens\": " << completion_tokens << ",\n"
       << "    \"total_tokens\": " << (prompt_tokens + completion_tokens) << ",\n"
       << "    \"prompt_tokens_details\": {\n"
       << "      \"cached_tokens\": " << cached_tokens << "\n"
       << "    },\n"
       << "    \"completion_tokens_details\": {\n"
       << "      \"reasoning_tokens\": " << reasoning_tokens << "\n"
       << "    }\n"
       << "  }\n"
       << "}";
    return ss.str();
}

std::string OpenAiFormatter::format_chat_chunk(const std::string& id, const std::string& model, int64_t created,
                                               const std::string& delta_role, const std::string& delta_content,
                                               const std::string& finish_reason, const std::string& delta_reasoning) {
    std::ostringstream ss;
    ss << "{\"id\":\"" << json::JsonValue::escape_string(id)
       << "\",\"object\":\"chat.completion.chunk\",\"created\":" << created << ",\"model\":\""
       << json::JsonValue::escape_string(model) << "\",\"choices\":[{\"index\":0,\"delta\":{";

    bool has_field = false;
    if (!delta_role.empty()) {
        ss << "\"role\":\"" << json::JsonValue::escape_string(delta_role) << "\"";
        has_field = true;
    }
    if (!delta_content.empty()) {
        if (has_field)
            ss << ",";
        ss << "\"content\":\"" << json::JsonValue::escape_string(delta_content) << "\"";
        has_field = true;
    }
    if (!delta_reasoning.empty()) {
        if (has_field)
            ss << ",";
        ss << "\"reasoning_content\":\"" << json::JsonValue::escape_string(delta_reasoning) << "\"";
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

std::string OpenAiFormatter::format_chat_tool_call_chunk(const std::string& id, const std::string& model,
                                                         int64_t created, int tool_index, const std::string& call_id,
                                                         const std::string& function_name,
                                                         const std::string& arguments_delta) {
    std::ostringstream ss;
    ss << "{\"id\":\"" << json::JsonValue::escape_string(id)
       << "\",\"object\":\"chat.completion.chunk\",\"created\":" << created << ",\"model\":\""
       << json::JsonValue::escape_string(model)
       << "\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":" << tool_index;
    if (!call_id.empty()) {
        ss << ",\"id\":\"" << json::JsonValue::escape_string(call_id) << "\",\"type\":\"function\"";
    }
    ss << ",\"function\":{";
    bool has_fn = false;
    if (!function_name.empty()) {
        ss << "\"name\":\"" << json::JsonValue::escape_string(function_name) << "\"";
        has_fn = true;
    }
    if (!arguments_delta.empty() || !function_name.empty()) {
        if (has_fn)
            ss << ",";
        ss << "\"arguments\":\"" << json::JsonValue::escape_string(arguments_delta) << "\"";
    }
    ss << "}}]},\"finish_reason\":null}]}";
    return ss.str();
}

std::string OpenAiFormatter::format_chat_usage_chunk(const std::string& id, const std::string& model, int64_t created,
                                                     const std::string& finish_reason, int prompt_tokens,
                                                     int completion_tokens, int reasoning_tokens, int cached_tokens) {
    std::ostringstream ss;
    ss << "{\"id\":\"" << json::JsonValue::escape_string(id)
       << "\",\"object\":\"chat.completion.chunk\",\"created\":" << created << ",\"model\":\""
       << json::JsonValue::escape_string(model) << "\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":";
    if (finish_reason.empty()) {
        ss << "null";
    } else {
        ss << "\"" << json::JsonValue::escape_string(finish_reason) << "\"";
    }
    ss << "}],\"usage\":{\"prompt_tokens\":" << prompt_tokens << ",\"completion_tokens\":" << completion_tokens
       << ",\"total_tokens\":" << (prompt_tokens + completion_tokens)
       << ",\"prompt_tokens_details\":{\"cached_tokens\":" << cached_tokens
       << "},\"completion_tokens_details\":{\"reasoning_tokens\":" << reasoning_tokens << "}}}";
    return ss.str();
}

std::string OpenAiFormatter::format_responses_object(const std::string& id, const std::string& model, int64_t created,
                                                     const std::string& content, const std::string& reasoning_content,
                                                     const std::vector<ToolCall>& tool_calls,
                                                     const std::string& finish_reason, int prompt_tokens,
                                                     int completion_tokens, int reasoning_tokens, int cached_tokens) {
    json::JsonValue root(std::vector<std::pair<std::string, json::JsonValue>>{});
    root["id"] = id;
    root["object"] = "response";
    root["created_at"] = created;
    root["status"] = (finish_reason == "length") ? "incomplete" : "completed";
    root["model"] = model;

    std::vector<json::JsonValue> output_items;
    if (!reasoning_content.empty()) {
        json::JsonValue rs_item(std::vector<std::pair<std::string, json::JsonValue>>{});
        rs_item["id"] = generate_id("rs_");
        rs_item["type"] = "reasoning";
        rs_item["status"] = "completed";
        json::JsonValue rpart(std::vector<std::pair<std::string, json::JsonValue>>{});
        rpart["type"] = "reasoning_text";
        rpart["text"] = reasoning_content;
        rs_item["content"] = std::vector<json::JsonValue>{std::move(rpart)};
        rs_item["summary"] = std::vector<json::JsonValue>{};
        output_items.push_back(std::move(rs_item));
    }

    if (!content.empty() || tool_calls.empty()) {
        json::JsonValue msg_item(std::vector<std::pair<std::string, json::JsonValue>>{});
        msg_item["id"] = generate_id("msg_");
        msg_item["type"] = "message";
        msg_item["status"] = "completed";
        msg_item["role"] = "assistant";
        json::JsonValue tpart(std::vector<std::pair<std::string, json::JsonValue>>{});
        tpart["type"] = "output_text";
        tpart["text"] = content;
        tpart["annotations"] = std::vector<json::JsonValue>{};
        msg_item["content"] = std::vector<json::JsonValue>{std::move(tpart)};
        output_items.push_back(std::move(msg_item));
    }

    for (const auto& tc : tool_calls) {
        json::JsonValue fc_item(std::vector<std::pair<std::string, json::JsonValue>>{});
        fc_item["id"] = generate_id("fc_");
        fc_item["type"] = "function_call";
        fc_item["status"] = "completed";
        fc_item["call_id"] = tc.id;
        fc_item["name"] = tc.function.name;
        fc_item["arguments"] = tc.function.arguments_json;
        output_items.push_back(std::move(fc_item));
    }

    root["output"] = std::move(output_items);
    root["output_text"] = content;

    json::JsonValue usage(std::vector<std::pair<std::string, json::JsonValue>>{});
    usage["input_tokens"] = static_cast<int64_t>(prompt_tokens);
    json::JsonValue in_details(std::vector<std::pair<std::string, json::JsonValue>>{});
    in_details["cached_tokens"] = static_cast<int64_t>(cached_tokens);
    usage["input_tokens_details"] = std::move(in_details);
    usage["output_tokens"] = static_cast<int64_t>(completion_tokens);
    usage["total_tokens"] = static_cast<int64_t>(prompt_tokens + completion_tokens);
    json::JsonValue out_details(std::vector<std::pair<std::string, json::JsonValue>>{});
    out_details["reasoning_tokens"] = static_cast<int64_t>(reasoning_tokens);
    usage["output_tokens_details"] = std::move(out_details);
    root["usage"] = std::move(usage);

    return root.dump_pretty();
}

std::string OpenAiFormatter::format_completion(const std::string& id, const std::string& model, int64_t created,
                                               const std::string& text, const std::string& finish_reason,
                                               int prompt_tokens, int completion_tokens) {
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

std::string OpenAiFormatter::format_completion_chunk(const std::string& id, const std::string& model, int64_t created,
                                                     const std::string& text, const std::string& finish_reason) {
    std::ostringstream ss;
    ss << "{\"id\":\"" << json::JsonValue::escape_string(id)
       << "\",\"object\":\"text_completion\",\"created\":" << created << ",\"model\":\""
       << json::JsonValue::escape_string(model) << "\",\"choices\":[{\"text\":\""
       << json::JsonValue::escape_string(text) << "\",\"index\":0,\"logprobs\":null,\"finish_reason\":";
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
        if (i > 0)
            ss << ",\n";
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

std::string OpenAiFormatter::format_model(const std::string& model_name) {
    std::ostringstream ss;
    ss << "{\n"
       << "  \"id\": \"" << json::JsonValue::escape_string(model_name) << "\",\n"
       << "  \"object\": \"model\",\n"
       << "  \"created\": 1728100000,\n"
       << "  \"owned_by\": \"guild\"\n"
       << "}";
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

std::string OpenAiFormatter::render_chatml(const std::vector<ChatMessage>& messages,
                                           const runtime::ChatTemplateOptions& options) {
    bool has_rich_turns = !options.tools.empty() || !options.enable_thinking || options.reasoning_effort != "xhigh";
    if (!has_rich_turns) {
        for (const auto& m : messages) {
            if (!m.tool_calls.empty() || !m.reasoning_content.empty() || m.role == "tool") {
                has_rich_turns = true;
                break;
            }
        }
    }
    if (has_rich_turns) {
        runtime::ChatTemplate synthetic;
        std::string load_err, rendered, render_err;
        if (synthetic.load_from_string("resolved_reasoning_effort <|im_start|> <|im_end|> render_content", load_err) &&
            synthetic.render(messages, options, rendered, render_err)) {
            return rendered;
        }
    }

    std::string out;
    for (size_t i = 0; i < messages.size(); ++i) {
        const auto& msg = messages[i];
        if (msg.role == "tool") {
            if (i == 0 || messages[i - 1].role != "tool") {
                out += "<|im_start|>user";
            }
            out += "\n<tool_response>\n" + msg.content + "\n</tool_response>";
            if (i + 1 == messages.size() || messages[i + 1].role != "tool") {
                out += "<|im_end|>\n";
            }
        } else if (msg.role == "assistant") {
            out += "<|im_start|>assistant\n";
            if (!msg.reasoning_content.empty()) {
                out += "<think>\n" + msg.reasoning_content + "\n</think>\n\n";
            }
            out += msg.content;
            for (size_t k = 0; k < msg.tool_calls.size(); ++k) {
                const auto& tc = msg.tool_calls[k];
                out += (k == 0 && !msg.content.empty()) ? "\n\n<tool_call>\n<function=" : "\n<tool_call>\n<function=";
                out += tc.function.name + ">\n";
                json::JsonValue args_val;
                std::string perr;
                if (json::JsonValue::parse(tc.function.arguments_json, args_val, perr) && args_val.is_object()) {
                    for (const auto& kv : args_val.obj_val) {
                        out += "<parameter=" + kv.first + ">\n";
                        out += kv.second.is_string() ? kv.second.as_string() : kv.second.dump();
                        out += "\n</parameter>\n";
                    }
                }
                out += "</function>\n</tool_call>";
            }
            out += "<|im_end|>\n";
        } else {
            out += "<|im_start|>" + msg.role + "\n" + msg.content + "<|im_end|>\n";
        }
    }
    out += "<|im_start|>assistant\n";
    return out;
}

} // namespace guild::server
