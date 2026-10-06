#pragma once

#include <cctype>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace guild::server::json {

enum class JsonType {
    Null,
    Bool,
    Number,
    String,
    Array,
    Object
};

class JsonValue {
public:
    JsonType type = JsonType::Null;
    bool bool_val = false;
    double num_val = 0.0;
    std::string str_val;
    std::vector<JsonValue> arr_val;
    std::vector<std::pair<std::string, JsonValue>> obj_val;

    JsonValue() : type(JsonType::Null) {}
    JsonValue(std::nullptr_t) : type(JsonType::Null) {}
    JsonValue(bool b) : type(JsonType::Bool), bool_val(b) {}
    JsonValue(int n) : type(JsonType::Number), num_val(n) {}
    JsonValue(int64_t n) : type(JsonType::Number), num_val(static_cast<double>(n)) {}
    JsonValue(double d) : type(JsonType::Number), num_val(d) {}
    JsonValue(const char* s) : type(JsonType::String), str_val(s ? s : "") {}
    JsonValue(std::string s) : type(JsonType::String), str_val(std::move(s)) {}
    JsonValue(std::vector<JsonValue> arr) : type(JsonType::Array), arr_val(std::move(arr)) {}
    JsonValue(std::vector<std::pair<std::string, JsonValue>> obj) : type(JsonType::Object), obj_val(std::move(obj)) {}

    bool is_null() const { return type == JsonType::Null; }
    bool is_bool() const { return type == JsonType::Bool; }
    bool is_number() const { return type == JsonType::Number; }
    bool is_string() const { return type == JsonType::String; }
    bool is_array() const { return type == JsonType::Array; }
    bool is_object() const { return type == JsonType::Object; }

    bool as_bool(bool def = false) const {
        return (type == JsonType::Bool) ? bool_val : def;
    }

    int64_t as_int(int64_t def = 0) const {
        return (type == JsonType::Number) ? static_cast<int64_t>(num_val) : def;
    }

    double as_double(double def = 0.0) const {
        return (type == JsonType::Number) ? num_val : def;
    }

    const std::string& as_string() const {
        return str_val;
    }

    std::string as_string(const std::string& def) const {
        return (type == JsonType::String) ? str_val : def;
    }

    // Object helpers
    bool contains(const std::string& key) const {
        if (type != JsonType::Object) return false;
        for (const auto& kv : obj_val) {
            if (kv.first == key) return true;
        }
        return false;
    }

    const JsonValue& operator[](const std::string& key) const {
        static const JsonValue null_instance;
        if (type != JsonType::Object) return null_instance;
        for (const auto& kv : obj_val) {
            if (kv.first == key) return kv.second;
        }
        return null_instance;
    }

    JsonValue& operator[](const std::string& key) {
        if (type != JsonType::Object) {
            type = JsonType::Object;
            obj_val.clear();
        }
        for (auto& kv : obj_val) {
            if (kv.first == key) return kv.second;
        }
        obj_val.push_back({key, JsonValue()});
        return obj_val.back().second;
    }

    // Array helpers
    size_t size() const {
        if (type == JsonType::Array) return arr_val.size();
        if (type == JsonType::Object) return obj_val.size();
        return 0;
    }

    const JsonValue& operator[](size_t index) const {
        static const JsonValue null_instance;
        if (type != JsonType::Array || index >= arr_val.size()) return null_instance;
        return arr_val[index];
    }

    JsonValue& operator[](size_t index) {
        if (type != JsonType::Array) {
            type = JsonType::Array;
            arr_val.clear();
        }
        if (index >= arr_val.size()) arr_val.resize(index + 1);
        return arr_val[index];
    }

    void push_back(JsonValue val) {
        if (type != JsonType::Array) {
            type = JsonType::Array;
            arr_val.clear();
        }
        arr_val.push_back(std::move(val));
    }

    // Check if an object contains keys outside the allowed list
    bool check_unsupported_keys(const std::vector<std::string>& allowed, std::string& out_unsupported) const {
        if (type != JsonType::Object) return true;
        for (const auto& kv : obj_val) {
            bool found = false;
            for (const auto& a : allowed) {
                if (kv.first == a) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                out_unsupported = kv.first;
                return false;
            }
        }
        return true;
    }

    // Serialization
    static std::string escape_string(const std::string& s) {
        std::string out;
        out.reserve(s.size() + 16);
        for (unsigned char c : s) {
            switch (c) {
                case '"': out += "\\\""; break;
                case '\\': out += "\\\\"; break;
                case '\b': out += "\\b"; break;
                case '\f': out += "\\f"; break;
                case '\n': out += "\\n"; break;
                case '\r': out += "\\r"; break;
                case '\t': out += "\\t"; break;
                default:
                    if (c < 0x20) {
                        char buf[8];
                        std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                        out += buf;
                    } else {
                        out += static_cast<char>(c);
                    }
                    break;
            }
        }
        return out;
    }

    std::string dump() const {
        switch (type) {
            case JsonType::Null: return "null";
            case JsonType::Bool: return bool_val ? "true" : "false";
            case JsonType::Number: {
                if (std::floor(num_val) == num_val && !std::isinf(num_val) && !std::isnan(num_val)) {
                    return std::to_string(static_cast<int64_t>(num_val));
                }
                std::ostringstream ss;
                ss << num_val;
                return ss.str();
            }
            case JsonType::String:
                return "\"" + escape_string(str_val) + "\"";
            case JsonType::Array: {
                std::string res = "[";
                for (size_t i = 0; i < arr_val.size(); ++i) {
                    if (i > 0) res += ",";
                    res += arr_val[i].dump();
                }
                res += "]";
                return res;
            }
            case JsonType::Object: {
                std::string res = "{";
                for (size_t i = 0; i < obj_val.size(); ++i) {
                    if (i > 0) res += ",";
                    res += "\"" + escape_string(obj_val[i].first) + "\":" + obj_val[i].second.dump();
                }
                res += "}";
                return res;
            }
        }
        return "null";
    }

    std::string dump_pretty(int indent_step = 2, int current_indent = 0) const {
        std::string pad(current_indent, ' ');
        switch (type) {
            case JsonType::Null: return "null";
            case JsonType::Bool: return bool_val ? "true" : "false";
            case JsonType::Number: {
                if (std::floor(num_val) == num_val && !std::isinf(num_val) && !std::isnan(num_val)) {
                    return std::to_string(static_cast<int64_t>(num_val));
                }
                std::ostringstream ss;
                ss << num_val;
                return ss.str();
            }
            case JsonType::String:
                return "\"" + escape_string(str_val) + "\"";
            case JsonType::Array: {
                if (arr_val.empty()) return "[]";
                std::string res = "[\n";
                std::string inner_pad(current_indent + indent_step, ' ');
                for (size_t i = 0; i < arr_val.size(); ++i) {
                    if (i > 0) res += ",\n";
                    res += inner_pad + arr_val[i].dump_pretty(indent_step, current_indent + indent_step);
                }
                res += "\n" + pad + "]";
                return res;
            }
            case JsonType::Object: {
                if (obj_val.empty()) return "{}";
                std::string res = "{\n";
                std::string inner_pad(current_indent + indent_step, ' ');
                for (size_t i = 0; i < obj_val.size(); ++i) {
                    if (i > 0) res += ",\n";
                    res += inner_pad + "\"" + escape_string(obj_val[i].first) + "\": " +
                           obj_val[i].second.dump_pretty(indent_step, current_indent + indent_step);
                }
                res += "\n" + pad + "}";
                return res;
            }
        }
        return "null";
    }

    std::string serialize_pretty() const {
        return dump_pretty();
    }

    // Parsing
    static bool parse(const std::string& input, JsonValue& out, std::string& err) {
        size_t idx = 0;
        skip_ws(input, idx);
        if (idx >= input.size()) {
            err = "Empty input";
            return false;
        }
        if (!parse_value(input, idx, out, err)) {
            return false;
        }
        skip_ws(input, idx);
        if (idx < input.size()) {
            err = "Trailing characters after JSON value at pos " + std::to_string(idx);
            return false;
        }
        return true;
    }

    static std::optional<JsonValue> parse(const std::string& input, std::string& err) {
        JsonValue val;
        if (parse(input, val, err)) {
            return val;
        }
        return std::nullopt;
    }

private:
    static void skip_ws(const std::string& s, size_t& idx) {
        while (idx < s.size() && (s[idx] == ' ' || s[idx] == '\t' || s[idx] == '\r' || s[idx] == '\n')) {
            ++idx;
        }
    }

    static bool parse_value(const std::string& s, size_t& idx, JsonValue& val, std::string& err) {
        skip_ws(s, idx);
        if (idx >= s.size()) {
            err = "Unexpected end of input";
            return false;
        }

        char c = s[idx];
        if (c == 'n') return parse_null(s, idx, val, err);
        if (c == 't' || c == 'f') return parse_bool(s, idx, val, err);
        if (c == '"') return parse_string(s, idx, val, err);
        if (c == '[') return parse_array(s, idx, val, err);
        if (c == '{') return parse_object(s, idx, val, err);
        if (c == '-' || (c >= '0' && c <= '9')) return parse_number(s, idx, val, err);

        err = std::string("Unexpected character '") + c + "' at position " + std::to_string(idx);
        return false;
    }

    static bool parse_null(const std::string& s, size_t& idx, JsonValue& val, std::string& err) {
        if (s.substr(idx, 4) == "null") {
            idx += 4;
            val = JsonValue(nullptr);
            return true;
        }
        err = "Expected 'null' at position " + std::to_string(idx);
        return false;
    }

    static bool parse_bool(const std::string& s, size_t& idx, JsonValue& val, std::string& err) {
        if (s.substr(idx, 4) == "true") {
            idx += 4;
            val = JsonValue(true);
            return true;
        }
        if (s.substr(idx, 5) == "false") {
            idx += 5;
            val = JsonValue(false);
            return true;
        }
        err = "Expected boolean at position " + std::to_string(idx);
        return false;
    }

    static bool parse_number(const std::string& s, size_t& idx, JsonValue& val, std::string& err) {
        size_t start = idx;
        if (idx < s.size() && s[idx] == '-') ++idx;
        if (idx >= s.size() || !std::isdigit(static_cast<unsigned char>(s[idx]))) {
            err = "Invalid number format at position " + std::to_string(idx);
            return false;
        }
        while (idx < s.size() && std::isdigit(static_cast<unsigned char>(s[idx]))) ++idx;
        if (idx < s.size() && s[idx] == '.') {
            ++idx;
            if (idx >= s.size() || !std::isdigit(static_cast<unsigned char>(s[idx]))) {
                err = "Invalid number fraction at position " + std::to_string(idx);
                return false;
            }
            while (idx < s.size() && std::isdigit(static_cast<unsigned char>(s[idx]))) ++idx;
        }
        if (idx < s.size() && (s[idx] == 'e' || s[idx] == 'E')) {
            ++idx;
            if (idx < s.size() && (s[idx] == '+' || s[idx] == '-')) ++idx;
            if (idx >= s.size() || !std::isdigit(static_cast<unsigned char>(s[idx]))) {
                err = "Invalid exponent at position " + std::to_string(idx);
                return false;
            }
            while (idx < s.size() && std::isdigit(static_cast<unsigned char>(s[idx]))) ++idx;
        }

        std::string num_str = s.substr(start, idx - start);
        try {
            double d = std::stod(num_str);
            val = JsonValue(d);
            return true;
        } catch (...) {
            err = "Failed to parse number '" + num_str + "'";
            return false;
        }
    }

    static bool parse_raw_string(const std::string& s, size_t& idx, std::string& out, std::string& err) {
        if (idx >= s.size() || s[idx] != '"') {
            err = "Expected '\"' at position " + std::to_string(idx);
            return false;
        }
        ++idx;
        out.clear();
        while (idx < s.size()) {
            char c = s[idx++];
            if (c == '"') {
                return true;
            }
            if (c == '\\') {
                if (idx >= s.size()) {
                    err = "Unexpected end of input inside escape sequence";
                    return false;
                }
                char esc = s[idx++];
                switch (esc) {
                    case '"': out += '"'; break;
                    case '\\': out += '\\'; break;
                    case '/': out += '/'; break;
                    case 'b': out += '\b'; break;
                    case 'f': out += '\f'; break;
                    case 'n': out += '\n'; break;
                    case 'r': out += '\r'; break;
                    case 't': out += '\t'; break;
                    case 'u': {
                        if (idx + 4 > s.size()) {
                            err = "Incomplete unicode escape sequence";
                            return false;
                        }
                        std::string hex = s.substr(idx, 4);
                        idx += 4;
                        try {
                            uint32_t codepoint = static_cast<uint32_t>(std::stoul(hex, nullptr, 16));
                            if (codepoint <= 0x7F) {
                                out += static_cast<char>(codepoint);
                            } else if (codepoint <= 0x7FF) {
                                out += static_cast<char>(0xC0 | ((codepoint >> 6) & 0x1F));
                                out += static_cast<char>(0x80 | (codepoint & 0x3F));
                            } else {
                                out += static_cast<char>(0xE0 | ((codepoint >> 12) & 0x0F));
                                out += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F));
                                out += static_cast<char>(0x80 | (codepoint & 0x3F));
                            }
                        } catch (...) {
                            err = "Invalid unicode hex sequence '\\u" + hex + "'";
                            return false;
                        }
                        break;
                    }
                    default:
                        out += esc;
                        break;
                }
            } else {
                out += c;
            }
        }
        err = "Unterminated string";
        return false;
    }

    static bool parse_string(const std::string& s, size_t& idx, JsonValue& val, std::string& err) {
        std::string str;
        if (!parse_raw_string(s, idx, str, err)) return false;
        val = JsonValue(std::move(str));
        return true;
    }

    static bool parse_array(const std::string& s, size_t& idx, JsonValue& val, std::string& err) {
        ++idx; // skip '['
        val = JsonValue(std::vector<JsonValue>{});
        skip_ws(s, idx);
        if (idx < s.size() && s[idx] == ']') {
            ++idx;
            return true;
        }
        while (idx < s.size()) {
            JsonValue elem;
            if (!parse_value(s, idx, elem, err)) return false;
            val.push_back(std::move(elem));
            skip_ws(s, idx);
            if (idx >= s.size()) break;
            if (s[idx] == ',') {
                ++idx;
                skip_ws(s, idx);
            } else if (s[idx] == ']') {
                ++idx;
                return true;
            } else {
                err = std::string("Expected ',' or ']' in array at position ") + std::to_string(idx);
                return false;
            }
        }
        err = "Unterminated array";
        return false;
    }

    static bool parse_object(const std::string& s, size_t& idx, JsonValue& val, std::string& err) {
        ++idx; // skip '{'
        val = JsonValue(std::vector<std::pair<std::string, JsonValue>>{});
        skip_ws(s, idx);
        if (idx < s.size() && s[idx] == '}') {
            ++idx;
            return true;
        }
        while (idx < s.size()) {
            skip_ws(s, idx);
            std::string key;
            if (!parse_raw_string(s, idx, key, err)) return false;
            skip_ws(s, idx);
            if (idx >= s.size() || s[idx] != ':') {
                err = "Expected ':' after key at position " + std::to_string(idx);
                return false;
            }
            ++idx;
            JsonValue sub_val;
            if (!parse_value(s, idx, sub_val, err)) return false;
            val.obj_val.push_back({std::move(key), std::move(sub_val)});
            skip_ws(s, idx);
            if (idx >= s.size()) break;
            if (s[idx] == ',') {
                ++idx;
                skip_ws(s, idx);
            } else if (s[idx] == '}') {
                ++idx;
                return true;
            } else {
                err = std::string("Expected ',' or '}' in object at position ") + std::to_string(idx);
                return false;
            }
        }
        err = "Unterminated object";
        return false;
    }
};

} // namespace guild::server::json
