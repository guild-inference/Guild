// src/decision/decision.cpp - typed decisions from next-token logits: the StartLux-Decision checkpoints' rendering
// (startlux_decision/jevfmt.py) and readout (startlux_decision/model.py), ported so that a request renders to the
// same prompt tokens and reads the same letters.  Python's own text conversions that shape a prompt are reproduced:
// json.dumps of a JSON state (number literals, escapes, key order), str.strip() and str.lower() on criteria, float()
// for score legends.  tests/core/decision_render_test.cpp checks the tokens against the reference's.
#include "guild/decision/decision.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>

namespace guild::decision {

using guild::server::json::JsonType;

const char* const kSystem =
    "Apply the criterion to the evidence. Choose exactly one listed option. Answer with its letter only.";

namespace {

constexpr const char* kThinkOffSuffix = "<think>\n\n</think>\n\n";
constexpr int kMaxLevels = 10;
constexpr int kAnnotateMin = 8;

// ---- UTF-8 <-> code points (lenient: a bad byte stands for itself, as U+FFFD would not round-trip)
std::vector<uint32_t> code_points(const std::string& s) {
    std::vector<uint32_t> out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size();) {
        const unsigned char c = (unsigned char) s[i];
        int n = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 0;
        if (n == 0 || i + n > s.size()) {
            out.push_back(c);
            ++i;
            continue;
        }
        uint32_t cp = n == 1 ? c : n == 2 ? (c & 0x1F) : n == 3 ? (c & 0x0F) : (c & 0x07);
        bool ok = true;
        for (int k = 1; k < n; ++k) {
            const unsigned char d = (unsigned char) s[i + k];
            if ((d >> 6) != 2) {
                ok = false;
                break;
            }
            cp = (cp << 6) | (d & 0x3F);
        }
        if (!ok) {
            out.push_back(c);
            ++i;
            continue;
        }
        out.push_back(cp);
        i += n;
    }
    return out;
}

void append_utf8(std::string& out, uint32_t cp) {
    if (cp < 0x80) {
        out += (char) cp;
    } else if (cp < 0x800) {
        out += (char) (0xC0 | (cp >> 6));
        out += (char) (0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        out += (char) (0xE0 | (cp >> 12));
        out += (char) (0x80 | ((cp >> 6) & 0x3F));
        out += (char) (0x80 | (cp & 0x3F));
    } else {
        out += (char) (0xF0 | (cp >> 18));
        out += (char) (0x80 | ((cp >> 12) & 0x3F));
        out += (char) (0x80 | ((cp >> 6) & 0x3F));
        out += (char) (0x80 | (cp & 0x3F));
    }
}

std::string from_code_points(const std::vector<uint32_t>& cps, size_t a, size_t b) {
    std::string out;
    for (size_t i = a; i < b; ++i)
        append_utf8(out, cps[i]);
    return out;
}

// str.isspace(): the characters Python's str.strip() and re's \s take for whitespace
bool py_isspace(uint32_t cp) {
    return (cp >= 0x09 && cp <= 0x0D) || (cp >= 0x1C && cp <= 0x20) || cp == 0x85 || cp == 0xA0 || cp == 0x1680 ||
           (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029 || cp == 0x202F || cp == 0x205F ||
           cp == 0x3000;
}

// str.lower() for the scripts with case that criteria are written in: Latin (ASCII, Latin-1, Extended-A), Greek,
// Cyrillic, Armenian and the fullwidth Latin letters.  Other case pairs are left as they are (a gap only for a
// criterion that repeats its option id in another case in such a script).
uint32_t py_lower(uint32_t cp) {
    if ((cp >= 'A' && cp <= 'Z') || (cp >= 0xC0 && cp <= 0xDE && cp != 0xD7))
        return cp + 0x20;
    if (cp >= 0x100 && cp <= 0x17F) {
        if (cp == 0x130)
            return 'i';
        if (cp == 0x178)
            return 0xFF;
        const bool odd_upper = (cp >= 0x139 && cp <= 0x148) || (cp >= 0x179 && cp <= 0x17E);
        if (cp == 0x131 || cp == 0x138 || cp == 0x149 || cp == 0x17F)
            return cp;
        return odd_upper ? ((cp & 1) ? cp + 1 : cp) : ((cp & 1) ? cp : cp + 1);
    }
    if ((cp >= 0x391 && cp <= 0x3AB && cp != 0x3A2))
        return cp + 0x20;
    if (cp == 0x386)
        return 0x3AC;
    if (cp >= 0x388 && cp <= 0x38A)
        return cp + 0x25;
    if (cp == 0x38C)
        return 0x3CC;
    if (cp == 0x38E || cp == 0x38F)
        return cp + 0x3F;
    if (cp >= 0x410 && cp <= 0x42F)
        return cp + 0x20;
    if (cp >= 0x400 && cp <= 0x40F)
        return cp + 0x50;
    if (cp >= 0x460 && cp <= 0x4FF && !(cp >= 0x482 && cp <= 0x489)) {
        if (cp >= 0x4C1 && cp <= 0x4CE)
            return (cp & 1) ? cp + 1 : cp;
        if (cp == 0x4C0)
            return 0x4CF;
        return (cp & 1) ? cp : cp + 1;
    }
    if (cp >= 0x531 && cp <= 0x556)
        return cp + 0x30;
    if (cp >= 0xFF21 && cp <= 0xFF3A)
        return cp + 0x20;
    return cp;
}

size_t py_len(const std::string& s) {
    return code_points(s).size();
}

// ---- numbers as Python prints them
// repr(float): the shortest digits that round-trip; positional when the decimal exponent is in -4..15, else
// d.ddde+XX (two exponent digits at least); an integral value keeps ".0"
std::string py_float_repr(double x) {
    if (std::isnan(x))
        return "NaN";
    if (std::isinf(x))
        return x > 0 ? "Infinity" : "-Infinity";
    char buf[64];
    const auto r = std::to_chars(buf, buf + sizeof(buf), x, std::chars_format::scientific);
    std::string sci(buf, r.ptr);
    std::string out;
    size_t i = 0;
    if (sci[0] == '-') {
        out += '-';
        i = 1;
    }
    const size_t e = sci.find('e');
    std::string digits;
    for (size_t k = i; k < e; ++k)
        if (sci[k] != '.')
            digits += sci[k];
    const int exp10 = std::atoi(sci.c_str() + e + 1);
    if (exp10 >= -4 && exp10 < 16) {
        if (exp10 >= 0) {
            std::string ip = digits.substr(0, std::min<size_t>(digits.size(), (size_t) exp10 + 1));
            while ((int) ip.size() < exp10 + 1)
                ip += '0';
            std::string fp = (size_t) exp10 + 1 < digits.size() ? digits.substr((size_t) exp10 + 1) : "";
            out += ip + "." + (fp.empty() ? "0" : fp);
        } else {
            out += "0." + std::string((size_t) (-exp10 - 1), '0') + digits;
        }
    } else {
        out += digits.substr(0, 1);
        if (digits.size() > 1)
            out += "." + digits.substr(1);
        char eb[16];
        std::snprintf(eb, sizeof(eb), "e%c%02d", exp10 < 0 ? '-' : '+', std::abs(exp10));
        out += eb;
    }
    return out;
}

// a JSON number as json.loads + str()/json.dumps print it: an integer literal stays an int (all its digits; -0 is
// 0), anything else becomes a float
std::string py_number(const JsonValue& v) {
    const std::string& t = v.num_text;
    if (!t.empty() && t.find_first_of(".eE") == std::string::npos) {
        const size_t neg = t[0] == '-' ? 1 : 0;
        if (t.find_first_not_of('0', neg) == std::string::npos)
            return "0";
        return t;
    }
    if (!t.empty())
        return py_float_repr(std::strtod(t.c_str(), nullptr));
    const double x = v.num_val; // a computed value: an int when integral, as json.dumps writes an int
    if (std::isfinite(x) && std::floor(x) == x && std::fabs(x) < 9.007199254740992e15) {
        char b[32];
        std::snprintf(b, sizeof(b), "%lld", (long long) x);
        return b;
    }
    return py_float_repr(x);
}

void py_escape(const std::string& s, std::string& out) {
    out += '"';
    for (unsigned char c : s) {
        switch (c) {
        case '"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        case '\b': out += "\\b"; break;
        case '\f': out += "\\f"; break;
        default:
            if (c < 0x20) {
                char b[8];
                std::snprintf(b, sizeof(b), "\\u%04x", c);
                out += b;
            } else {
                out += (char) c;
            }
        }
    }
    out += '"';
}

void py_dumps(const JsonValue& v, std::string& out) {
    switch (v.type) {
    case JsonType::Null: out += "null"; break;
    case JsonType::Bool: out += v.bool_val ? "true" : "false"; break;
    case JsonType::Number: out += py_number(v); break;
    case JsonType::String: py_escape(v.str_val, out); break;
    case JsonType::Array:
        out += '[';
        for (size_t i = 0; i < v.arr_val.size(); ++i) {
            if (i)
                out += ", ";
            py_dumps(v.arr_val[i], out);
        }
        out += ']';
        break;
    case JsonType::Object:
        out += '{';
        for (size_t i = 0; i < v.obj_val.size(); ++i) {
            if (i)
                out += ", ";
            py_escape(v.obj_val[i].first, out);
            out += ": ";
            py_dumps(v.obj_val[i].second, out);
        }
        out += '}';
        break;
    }
}

// a dict literal {"_index": i, **x}: a key x already has keeps "_index"'s place and takes x's value
JsonValue indexed(int64_t i, const JsonValue& x) {
    JsonValue o(std::vector<std::pair<std::string, JsonValue>>{});
    JsonValue idx((int64_t) i);
    o.obj_val.emplace_back("_index", idx);
    for (const auto& kv : x.obj_val) {
        if (kv.first == "_index")
            o.obj_val[0].second = kv.second;
        else
            o.obj_val.push_back(kv);
    }
    return o;
}

// jevfmt.annotate_indices: arrays of 8 or more elements carry each element's index
JsonValue annotate(const JsonValue& x) {
    if (x.is_array()) {
        JsonValue a(std::vector<JsonValue>{});
        const bool mark = (int) x.arr_val.size() >= kAnnotateMin;
        for (size_t i = 0; i < x.arr_val.size(); ++i) {
            JsonValue v = annotate(x.arr_val[i]);
            if (!mark) {
                a.arr_val.push_back(std::move(v));
            } else if (x.arr_val[i].is_object()) {
                a.arr_val.push_back(indexed((int64_t) i, v));
            } else {
                JsonValue o(std::vector<std::pair<std::string, JsonValue>>{});
                o.obj_val.emplace_back("_index", JsonValue((int64_t) i));
                o.obj_val.emplace_back("value", std::move(v));
                a.arr_val.push_back(std::move(o));
            }
        }
        return a;
    }
    if (x.is_object()) {
        JsonValue o(std::vector<std::pair<std::string, JsonValue>>{});
        for (const auto& kv : x.obj_val)
            o.obj_val.emplace_back(kv.first, annotate(kv.second));
        return o;
    }
    return x;
}

// jevfmt.value_text: a string as it is, anything else as JSON; stripped; empty is None
std::optional<std::string> value_text(const JsonValue& v) {
    if (v.is_null())
        return std::nullopt;
    std::string s = v.is_string() ? v.str_val : py_json_dumps(v);
    s = py_strip(s);
    if (s.empty())
        return std::nullopt;
    return s;
}

// str(v) of a JSON scalar (a choice criteria list's items become option ids this way)
bool py_str(const JsonValue& v, std::string& out) {
    switch (v.type) {
    case JsonType::String: out = v.str_val; return true;
    case JsonType::Number: out = py_number(v); return true;
    case JsonType::Bool: out = v.bool_val ? "True" : "False"; return true;
    case JsonType::Null: out = "None"; return true;
    default: return false; // str() of a list or dict is Python's repr: not reproduced
    }
}

// float(s), as score_keys applies it to legend keys: surrounding whitespace, a sign, digits with single
// underscores between them, a fraction, an exponent; inf, infinity and nan in any case
bool py_float(const std::string& raw, double& out) {
    std::string s = py_strip(raw);
    std::string low;
    for (char c : s)
        low += (char) std::tolower((unsigned char) c);
    std::string body = low[0] == '+' || low[0] == '-' ? low.substr(1) : low;
    if (body == "inf" || body == "infinity" || body == "nan") {
        out = body == "nan" ? std::numeric_limits<double>::quiet_NaN() : std::numeric_limits<double>::infinity();
        if (!low.empty() && low[0] == '-')
            out = -out;
        return true;
    }
    std::string clean;
    for (size_t i = 0; i < s.size(); ++i) {
        const char c = s[i];
        if (c == '_') {
            const bool between = i > 0 && i + 1 < s.size() && std::isdigit((unsigned char) s[i - 1]) &&
                                 std::isdigit((unsigned char) s[i + 1]);
            if (!between)
                return false;
            continue;
        }
        if (!(std::isdigit((unsigned char) c) || c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-'))
            return false;
        clean += c;
    }
    if (clean.empty())
        return false;
    char* end = nullptr;
    out = std::strtod(clean.c_str(), &end);
    if (end != clean.c_str() + clean.size())
        return false;
    // strtod takes hex and lone "."; float() takes neither
    if (clean.find_first_of("0123456789") == std::string::npos)
        return false;
    return true;
}

// jevfmt.score_keys: a legend's level order, numeric keys ascending (a stable sort), otherwise as given
std::vector<std::string> score_keys(const JsonValue& crit) {
    std::vector<std::string> keys;
    for (const auto& kv : crit.obj_val)
        keys.push_back(kv.first);
    std::vector<std::pair<double, size_t>> num;
    for (size_t i = 0; i < keys.size(); ++i) {
        double d;
        if (!py_float(keys[i], d))
            return keys;
        num.emplace_back(d, i);
    }
    std::stable_sort(num.begin(), num.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    std::vector<std::string> sorted;
    for (const auto& p : num)
        sorted.push_back(keys[p.second]);
    return sorted;
}

const JsonValue* field(const JsonValue& obj, const char* key) {
    if (!obj.is_object())
        return nullptr;
    for (const auto& kv : obj.obj_val)
        if (kv.first == key)
            return &kv.second;
    return nullptr;
}

// dict.get(a, dict.get(b)): a present key wins even when its value is null
const JsonValue* field2(const JsonValue& obj, const char* a, const char* b) {
    const JsonValue* v = field(obj, a);
    return v ? v : field(obj, b);
}

// re.sub(r"[\s_\-]+", " ", s).strip().lower()
std::string norm(const std::string& s) {
    const auto cps = code_points(s);
    std::vector<uint32_t> out;
    bool run = false;
    for (uint32_t cp : cps) {
        if (py_isspace(cp) || cp == '_' || cp == '-') {
            if (!run)
                out.push_back(' ');
            run = true;
        } else {
            out.push_back(py_lower(cp));
            run = false;
        }
    }
    size_t a = 0, b = out.size();
    while (a < b && py_isspace(out[a]))
        ++a;
    while (b > a && py_isspace(out[b - 1]))
        --b;
    return from_code_points(out, a, b);
}

// jevfmt._BARE (re.I): "A", "(b)", "c.", "12", "(3)", "option_4", "opt 5" - ids that would read as a letter answer
bool bare_id(const std::string& id) {
    const auto c = code_points(id);
    auto ascii_letter = [](uint32_t x) { return (x >= 'A' && x <= 'Z') || (x >= 'a' && x <= 'z'); };
    auto digit = [](uint32_t x) { return x >= '0' && x <= '9'; };
    size_t i = 0;
    const size_t n = c.size();
    // \(?[A-Za-z][\).]?  and  \(?\d{1,3}[\).]?
    {
        size_t k = i;
        if (k < n && c[k] == '(')
            ++k;
        if (k < n && ascii_letter(c[k])) {
            size_t m = k + 1;
            if (m < n && (c[m] == ')' || c[m] == '.'))
                ++m;
            if (m == n)
                return true;
        }
        size_t d = k;
        while (d < n && d - k < 3 && digit(c[d]))
            ++d;
        if (d > k) {
            size_t m = d;
            if (m < n && (c[m] == ')' || c[m] == '.'))
                ++m;
            if (m == n)
                return true;
        }
    }
    // option[_ ]?\d{1,3} and opt[_ ]?\d{1,3}
    for (const char* word : {"option", "opt"}) {
        size_t k = 0;
        bool ok = true;
        for (const char* p = word; *p; ++p, ++k)
            if (k >= n || py_lower(c[k]) != (uint32_t) *p) {
                ok = false;
                break;
            }
        if (!ok)
            continue;
        if (k < n && (c[k] == '_' || c[k] == ' '))
            ++k;
        size_t d = k;
        while (d < n && d - k < 3 && digit(c[d]))
            ++d;
        if (d > k && d == n)
            return true;
    }
    return false;
}

const char* const kLetters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

std::vector<std::string> option_lines(const Row& row) {
    const bool hide = row.type == "choice" && std::all_of(row.options.begin(), row.options.end(), [](const Option& o) {
                          return bare_id(o.id) && o.criterion.has_value() && !o.criterion->empty();
                      });
    std::vector<std::string> lines;
    for (size_t k = 0; k < row.options.size(); ++k) {
        const Option& o = row.options[k];
        const std::string name = row.type == "noul" ? (o.id == "true" ? "yes" : "no") : o.id;
        std::string body;
        if (hide)
            body = *o.criterion;
        else if (!o.criterion || py_strip(*o.criterion).empty() || norm(*o.criterion) == norm(name))
            body = name;
        else
            body = name + ": " + *o.criterion;
        lines.push_back(std::string(1, kLetters[k]) + ") " + body);
    }
    return lines;
}

// number formatting for the CLI's JSON output
std::string num9(double x) {
    if (!std::isfinite(x))
        return "null";
    char b[40];
    std::snprintf(b, sizeof(b), "%.9g", x);
    return b;
}

void dump_to(const JsonValue& v, std::string& out) {
    switch (v.type) {
    case JsonType::Null: out += "null"; break;
    case JsonType::Bool: out += v.bool_val ? "true" : "false"; break;
    case JsonType::Number: out += v.num_text.empty() ? num9(v.num_val) : v.num_text; break;
    case JsonType::String: py_escape(v.str_val, out); break;
    case JsonType::Array:
        out += '[';
        for (size_t i = 0; i < v.arr_val.size(); ++i) {
            if (i)
                out += ", ";
            dump_to(v.arr_val[i], out);
        }
        out += ']';
        break;
    case JsonType::Object:
        out += '{';
        for (size_t i = 0; i < v.obj_val.size(); ++i) {
            if (i)
                out += ", ";
            py_escape(v.obj_val[i].first, out);
            out += ": ";
            dump_to(v.obj_val[i].second, out);
        }
        out += '}';
        break;
    }
}

JsonValue object() {
    return JsonValue(std::vector<std::pair<std::string, JsonValue>>{});
}

void put(JsonValue& o, const std::string& k, JsonValue v) {
    for (auto& kv : o.obj_val)
        if (kv.first == k) {
            kv.second = std::move(v);
            return;
        }
    o.obj_val.emplace_back(k, std::move(v));
}

std::vector<double> softmax(const std::vector<float>& z, double t) {
    std::vector<double> p(z.size());
    double m = -INFINITY;
    for (float v : z)
        m = std::max(m, (double) v / t);
    double s = 0;
    for (size_t i = 0; i < z.size(); ++i)
        s += p[i] = std::exp((double) z[i] / t - m);
    for (double& v : p)
        v /= s;
    return p;
}

} // namespace

std::string py_strip(const std::string& s) {
    const auto cps = code_points(s);
    size_t a = 0, b = cps.size();
    while (a < b && py_isspace(cps[a]))
        ++a;
    while (b > a && py_isspace(cps[b - 1]))
        --b;
    if (a == 0 && b == cps.size())
        return s;
    return from_code_points(cps, a, b);
}

std::string py_json_dumps(const JsonValue& v) {
    std::string out;
    py_dumps(v, out);
    return out;
}

std::string state_text(const JsonValue& state) {
    if (state.is_string())
        return state.str_val;
    if (state.is_null() || (state.is_object() && state.obj_val.empty()) || (state.is_array() && state.arr_val.empty()))
        return "";
    return py_json_dumps(annotate(state));
}

std::string dump(const JsonValue& v) {
    std::string out;
    dump_to(v, out);
    return out;
}

bool load_config(const std::string& path, Config& cfg, std::string& err) {
    std::ifstream f(path);
    if (!f) {
        err = "cannot read " + path;
        return false;
    }
    std::stringstream ss;
    ss << f.rdbuf();
    auto root = JsonValue::parse(ss.str(), err);
    if (!root || !root->is_object()) {
        err = path + ": " + (err.empty() ? "not a JSON object" : err);
        return false;
    }
    cfg = Config{};
    cfg.format = (*root)["format"].as_string("");
    for (const auto& v : (*root)["letter_token_ids"].arr_val)
        cfg.letter_ids.push_back((int32_t) v.as_int(-1));
    for (const auto& kv : (*root)["temperature_by_type"].obj_val)
        cfg.temperature[kv.first] = kv.second.as_double(1.0);
    cfg.max_options = (int) (*root)["max_options_per_pass"].as_int(26);
    if (const JsonValue* w = field(*root, "wide_choice")) {
        cfg.wide_group = (int) (*w)["group"].as_int(25);
        cfg.wide_keep = (int) (*w)["keep"].as_int(3);
        cfg.wide_residual = (*w)["residual"].as_double(1e-3);
    }
    if (cfg.letter_ids.size() != 26 || cfg.max_options < 2 || cfg.max_options > 26 || cfg.wide_group < 2 ||
        cfg.wide_keep < 1) {
        err = path + ": letter_token_ids must list A..Z, max_options_per_pass 2..26, wide_choice group >= 2";
        return false;
    }
    return true;
}

bool from_systemone(const JsonValue& state, const JsonValue& spec, const std::string& qid, Row& row, std::string& err) {
    if (!spec.is_object()) {
        err = "question " + qid + ": a question is an object";
        return false;
    }
    const JsonValue* tv = field(spec, "type");
    std::string t = tv ? (tv->is_string() ? tv->str_val : std::string("?")) : "choice";
    if (t == "bool")
        t = "noul";
    const JsonValue* crit = field2(spec, "criteria", "options");
    const JsonValue* inst = field2(spec, "instructions", "question");
    std::string ins = inst ? value_text(*inst).value_or("") : "";
    row = Row{};
    row.id = qid;
    row.type = t;
    row.state = state_text(state);
    if (t == "choice") {
        if (!crit || !(crit->is_object() || crit->is_array())) {
            err = "question " + qid + ": a choice needs criteria (an object or a list)";
            return false;
        }
        if (crit->is_array()) {
            for (const auto& c : crit->arr_val) {
                std::string id;
                if (!py_str(c, id)) {
                    err = "question " + qid + ": list criteria must be strings, numbers, booleans or null";
                    return false;
                }
                // {str(c): None}: a repeated value keeps its first place
                if (std::none_of(row.options.begin(), row.options.end(), [&](const Option& o) { return o.id == id; }))
                    row.options.push_back(Option{id, std::nullopt});
            }
        } else {
            for (const auto& kv : crit->obj_val)
                row.options.push_back(Option{kv.first, value_text(kv.second)});
        }
    } else if (t == "score") {
        std::vector<const JsonValue*> levels;
        if (crit && crit->is_object()) {
            for (const auto& k : score_keys(*crit))
                levels.push_back(field(*crit, k.c_str()));
        } else if (crit && crit->is_array()) {
            for (const auto& c : crit->arr_val)
                levels.push_back(&c);
        } else {
            err = "question " + qid + ": a score needs criteria (a list of levels or a legend object)";
            return false;
        }
        for (size_t i = 0; i < levels.size(); ++i)
            row.options.push_back(Option{std::to_string(i), value_text(*levels[i])});
    } else if (t == "noul") {
        static const JsonValue none;
        const JsonValue& c = crit && crit->is_object() ? *crit : none;
        const JsonValue* tr = field(c, "true");
        const JsonValue* fa = field(c, "false");
        row.options.push_back(Option{"true", tr ? value_text(*tr) : std::nullopt});
        row.options.push_back(Option{"false", fa ? value_text(*fa) : std::nullopt});
        if (ins.empty())
            ins = "Which answer fits the evidence?";
    } else {
        err = "unknown question type '" + t + "'";
        return false;
    }
    row.instructions = ins;
    return true;
}

bool validate(const Row& row, std::string& err) {
    if (row.type != "choice" && row.type != "score" && row.type != "noul") {
        err = "unknown type '" + row.type + "'";
        return false;
    }
    if (py_strip(row.instructions).empty()) {
        err = "instructions required";
        return false;
    }
    const size_t n = row.options.size();
    if (n < 2 || n > 26) {
        err = "2..26 options required, got " + std::to_string(n);
        return false;
    }
    for (size_t i = 0; i < n; ++i) {
        const std::string& id = row.options[i].id;
        if (py_strip(id).empty() || id.find('\n') != std::string::npos || id.find('\r') != std::string::npos ||
            py_len(id) > 200) {
            err = "bad option id '" + id + "'";
            return false;
        }
        for (size_t j = 0; j < i; ++j)
            if (row.options[j].id == id) {
                err = "duplicate option id";
                return false;
            }
    }
    if (row.type == "score" && n > (size_t) kMaxLevels) {
        err = "score needs 2..10 levels";
        return false;
    }
    return true;
}

std::string user_message(const Row& row) {
    const std::string state = py_strip(row.state).empty() ? "(none)" : row.state;
    std::string s = "Evidence:\n" + state + "\n\nQuestion: " + py_strip(row.instructions) + "\nOptions:\n";
    const auto lines = option_lines(row);
    for (size_t i = 0; i < lines.size(); ++i)
        s += (i ? "\n" : "") + lines[i];
    return s;
}

double choice_confidence(const std::vector<double>& p) {
    const size_t n = p.size();
    if (n < 2)
        return 1.0;
    const double mx = *std::max_element(p.begin(), p.end());
    return std::min(1.0, std::max(0.0, ((double) n * mx - 1) / (double) (n - 1)));
}

double score_confidence(const std::vector<double>& p) {
    const size_t n = p.size();
    const size_t m = (size_t) (std::max_element(p.begin(), p.end()) - p.begin()); // the first of equal maxima
    double spread = 0, even = 0;
    for (size_t i = 0; i < n; ++i) {
        spread += p[i] * std::fabs((double) i - (double) m);
        even += std::fabs((double) i - ((double) n - 1) / 2);
    }
    even /= (double) n;
    return even != 0 ? std::max(0.0, 1 - spread / even) : 1.0;
}

// ---- Decider

double Decider::temperature(const std::string& type) const {
    const auto it = cfg_.temperature.find(type);
    return it == cfg_.temperature.end() ? 1.0 : it->second;
}

bool render_ids(const Row& row, const runtime::ChatTemplate& tmpl, const runtime::Tokenizer& tok,
                std::vector<int32_t>& ids, std::string& err) {
    if (!validate(row, err))
        return false;
    std::vector<runtime::ChatMessage> msgs;
    msgs.emplace_back("system", kSystem);
    // the template trims the message (Jinja's |trim, Python's str.strip): done here with Python's whitespace
    msgs.emplace_back("user", py_strip(user_message(row)));
    runtime::ChatTemplateOptions opts;
    opts.add_generation_prompt = true;
    opts.enable_thinking = false;
    std::string text;
    if (!tmpl.render(msgs, opts, text, err))
        return false;
    if (text.size() < std::strlen(kThinkOffSuffix) ||
        text.compare(text.size() - std::strlen(kThinkOffSuffix), std::string::npos, kThinkOffSuffix) != 0) {
        err = "the chat template lacks the thinking-off assistant prefix";
        return false;
    }
    ids.clear();
    return tok.encode(text, ids, true, err);
}

bool Decider::check(std::string& err) const {
    Row probe;
    probe.type = "noul";
    probe.state = "s";
    probe.instructions = "q";
    probe.options = {Option{"true", std::nullopt}, Option{"false", std::nullopt}};
    std::vector<int32_t> ids;
    if (!render_ids(probe, tmpl_, tok_, ids, err))
        return false;
    std::vector<runtime::ChatMessage> msgs{{"system", kSystem}, {"user", py_strip(user_message(probe))}};
    runtime::ChatTemplateOptions opts;
    opts.enable_thinking = false;
    std::string text;
    if (!tmpl_.render(msgs, opts, text, err))
        return false;
    for (int k = 0; k < 26; ++k) {
        std::vector<int32_t> one, joined;
        if (!tok_.encode(std::string(1, kLetters[k]), one, false, err) ||
            !tok_.encode(text + kLetters[k], joined, true, err))
            return false;
        std::vector<int32_t> want = ids;
        want.insert(want.end(), one.begin(), one.end());
        if (one.size() != 1 || joined != want) {
            err = std::string("letter ") + kLetters[k] + " is not a separate single token after the prefix";
            return false;
        }
        if (one[0] != cfg_.letter_ids[(size_t) k]) {
            err = "tokenizer letter ids differ from decision_config.json";
            return false;
        }
    }
    return true;
}

bool Decider::logits(const std::vector<Row>& rows, std::vector<std::vector<float>>& out,
                     std::vector<std::vector<int32_t>>* tokens, Usage& usage, std::string& err) {
    std::vector<std::vector<int32_t>> enc(rows.size());
    for (size_t i = 0; i < rows.size(); ++i) {
        if ((int) rows[i].options.size() > cfg_.max_options) {
            err = "question " + rows[i].id + ": more options than one pass reads";
            return false;
        }
        if (!render_ids(rows[i], tmpl_, tok_, enc[i], err)) {
            err = "question " + rows[i].id + ": " + err;
            return false;
        }
        if ((int64_t) enc[i].size() > max_length) {
            err = "length " + std::to_string(enc[i].size()) + " > " + std::to_string(max_length);
            return false;
        }
    }
    // the beginning every prompt of this pass shares (each keeps at least its last token): read once, kept
    int64_t common = 0;
    if (share_prefix && rows.size() > 1) {
        common = (int64_t) enc[0].size() - 1;
        for (size_t i = 1; i < enc.size(); ++i) {
            const auto& a = enc[0];
            const auto& b = enc[i];
            int64_t c = 0;
            const int64_t n = (int64_t) std::min(a.size(), b.size());
            while (c < n && a[(size_t) c] == b[(size_t) c])
                ++c;
            common = std::min({common, c, (int64_t) b.size() - 1});
        }
        if (common < 32) // a checkpoint (the GDN state to the host and back) costs more than a few tokens
            common = 0;
    }
    out.assign(rows.size(), {});
    for (size_t i = 0; i < rows.size(); ++i) {
        runtime::ScoreRequest sr;
        sr.tokens = enc[i];
        sr.candidates.assign(cfg_.letter_ids.begin(),
                             cfg_.letter_ids.begin() + (std::ptrdiff_t) rows[i].options.size());
        sr.keep_prefix = i == 0 ? common : 0;
        runtime::ScoreResult res;
        if (!session_.score(sr, res, err))
            return false;
        out[i] = std::move(res.logits);
        usage.input_tokens += (int64_t) enc[i].size();
        usage.cached_tokens += res.cached_tokens;
        usage.prefill_ms += res.prefill_ms;
        usage.window_ms += res.window_ms;
        usage.passes += 1;
    }
    if (tokens)
        *tokens = std::move(enc);
    return true;
}

bool Decider::probs(const std::vector<Row>& rows, std::vector<std::vector<double>>& out, Usage& usage,
                    std::vector<std::vector<float>>* raw, std::vector<std::vector<int32_t>>* tokens, std::string& err) {
    std::vector<std::vector<float>> z;
    if (!logits(rows, z, tokens, usage, err))
        return false;
    out.clear();
    for (size_t i = 0; i < rows.size(); ++i)
        out.push_back(softmax(z[i], temperature(rows[i].type)));
    if (raw)
        *raw = std::move(z);
    return true;
}

// model.py _wide: a choice over more than 26 options.  Near-equal groups (one pass); the top `keep` of each group go
// to a final round (again in groups when there are still too many); the options left out share the residual mass in
// proportion to their first-round probabilities.  The result lists the finalists, then the rest.
bool Decider::wide(const JsonValue& state, const JsonValue& spec, std::vector<std::pair<std::string, double>>& out,
                   Usage& usage, std::string& err) {
    const JsonValue& crit = *field(spec, "criteria");
    std::vector<std::string> keys;
    for (const auto& kv : crit.obj_val)
        keys.push_back(kv.first);
    const size_t n_groups = (keys.size() + (size_t) cfg_.wide_group - 1) / (size_t) cfg_.wide_group;
    const size_t size = keys.size() / n_groups, extra = keys.size() % n_groups;
    std::vector<std::vector<std::string>> groups;
    size_t start = 0;
    for (size_t g = 0; g < n_groups; ++g) {
        const size_t end = start + size + (g < extra ? 1 : 0);
        groups.emplace_back(keys.begin() + (std::ptrdiff_t) start, keys.begin() + (std::ptrdiff_t) end);
        start = end;
    }
    auto subset = [&](const std::vector<std::string>& ks) {
        JsonValue s = spec;
        JsonValue c = object();
        for (const auto& k : ks)
            c.obj_val.emplace_back(k, *field(crit, k.c_str()));
        put(s, "criteria", c);
        return s;
    };
    std::vector<Row> rows(groups.size());
    for (size_t g = 0; g < groups.size(); ++g)
        if (!from_systemone(state, subset(groups[g]), "q", rows[g], err))
            return false;
    std::vector<std::vector<double>> first;
    if (!probs(rows, first, usage, nullptr, nullptr, err))
        return false;
    std::map<std::string, double> first_p;
    std::vector<std::string> finalists;
    for (size_t g = 0; g < groups.size(); ++g) {
        std::vector<size_t> order(groups[g].size());
        for (size_t i = 0; i < order.size(); ++i) {
            order[i] = i;
            first_p[groups[g][i]] = first[g][i];
        }
        std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return first[g][a] > first[g][b]; });
        for (size_t i = 0; i < order.size() && (int) i < cfg_.wide_keep; ++i)
            finalists.push_back(groups[g][order[i]]);
    }
    const JsonValue fin_spec = subset(finalists);
    std::map<std::string, double> final_p;
    if ((int) finalists.size() > cfg_.max_options) {
        std::vector<std::pair<std::string, double>> deeper;
        if (!wide(state, fin_spec, deeper, usage, err))
            return false;
        for (const auto& kv : deeper)
            final_p[kv.first] = kv.second;
    } else {
        Row fin;
        if (!from_systemone(state, fin_spec, "q", fin, err))
            return false;
        std::vector<std::vector<double>> p;
        if (!probs({fin}, p, usage, nullptr, nullptr, err))
            return false;
        for (size_t i = 0; i < finalists.size(); ++i)
            final_p[finalists[i]] = p[0][i];
    }
    std::vector<std::string> rest;
    for (const auto& k : keys)
        if (std::find(finalists.begin(), finalists.end(), k) == finalists.end())
            rest.push_back(k);
    double mass = 0;
    for (const auto& k : rest)
        mass += first_p[k];
    if (mass == 0)
        mass = 1.0;
    out.clear();
    double z = 0;
    for (const auto& k : finalists) {
        out.emplace_back(k, final_p[k] * (1 - cfg_.wide_residual));
        z += out.back().second;
    }
    for (const auto& k : rest) {
        out.emplace_back(k, cfg_.wide_residual * first_p[k] / mass);
        z += out.back().second;
    }
    for (auto& kv : out)
        kv.second /= z;
    return true;
}

bool Decider::decide(const JsonValue& request, JsonValue& answers, Usage& usage,
                     std::map<std::string, QuestionDetail>* detail, std::string& err) {
    answers = object();
    usage = Usage{};
    if (!request.is_object()) {
        err = "a request is a JSON object";
        return false;
    }
    if (const JsonValue* im = field(request, "images");
        im && !im->is_null() && !(im->is_array() && im->arr_val.empty())) {
        err = "images are not supported: this runner reads text and JSON states only";
        return false;
    }
    static const JsonValue none;
    const JsonValue* sp = field(request, "state");
    const JsonValue& state = sp ? *sp : none;
    const JsonValue* qs = field(request, "questions");
    if (!qs || !qs->is_object() || qs->obj_val.empty()) {
        err = "questions must be a non-empty object";
        return false;
    }
    // model.py _split: a single option and wide choices are decided first, the rest read in one pass
    std::vector<std::pair<std::string, Row>> rows;
    for (const auto& [k, q] : qs->obj_val) {
        if (!q.is_object()) {
            err = "question " + k + ": a question is an object";
            return false;
        }
        const JsonValue* tv = field(q, "type");
        const std::string t = tv && tv->is_string() ? tv->str_val : (tv ? "?" : "choice");
        const JsonValue* crit = field(q, "criteria");
        const bool dict = crit && crit->is_object();
        if (t == "choice" && dict && crit->obj_val.size() == 1) {
            const std::string only = crit->obj_val[0].first;
            JsonValue a = object();
            put(a, "type", "choice");
            put(a, "choice", only);
            put(a, "confidence", 1.0);
            JsonValue pr = object();
            put(pr, only, 1.0);
            put(a, "probabilities", pr);
            put(answers, k, a);
        } else if (t == "choice" && dict && (int) crit->obj_val.size() > cfg_.max_options) {
            std::vector<std::pair<std::string, double>> p;
            if (!wide(state, q, p, usage, err)) {
                err = "question " + k + ": " + err;
                return false;
            }
            std::vector<double> pv;
            JsonValue pr = object();
            size_t best = 0;
            for (size_t i = 0; i < p.size(); ++i) {
                pv.push_back(p[i].second);
                put(pr, p[i].first, p[i].second);
                if (p[i].second > p[best].second)
                    best = i;
            }
            JsonValue a = object();
            put(a, "type", "choice");
            put(a, "choice", p[best].first);
            put(a, "confidence", choice_confidence(pv));
            put(a, "probabilities", pr);
            put(answers, k, a);
        } else {
            Row row;
            if (!from_systemone(state, q, k, row, err))
                return false;
            rows.emplace_back(k, std::move(row));
        }
    }
    if (rows.empty())
        return true;
    std::vector<Row> rr;
    for (auto& kv : rows)
        rr.push_back(kv.second);
    std::vector<std::vector<double>> p;
    std::vector<std::vector<float>> z;
    std::vector<std::vector<int32_t>> toks;
    if (!probs(rr, p, usage, &z, &toks, err))
        return false;
    for (size_t i = 0; i < rows.size(); ++i) {
        const std::string& k = rows[i].first;
        const Row& row = rows[i].second;
        const JsonValue& q = *field(*qs, k.c_str());
        JsonValue a = object();
        if (row.type == "noul") {
            put(a, "type", "noul");
            put(a, "noul", p[i][0]); // options are true, false
        } else if (row.type == "score") {
            put(a, "type", "score");
            double s = 0;
            for (size_t j = 0; j < p[i].size(); ++j)
                s += (double) j * p[i][j];
            put(a, "score", s);
            put(a, "confidence", score_confidence(p[i]));
            // levels = question.get("criteria") or []
            std::vector<JsonValue> levels;
            if (const JsonValue* c = field(q, "criteria"); c && c->is_object() && !c->obj_val.empty()) {
                for (const auto& key : score_keys(*c))
                    levels.push_back(*field(*c, key.c_str()));
            } else if (c && c->is_array()) {
                levels = c->arr_val;
            }
            JsonValue legend = object(), pr = object();
            for (size_t j = 0; j < p[i].size(); ++j) {
                put(legend, std::to_string(j), j < levels.size() ? levels[j] : JsonValue(std::to_string(j)));
                put(pr, std::to_string(j), p[i][j]);
            }
            put(a, "legend", legend);
            put(a, "probabilities", pr);
        } else {
            JsonValue pr = object();
            size_t best = 0;
            for (size_t j = 0; j < p[i].size(); ++j) {
                put(pr, row.options[j].id, p[i][j]);
                if (p[i][j] > p[i][best])
                    best = j;
            }
            put(a, "type", "choice");
            put(a, "choice", row.options[best].id);
            put(a, "confidence", choice_confidence(p[i]));
            put(a, "probabilities", pr);
        }
        put(answers, k, a);
        if (detail) {
            QuestionDetail d;
            d.type = row.type;
            for (const auto& o : row.options)
                d.options.push_back(o.id);
            d.tokens = toks[i];
            d.letter_logits = z[i];
            d.temperature = temperature(row.type);
            d.probs = p[i];
            (*detail)[k] = std::move(d);
        }
    }
    return true;
}

} // namespace guild::decision
