// tests/core/decision_render_test.cpp - the decision prompts (include/guild/decision/decision.hpp) against the
// reference package's (startlux_decision/jevfmt.py with the checkpoint's tokenizer).
//
//   decision_render_test                                   Python's json.dumps / state_text on fixed cases
//   decision_render_test <tokenizer dir> <requests.jsonl> <reference.jsonl>
//       and every question the reference read in one pass: its prompt tokens, rendered here from the request,
//       must equal the reference's (reference.jsonl lines: {"id", "questions": {name: {"tokens": [...]}}})
#include "guild/decision/decision.hpp"

#include <cstdio>
#include <fstream>
#include <map>
#include <string>

using guild::decision::JsonValue;

namespace {

int fails = 0;

void expect(const std::string& what, const std::string& got, const std::string& want) {
    if (got != want) {
        std::printf("FAIL %s\n  got  %s\n  want %s\n", what.c_str(), got.c_str(), want.c_str());
        ++fails;
    }
}

JsonValue parse(const std::string& s) {
    std::string err;
    auto v = JsonValue::parse(s, err);
    if (!v) {
        std::printf("FAIL parse %s: %s\n", s.c_str(), err.c_str());
        ++fails;
        return JsonValue();
    }
    return *v;
}

// expected values printed by Python 3.14 with the reference's jevfmt (json.dumps(v, ensure_ascii=False) and
// jevfmt.state_text(v) of json.loads(input))
void python_text() {
    const std::string in1 =
        R"({"a": 1, "b": 1.0, "c": 1e16, "d": 1e-05, "e": [1.5, -0.0, 123456789012345678901234567890, )"
        R"(0.1, 100.0, 1e15, 12345678901234567.0, 2.5e-7, -0], "f": "x\"y\n\u0001é\t\\", )"
        R"("g": null, "h": true, "i": {}})";
    const std::string dump1 = "{\"a\": 1, \"b\": 1.0, \"c\": 1e+16, \"d\": 1e-05, \"e\": [1.5, -0.0, "
                              "123456789012345678901234567890, 0.1, 100.0, 1000000000000000.0, 1.2345678901234568e+16, "
                              "2.5e-07, 0], \"f\": \"x\\\"y\\n\\u0001\xc3\xa9\\t\\\\\", \"g\": null, \"h\": true, "
                              "\"i\": {}}";
    const std::string state1 =
        "{\"a\": 1, \"b\": 1.0, \"c\": 1e+16, \"d\": 1e-05, \"e\": [{\"_index\": 0, \"value\": 1.5}, {\"_index\": 1, "
        "\"value\": -0.0}, {\"_index\": 2, \"value\": 123456789012345678901234567890}, {\"_index\": 3, \"value\": "
        "0.1}, {\"_index\": 4, \"value\": 100.0}, {\"_index\": 5, \"value\": 1000000000000000.0}, {\"_index\": 6, "
        "\"value\": 1.2345678901234568e+16}, {\"_index\": 7, \"value\": 2.5e-07}, {\"_index\": 8, \"value\": 0}], "
        "\"f\": \"x\\\"y\\n\\u0001\xc3\xa9\\t\\\\\", \"g\": null, \"h\": true, \"i\": {}}";
    const JsonValue v1 = parse(in1);
    expect("json.dumps 1", guild::decision::py_json_dumps(v1), dump1);
    expect("state_text 1", guild::decision::state_text(v1), state1);
    const JsonValue v2 = parse("[1, 2, 3, 4, 5, 6, 7, 8]");
    expect("state_text 2", guild::decision::state_text(v2),
           "[{\"_index\": 0, \"value\": 1}, {\"_index\": 1, \"value\": 2}, {\"_index\": 2, \"value\": 3}, "
           "{\"_index\": 3, \"value\": 4}, {\"_index\": 4, \"value\": 5}, {\"_index\": 5, \"value\": 6}, "
           "{\"_index\": 6, \"value\": 7}, {\"_index\": 7, \"value\": 8}]");
    const JsonValue v3 = parse(R"([{"_index": 9, "a": 1}, {"b": 2}, 3, "s", null, [1], 7, 8])");
    expect("state_text 3", guild::decision::state_text(v3),
           "[{\"_index\": 9, \"a\": 1}, {\"_index\": 1, \"b\": 2}, {\"_index\": 2, \"value\": 3}, {\"_index\": 3, "
           "\"value\": \"s\"}, {\"_index\": 4, \"value\": null}, {\"_index\": 5, \"value\": [1]}, {\"_index\": 6, "
           "\"value\": 7}, {\"_index\": 7, \"value\": 8}]");
    expect("state_text null", guild::decision::state_text(parse("null")), "");
    expect("state_text {}", guild::decision::state_text(parse("{}")), "");
    expect("strip", guild::decision::py_strip("\xe3\x80\x80 x\x1f \t"), "x");
}

int against_reference(const std::string& tok_dir, const std::string& requests, const std::string& reference) {
    guild::runtime::Tokenizer tok;
    guild::runtime::ChatTemplate tmpl;
    std::string err;
    if (!tok.load(tok_dir, err) || !tmpl.load(tok_dir, err)) {
        std::printf("FAIL tokenizer %s: %s\n", tok_dir.c_str(), err.c_str());
        return 1;
    }
    std::map<std::string, JsonValue> reqs;
    std::ifstream rq(requests);
    for (std::string line; std::getline(rq, line);) {
        if (line.empty())
            continue;
        JsonValue v = parse(line);
        reqs[v["id"].as_string()] = v;
    }
    std::ifstream rf(reference);
    int questions = 0, same = 0;
    for (std::string line; std::getline(rf, line);) {
        if (line.empty())
            continue;
        const JsonValue ref = parse(line);
        const std::string id = ref["id"].as_string();
        const auto it = reqs.find(id);
        if (it == reqs.end()) {
            std::printf("FAIL %s: not in %s\n", id.c_str(), requests.c_str());
            ++fails;
            continue;
        }
        const JsonValue& req = it->second;
        for (const auto& [k, q] : ref["questions"].obj_val) {
            guild::decision::Row row;
            std::vector<int32_t> ids;
            ++questions;
            if (!guild::decision::from_systemone(req["state"], req["questions"][k], k, row, err) ||
                !guild::decision::render_ids(row, tmpl, tok, ids, err)) {
                std::printf("FAIL %s/%s: %s\n", id.c_str(), k.c_str(), err.c_str());
                ++fails;
                continue;
            }
            std::vector<int32_t> want;
            for (const auto& t : q["tokens"].arr_val)
                want.push_back((int32_t) t.as_int());
            if (ids != want) {
                size_t at = 0;
                while (at < ids.size() && at < want.size() && ids[at] == want[at])
                    ++at;
                std::printf("FAIL %s/%s: %zu tokens against %zu, first difference at %zu (%d against %d)\n", id.c_str(),
                            k.c_str(), ids.size(), want.size(), at, at < ids.size() ? ids[at] : -1,
                            at < want.size() ? want[at] : -1);
                ++fails;
                continue;
            }
            ++same;
        }
    }
    std::printf("%d of %d prompts token-identical to the reference\n", same, questions);
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    python_text();
    if (argc == 4)
        against_reference(argv[1], argv[2], argv[3]);
    else if (argc != 1) {
        std::printf("usage: decision_render_test [<tokenizer dir> <requests.jsonl> <reference.jsonl>]\n");
        return 2;
    }
    std::printf("%s (%d failures)\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}
