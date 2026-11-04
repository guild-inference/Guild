// tests/core/glm5_tokenizer_test.cpp - the C++ tokenizer's `glm4` rules (GLM-5.3-Flash) against llama.cpp's token
// ids (tests/data/glm5_tokenizer_golden.json, written by llama-tokenize).
//
//   glm5_tokenizer_test <golden json> <tokenizer dir>     (tools/guild_tokenizer.py --gguf <shard 1> --out <dir>)
//
// Exit 77 (skipped) without the tokenizer directory.
#include "guild/runtime/tokenizer.hpp"
#include "guild/server/json.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: glm5_tokenizer_test <golden json> <tokenizer dir>\n");
        return 2;
    }
    if (!std::filesystem::exists(std::filesystem::path(argv[2]) / "tokenizer.json")) {
        std::printf("glm5_tokenizer_test: no tokenizer at %s, skipped\n", argv[2]);
        return 77;
    }
    std::ifstream in(argv[1]);
    std::stringstream ss;
    ss << in.rdbuf();
    guild::server::json::JsonValue golden;
    std::string err;
    if (!guild::server::json::JsonValue::parse(ss.str(), golden, err)) {
        std::fprintf(stderr, "glm5_tokenizer_test: %s\n", err.c_str());
        return 1;
    }
    guild::runtime::Tokenizer tok;
    if (!tok.load(argv[2], err)) {
        std::fprintf(stderr, "glm5_tokenizer_test: %s\n", err.c_str());
        return 1;
    }
    int bad = 0, n = 0;
    for (const auto& c : golden["cases"].arr_val) {
        const std::string text = c["text"].as_string();
        std::vector<int32_t> want, got;
        for (const auto& v : c["tokens"].arr_val)
            want.push_back((int32_t) v.as_int(-1));
        if (!tok.encode(text, got, c["allow_special"].as_bool(false), err)) {
            std::fprintf(stderr, "encode failed: %s\n", err.c_str());
            return 1;
        }
        ++n;
        if (got != want) {
            ++bad;
            std::printf("DIFF (special %d) %.40s\n  want", (int) c["allow_special"].as_bool(false), text.c_str());
            for (int32_t i : want)
                std::printf(" %d", i);
            std::printf("\n  got ");
            for (int32_t i : got)
                std::printf(" %d", i);
            std::printf("\n");
        }
    }
    std::printf("%d of %d cases equal to llama.cpp\n", n - bad, n);
    return bad ? 1 : 0;
}
