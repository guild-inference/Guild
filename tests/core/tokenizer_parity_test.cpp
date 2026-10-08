#include "../check.hpp"
#include "guild/runtime/tokenizer.hpp"
#include "guild/runtime/chat_template.hpp"
#include "guild/server/json.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

bool is_valid_utf8(const std::string& str) {
    const size_t len = str.size();
    size_t i = 0;
    while (i < len) {
        const unsigned char c = static_cast<unsigned char>(str[i]);
        if (c < 0x80) {
            i += 1;
        } else if ((c & 0xE0) == 0xC0) {
            if (i + 1 >= len || (static_cast<unsigned char>(str[i + 1]) & 0xC0) != 0x80) return false;
            i += 2;
        } else if ((c & 0xF0) == 0xE0) {
            if (i + 2 >= len ||
                (static_cast<unsigned char>(str[i + 1]) & 0xC0) != 0x80 ||
                (static_cast<unsigned char>(str[i + 2]) & 0xC0) != 0x80) return false;
            i += 3;
        } else if ((c & 0xF8) == 0xF0) {
            if (i + 3 >= len ||
                (static_cast<unsigned char>(str[i + 1]) & 0xC0) != 0x80 ||
                (static_cast<unsigned char>(str[i + 2]) & 0xC0) != 0x80 ||
                (static_cast<unsigned char>(str[i + 3]) & 0xC0) != 0x80) return false;
            i += 4;
        } else {
            return false;
        }
    }
    return true;
}

} // namespace

int main(int argc, char** argv) {
    std::string golden_json_path = "tests/data/tokenizer_golden.json";
    std::string qwen_tok_dir = "/mnt/models-ssd/Strata-data/packs/unsloth-ud-iq4_xs/tokenizer";
    std::string ornith_tok_dir = "/mnt/models-ssd/Strata-data/packs/ornith-1.5-35b/tokenizer";

    if (argc >= 2) golden_json_path = argv[1];
    if (argc >= 3) qwen_tok_dir = argv[2];
    if (argc >= 4) ornith_tok_dir = argv[3];

    std::cout << "--- 1. Testing Unloaded Tokenizer Fail-Closed ---\n";
    {
        guild::runtime::Tokenizer tok;
        CHECK(!tok.is_loaded());
        std::vector<int32_t> ids;
        std::string err;
        CHECK(!tok.encode("hello", ids, false, err));
        CHECK(ids.empty());
        CHECK(!err.empty());
        CHECK(tok.tokenize("hello").empty());
        CHECK(tok.decode(100).empty());
        CHECK(tok.decode(std::vector<int32_t>{100, 200}).empty());
        CHECK(!tok.chat_template().is_loaded());

        // Missing directory
        CHECK(!tok.load("/tmp/nonexistent-tokenizer-dir-12345", err));
        CHECK(!err.empty());
        CHECK(!tok.is_loaded());
    }

    std::cout << "--- 2. Loading Real Qwen Tokenizer Assets ---\n";
    guild::runtime::Tokenizer qwen_tok;
    std::string load_err;
    CHECK(qwen_tok.load(qwen_tok_dir, load_err));
    CHECK(qwen_tok.is_loaded());
    CHECK(qwen_tok.vocab_size() == 248320);
    CHECK(qwen_tok.eos_token_id() == 248046);
    CHECK(qwen_tok.chat_template().is_loaded());
    CHECK(qwen_tok.chat_template().type() == guild::runtime::TemplateType::Qwen);

    std::cout << "--- 3. Loading Golden Test Vectors ---\n";
    std::ifstream gf(golden_json_path);
    CHECK(gf.is_open());
    std::string gcontent((std::istreambuf_iterator<char>(gf)), std::istreambuf_iterator<char>());
    guild::server::json::JsonValue root;
    std::string parse_err;
    CHECK(guild::server::json::JsonValue::parse(gcontent, root, parse_err));
    CHECK(root.is_object());

    const auto& text_cases = root["text_cases"];
    CHECK(text_cases.is_array());
    std::cout << "Testing " << text_cases.size() << " text tokenization cases...\n";

    for (size_t i = 0; i < text_cases.size(); ++i) {
        const auto& c = text_cases[i];
        const std::string name = c["name"].as_string();
        const std::string text = c["text"].as_string();
        const bool allow_special = c["allow_special"].as_bool(true);

        std::vector<int32_t> expected_tokens;
        for (size_t t = 0; t < c["tokens"].size(); ++t) {
            expected_tokens.push_back(static_cast<int32_t>(c["tokens"][t].as_int()));
        }

        std::vector<int32_t> actual_tokens;
        std::string enc_err;
        CHECK(qwen_tok.encode(text, actual_tokens, allow_special, enc_err));
        CHECK(actual_tokens.size() == expected_tokens.size());
        for (size_t j = 0; j < expected_tokens.size(); ++j) {
            if (actual_tokens[j] != expected_tokens[j]) {
                std::cerr << "Mismatch in test case '" << name << "' at token index " << j
                          << ": expected " << expected_tokens[j] << ", got " << actual_tokens[j] << "\n";
                CHECK(actual_tokens[j] == expected_tokens[j]);
            }
        }

        // Test detokenization
        std::string decoded = qwen_tok.decode(actual_tokens, false);
        const std::string expected_decoded = c["decoded"].as_string();
        CHECK(decoded == expected_decoded);
    }
    std::cout << "All text tokenization cases PASSED with 100% exact parity.\n";

    std::cout << "--- 4. Testing Stateful Incremental Decoder ---\n";
    {
        for (size_t i = 0; i < text_cases.size(); ++i) {
            const auto& c = text_cases[i];
            const std::string name = c["name"].as_string();
            std::vector<int32_t> tokens;
            for (size_t t = 0; t < c["tokens"].size(); ++t) {
                tokens.push_back(static_cast<int32_t>(c["tokens"][t].as_int()));
            }

            auto decoder = qwen_tok.create_incremental_decoder(true);
            std::string reconstructed;
            for (int32_t tid : tokens) {
                std::string chunk = decoder->add(tid);
                if (!chunk.empty()) {
                    CHECK(is_valid_utf8(chunk));
                    reconstructed += chunk;
                }
            }
            std::string final_chunk = decoder->flush();
            if (!final_chunk.empty()) {
                CHECK(is_valid_utf8(final_chunk));
                reconstructed += final_chunk;
            }

            // Must match non-special decode
            std::string expected = qwen_tok.decode(tokens, true);
            CHECK(reconstructed == expected);
        }
        std::cout << "Incremental decoder validated on all golden token streams.\n";
    }

    std::cout << "--- 5. Testing Chat Templates ---\n";
    guild::runtime::Tokenizer ornith_tok;
    std::string ornith_load_err;
    CHECK(ornith_tok.load(ornith_tok_dir, ornith_load_err));
    CHECK(ornith_tok.is_loaded());
    CHECK(ornith_tok.chat_template().is_loaded());
    CHECK(ornith_tok.chat_template().type() == guild::runtime::TemplateType::Ornith);

    const auto& chat_cases = root["chat_cases"];
    CHECK(chat_cases.is_array());
    std::cout << "Testing " << chat_cases.size() << " chat template cases...\n";

    for (size_t i = 0; i < chat_cases.size(); ++i) {
        const auto& c = chat_cases[i];
        const std::string name = c["name"].as_string();
        const std::string model = c["model"].as_string();
        const std::string expected_rendered = c["rendered"].as_string();

        std::vector<guild::runtime::ChatMessage> messages;
        for (size_t m = 0; m < c["messages"].size(); ++m) {
            messages.push_back({
                c["messages"][m]["role"].as_string(),
                c["messages"][m]["content"].as_string()
            });
        }

        std::vector<int32_t> expected_tokens;
        for (size_t t = 0; t < c["tokens"].size(); ++t) {
            expected_tokens.push_back(static_cast<int32_t>(c["tokens"][t].as_int()));
        }

        const auto& target_tok = (model == "ornith") ? ornith_tok : qwen_tok;
        std::string actual_rendered;
        std::string render_err;
        CHECK(target_tok.chat_template().render(messages, true, actual_rendered, render_err));
        if (actual_rendered != expected_rendered) {
            std::cerr << "Rendered mismatch in chat case '" << name << "':\nExpected:\n"
                      << expected_rendered << "\nActual:\n" << actual_rendered << "\n";
            CHECK(actual_rendered == expected_rendered);
        }

        std::vector<int32_t> actual_tokens;
        std::string enc_err;
        CHECK(target_tok.encode(actual_rendered, actual_tokens, true, enc_err));
        CHECK(actual_tokens.size() == expected_tokens.size());
        for (size_t j = 0; j < expected_tokens.size(); ++j) {
            CHECK(actual_tokens[j] == expected_tokens[j]);
        }
    }
    std::cout << "All chat template cases PASSED with 100% exact parity.\n";

    std::cout << "--- 6. Testing Chat Template Error Handling ---\n";
    {
        std::string out, err;
        // Empty messages
        CHECK(!qwen_tok.chat_template().render({}, true, out, err));
        CHECK(err == "No messages provided.");

        // System message not at start
        std::vector<guild::runtime::ChatMessage> bad_order = {
            {"user", "hello"},
            {"system", "be concise"}
        };
        CHECK(!qwen_tok.chat_template().render(bad_order, true, out, err));
        CHECK(err == "System message must be at the beginning.");

        // Unexpected role
        std::vector<guild::runtime::ChatMessage> bad_role = {
            {"tool", "result"}
        };
        CHECK(!qwen_tok.chat_template().render(bad_role, true, out, err));
        CHECK(err.find("Unexpected message role") != std::string::npos);
        std::cout << "Chat template error contracts PASSED.\n";
    }

    std::cout << "--- 7. Measuring Tokenizer Performance ---\n";
    {
        // 1. Measure encoding throughput
        std::string bench_text =
            "The quick brown fox jumps over the lazy dog. Artificial intelligence and machine learning "
            "have transformed computing across data centers and edge devices. 1234567890! "
            "Hierarchical mixture of experts and speculative decoding enable efficient local execution.\n";
        for (int i = 0; i < 6; ++i) bench_text += bench_text; // ~12 KB text

        std::vector<int32_t> bench_ids;
        std::string enc_err;
        CHECK(qwen_tok.encode(bench_text, bench_ids, false, enc_err));

        const int iters = 200;
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) {
            std::vector<int32_t> tmp;
            qwen_tok.encode(bench_text, tmp, false, enc_err);
        }
        auto t1 = std::chrono::steady_clock::now();
        double enc_s = std::chrono::duration<double>(t1 - t0).count();
        double total_bytes = static_cast<double>(bench_text.size() * iters);
        double total_tokens = static_cast<double>(bench_ids.size() * iters);
        double mb_s = (total_bytes / (1024.0 * 1024.0)) / enc_s;
        double tok_s = total_tokens / enc_s;
        std::cout << "Encoding throughput: " << mb_s << " MB/s (" << tok_s << " tokens/sec)\n";

        // 2. Measure decoding throughput
        t0 = std::chrono::steady_clock::now();
        const int dec_iters = 500;
        for (int i = 0; i < dec_iters; ++i) {
            std::string dec = qwen_tok.decode(bench_ids, true);
        }
        t1 = std::chrono::steady_clock::now();
        double dec_s = std::chrono::duration<double>(t1 - t0).count();
        double dec_tok_s = (bench_ids.size() * dec_iters) / dec_s;
        std::cout << "Decoding throughput: " << dec_tok_s << " tokens/sec\n";

        // 3. Measure incremental per-token overhead
        auto decoder = qwen_tok.create_incremental_decoder(true);
        t0 = std::chrono::steady_clock::now();
        const int inc_iters = 100;
        size_t total_inc_tokens = bench_ids.size() * inc_iters;
        for (int i = 0; i < inc_iters; ++i) {
            decoder->reset();
            for (int32_t tid : bench_ids) {
                decoder->add(tid);
            }
            decoder->flush();
        }
        t1 = std::chrono::steady_clock::now();
        double inc_s = std::chrono::duration<double>(t1 - t0).count();
        double ns_per_tok = (inc_s / total_inc_tokens) * 1e9;
        std::cout << "Incremental decoder per-token overhead: " << ns_per_tok << " ns/token\n";
    }

    std::cout << "\nALL TOKENIZER AND CHAT TEMPLATE PARITY TESTS PASSED!\n";
    return 0;
}
