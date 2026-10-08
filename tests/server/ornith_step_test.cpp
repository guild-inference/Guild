#include "guild/models/store.hpp"
#include "guild/models/registry.hpp"
#include "guild/runtime/model.hpp"
#include "guild/runtime/session.hpp"
#include "../../src/runtime/model_impl.hpp"
#include "guild/memory/planner.hpp"
#include "guild/cli/hardware.hpp"
#include "../check.hpp"

#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>

int main() {
    std::cout << "[ornith_step_test] Starting diagnostic...\n";

    guild::models::StoreOptions st_opts;
    guild::models::ModelStore store(st_opts);
    store.init();

    auto m_opt = guild::models::ModelRegistry::instance().resolve("ornith:35b", store);
    CHECK(m_opt.has_value());

    const auto& manifest = *m_opt;
    auto desc = manifest.to_descriptor();
    const auto hw = guild::cli::detect_hardware();

    guild::memory::PlannerOptions popts;
    popts.context_length = 8192; // 8K context supporting test past 2051 boundary
    auto plan = guild::memory::MemoryPlanner::plan(desc, hw, popts);

    std::string native_model;
    const auto* prim = manifest.find_file_by_role("primary");
    if (!prim) prim = manifest.find_file_by_role("shard");
    if (prim) native_model = prim->local_path;

    std::string pack_dir = manifest.metadata.at("pack_dir");
    std::string tokenizer_dir = pack_dir + "/tokenizer";

    guild::runtime::ModelPaths paths;
    paths.primary_model_path = native_model;
    paths.pack_dir = pack_dir;
    paths.tokenizer_dir = tokenizer_dir;

    std::string err;
    auto model = guild::runtime::GuildModel::load(paths, desc, plan, err);
    CHECK(model != nullptr);

    // 1. Check tokenization
    std::string prompt = "<|im_start|>";
    std::vector<int32_t> tokens;
    std::string tok_err;
    CHECK(model->tokenizer().encode(prompt, tokens, true, tok_err));
    CHECK(!tokens.empty());
    std::cout << "Prompt: " << prompt << "\nToken count: " << tokens.size() << "\n\n";

    // 2. Run session generate 1 token
    auto session = model->create_session(8192);
    guild::runtime::GenerationRequest req;
    req.prompt = prompt;
    req.max_new_tokens = 1;
    req.temperature = 0.0f; // greedy

    guild::runtime::RuntimeTelemetry telem;
    std::string gen_err;
    bool ok = session->generate(req, [](const guild::runtime::TokenEvent& ev) {
        std::cout << "[" << ev.token_id << "]: '" << ev.text << "'\n" << std::flush;
        return true;
    }, telem, gen_err);
    CHECK(ok);

    int64_t n_vocab = model->descriptor().attn.vocab_size;
    if (n_vocab <= 0) n_vocab = 248320;
    std::vector<float> logits(n_vocab);
    CHECK(model->impl() && model->impl()->ver && model->impl()->ver->copy_logits(0, logits.data()));

    float max_l = -1e9f, min_l = 1e9f;
    int n_nan = 0, n_inf = 0;
    for (int i = 0; i < n_vocab; ++i) {
        float v = logits[i];
        if (std::isnan(v)) n_nan++;
        else if (std::isinf(v)) n_inf++;
        else {
            if (v > max_l) max_l = v;
            if (v < min_l) min_l = v;
        }
    }
    CHECK(n_nan == 0 && n_inf == 0);
    std::cout << "Logits stats: min=" << min_l << ", max=" << max_l << ", nans=" << n_nan << ", infs=" << n_inf << "\n";
    std::cout << "Generation success: " << ok << ", prompt_ms=" << telem.prompt_ms
              << ", decode_tok_s=" << telem.decode_tok_s << "\n\n";

    // 3. Multi-token generation test
    std::string q_prompt = "<|im_start|>user\nWhat is 2 + 2?<|im_end|>\n<|im_start|>assistant\n";
    std::cout << "=== Multi-token chat generation test ===\nPrompt:\n" << q_prompt << "\nResponse: ";
    guild::runtime::GenerationRequest q_req;
    q_req.prompt = q_prompt;
    q_req.max_new_tokens = 32;
    q_req.temperature = 0.0f;
    guild::runtime::RuntimeTelemetry q_telem;
    std::string q_err;
    auto q_session = model->create_session(8192);
    bool q_ok = q_session->generate(q_req, [](const guild::runtime::TokenEvent& ev) {
        std::cout << ev.text << std::flush;
        return true;
    }, q_telem, q_err);
    CHECK(q_ok);
    std::cout << "\n\nMulti-token success: " << q_ok << ", prompt_ms=" << q_telem.prompt_ms
              << ", decode_tok_s=" << q_telem.decode_tok_s
              << ", ram_expert_hits=" << q_telem.ram_expert_hits
              << ", file_expert_reads=" << q_telem.file_expert_reads
              << ", gpu_cache_hits=" << q_telem.gpu_cache_hits << "\n\n";

    // 4. Test prompt length crossing old 2051 selection bound: exactly 2,052 tokens
    std::cout << "=== Test 4: Crossing 2,051-token boundary with 2,052 prompt tokens ===\n";
    {
        std::vector<int32_t> long_prompt_ids(2052, 9419); // 9419 = 'Hello'
        guild::runtime::GenerationRequest b_req;
        b_req.prompt_tokens = long_prompt_ids;
        b_req.max_new_tokens = 4;
        b_req.temperature = 0.0f;
        guild::runtime::RuntimeTelemetry b_telem;
        std::string b_err;
        auto b_session = model->create_session(8192);
        bool b_ok = b_session->generate(b_req, nullptr, b_telem, b_err);
        if (!b_ok) {
            std::cerr << "2052 prompt test failed: " << b_err << "\n";
        }
        CHECK(b_ok);
        CHECK(b_telem.completion_tokens == 4);
        CHECK(b_telem.prompt_tokens == 2052);
        std::cout << "2052 tokens prompt success: tokens=" << b_telem.completion_tokens
                  << ", prompt_ms=" << b_telem.prompt_ms
                  << ", prompt_tok_s=" << b_telem.prompt_tok_s
                  << ", decode_tok_s=" << b_telem.decode_tok_s << "\n\n";
    }

    // 5. Test prompt length at 4,096 tokens
    std::cout << "=== Test 5: Dense attention at 4,096 prompt tokens ===\n";
    {
        std::vector<int32_t> p4k_ids(4096, 9419);
        guild::runtime::GenerationRequest c_req;
        c_req.prompt_tokens = p4k_ids;
        c_req.max_new_tokens = 4;
        c_req.temperature = 0.0f;
        guild::runtime::RuntimeTelemetry c_telem;
        std::string c_err;
        auto c_session = model->create_session(8192);
        bool c_ok = c_session->generate(c_req, nullptr, c_telem, c_err);
        if (!c_ok) {
            std::cerr << "4096 prompt test failed: " << c_err << "\n";
        }
        CHECK(c_ok);
        CHECK(c_telem.completion_tokens == 4);
        CHECK(c_telem.prompt_tokens == 4096);
        std::cout << "4096 tokens prompt success: tokens=" << c_telem.completion_tokens
                  << ", prompt_ms=" << c_telem.prompt_ms
                  << ", prompt_tok_s=" << c_telem.prompt_tok_s
                  << ", decode_tok_s=" << c_telem.decode_tok_s << "\n\n";
    }

    std::cout << "ALL ORNITH STEP AND EXTENDED CONTEXT TESTS PASSED!\n";
    return 0;
}
