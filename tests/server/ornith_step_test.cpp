#include "guild/models/store.hpp"
#include "guild/models/registry.hpp"
#include "guild/runtime/model.hpp"
#include "guild/runtime/session.hpp"
#include "../../src/runtime/model_impl.hpp"
#include "guild/memory/planner.hpp"
#include "guild/cli/hardware.hpp"

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
    if (!m_opt.has_value()) {
        std::cerr << "ornith:35b not found in store\n";
        return 1;
    }

    const auto& manifest = *m_opt;
    auto desc = manifest.to_descriptor();
    const auto hw = guild::cli::detect_hardware();

    guild::memory::PlannerOptions popts;
    popts.context_length = 2048; // small context for test
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
    if (!model) {
        std::cerr << "GuildModel::load failed: " << err << "\n";
        return 1;
    }

    // 1. Check tokenization
    std::string prompt = "<|im_start|>";
    auto tokens = model->tokenizer().tokenize(prompt);
    std::cout << "Prompt: " << prompt << "\nToken count: " << tokens.size() << "\nTokens: ";
    for (auto t : tokens) {
        std::cout << t << " ('" << model->tokenizer().decode(t) << "') ";
    }
    std::cout << "\n\n";

    // 2. Run session generate 1 token
    auto session = model->create_session();
    guild::runtime::GenerationRequest req;
    req.prompt = prompt;
    req.max_new_tokens = 1;
    req.temperature = 0.0f; // greedy

    guild::runtime::RuntimeTelemetry telem;
    std::string gen_err;
    std::cout << "Generated tokens: \n";
    bool ok = session->generate(req, [](const guild::runtime::TokenEvent& ev) {
        std::cout << "[" << ev.token_id << "]: '" << ev.text << "'\n" << std::flush;
        return true;
    }, telem, gen_err);

    int64_t n_vocab = model->descriptor().attn.vocab_size;
    if (n_vocab <= 0) n_vocab = 248320;
    std::vector<float> logits(n_vocab);
    if (model->impl() && model->impl()->ver) {
        model->impl()->ver->debug_dump_layer0();
    }
    // Verifier is in model->impl()->ver
    if (model->impl() && model->impl()->ver && model->impl()->ver->copy_logits(0, logits.data())) {
        float max_l = -1e9f, min_l = 1e9f;
        int n_nan = 0, n_inf = 0;
        std::vector<std::pair<float, int>> top;
        for (int i = 0; i < n_vocab; ++i) {
            float v = logits[i];
            if (std::isnan(v)) n_nan++;
            else if (std::isinf(v)) n_inf++;
            else {
                if (v > max_l) max_l = v;
                if (v < min_l) min_l = v;
                top.push_back({v, i});
            }
        }
        std::sort(top.rbegin(), top.rend());
        std::cout << "Logits stats: min=" << min_l << ", max=" << max_l
                  << ", nans=" << n_nan << ", infs=" << n_inf << "\n";
        std::cout << "Top 10 predicted tokens:\n";
        for (int i = 0; i < 10 && i < (int)top.size(); ++i) {
            std::cout << "  #" << i + 1 << ": token " << top[i].second
                      << " (logit " << top[i].first << "): '"
                      << model->tokenizer().decode(top[i].second) << "'\n";
        }
    }

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
    auto q_session = model->create_session();
    bool q_ok = q_session->generate(q_req, [](const guild::runtime::TokenEvent& ev) {
        std::cout << ev.text << std::flush;
        return true;
    }, q_telem, q_err);
    std::cout << "\n\nMulti-token success: " << q_ok << ", prompt_ms=" << q_telem.prompt_ms
              << ", decode_tok_s=" << q_telem.decode_tok_s
              << ", ram_expert_hits=" << q_telem.ram_expert_hits
              << ", file_expert_reads=" << q_telem.file_expert_reads
              << ", gpu_cache_hits=" << q_telem.gpu_cache_hits << "\n";

    return (ok && q_ok) ? 0 : 1;
}
