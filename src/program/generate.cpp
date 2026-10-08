// src/program/generate.cpp - Thin CLI / process wrapper around GuildModel and GuildSession
#include "guild/runtime/model.hpp"
#include "guild/runtime/session.hpp"
#include "guild/model/model_descriptor.hpp"
#include "guild/model/archetype.hpp"
#include "guild/memory/planner.hpp"
#include "guild/cli/hardware.hpp"
#include "guild/artifact/gguf_reader.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace guild::runtime;

namespace {

void usage() {
    std::cout << "Usage: guild-generate [options]\n"
              << "  --pack <dir>               Model pack directory\n"
              << "  --native <file>            Primary model GGUF file\n"
              << "  --expert-profile <file>    Expert profile binary\n"
              << "  --mtp <dir>                MTP speculative draft directory\n"
              << "  --tokenizer <dir>          Tokenizer directory\n"
              << "  --max-context <n>          Maximum context length (default 262144)\n"
              << "  --kv <fp16|q8_0|...>       KV cache precision\n"
              << "  --kv-host-only             Host-only KV cache with bounded GPU staging\n"
              << "  --resident-budget-gib <n>  RAM expert residency budget in GiB\n"
              << "  --expert-cache <auto|n>    GPU expert cache slots\n"
              << "  --prefill <auto|n>         Prefill chunk size\n"
              << "  --spec <n>                 Speculative draft length (default 4)\n"
              << "  --spec-min-p <p>           Speculative min probability (default 0.5)\n"
              << "  --pool-workers <n>         CPU expert pool worker count\n"
              << "  --prompt <string>          Input prompt text\n"
              << "  --tokens <ids>             Comma-separated prompt token IDs\n"
              << "  --tokens-file <path>       File containing token IDs\n"
              << "  --max-new <n>              Maximum tokens to generate (default 512)\n"
              << "  --temperature <t>          Sampling temperature (default 0 = greedy)\n"
              << "  --top-p <p>                Top-p nucleus sampling (default 1.0)\n"
              << "  --top-k <k>                Top-k sampling (default 0)\n"
              << "  --seed <s>                 Random seed (default 0)\n"
              << "  --greedy                   Enforce greedy argmax sampling\n"
              << "  --serve                    Run in pipe-based server mode\n";
}

bool parse_token_list(const std::string& str, std::vector<int32_t>& tokens) {
    tokens.clear();
    std::string text = str;
    std::replace(text.begin(), text.end(), ',', ' ');
    std::istringstream iss(text);
    std::string item;
    while (iss >> item) {
        try {
            size_t used = 0;
            const int64_t id = std::stoll(item, &used);
            if (used != item.size() || id < 0 || id > INT32_MAX) { tokens.clear(); return false; }
            tokens.push_back(static_cast<int32_t>(id));
        } catch (...) {
            tokens.clear();
            return false;
        }
    }
    return !tokens.empty();
}

guild::model::ModelDescriptor make_default_descriptor(const std::string& path) {
    guild::model::ModelDescriptor desc;
    desc.name = "Qwen3.8-Flash-Next";
    desc.archetype = guild::model::ModelArchetype::Qwen4Exp;
    desc.arch_name = "qwen4exp";
    desc.file_path = path.empty() ? "Qwen3.8-Flash-Next" : path;
    desc.attn.n_embd = 2560;
    desc.attn.n_layers = 48;
    desc.attn.n_heads = 24;
    desc.attn.n_kv_heads = 2;
    desc.attn.head_dim = 256;
    desc.attn.context_length = 262144;
    desc.attn.vocab_size = 151936;
    desc.attn.pattern = guild::model::AttentionPattern::HybridGDN;
    desc.attn.full_attn_interval = 4;
    desc.moe.n_routed_experts = 512;
    desc.moe.k_active_experts = 10;
    desc.moe.expert_dim_ff = 640;
    desc.moe.n_shared_experts = 1;
    desc.moe.shared_dim_ff = 2560;
    desc.moe.expert_blob_bytes = 2421813;
    return desc;
}

} // namespace

int main(int argc, char** argv) try {
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    if (std::getenv("CUDA_MODULE_LOADING") == nullptr) {
#if defined(_WIN32)
        _putenv_s("CUDA_MODULE_LOADING", "EAGER");
#else
        setenv("CUDA_MODULE_LOADING", "EAGER", 0);
#endif
    }

    std::string pack_dir;
    std::string native_path;
    std::string profile_path;
    std::string mtp_dir;
    std::string tokenizer_dir;
    int64_t max_context = 262144;
    std::string kv_precision = "fp16";
    bool kv_host_only = false;
    int resident_budget_gib = 0;
    std::string expert_cache = "auto";
    std::string prefill = "auto";
    int spec = 4;
    float spec_min_p = 0.5f;
    int pool_workers = 0;
    std::string prompt;
    std::vector<int32_t> prompt_tokens;
    int max_new = 512;
    float temperature = 0.0f;
    float top_p = 1.0f;
    int top_k = 0;
    int seed = 0;
    bool greedy = true;
    bool serve_mode = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* what) -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s requires an argument\n", what);
                std::exit(2);
            }
            return argv[++i];
        };

        if (a == "-h" || a == "--help") {
            usage();
            return 0;
        } else if (a == "--pack") {
            pack_dir = next("--pack");
        } else if (a == "--native") {
            native_path = next("--native");
        } else if (a == "--expert-profile") {
            profile_path = next("--expert-profile");
        } else if (a == "--mtp") {
            mtp_dir = next("--mtp");
        } else if (a == "--tokenizer") {
            tokenizer_dir = next("--tokenizer");
        } else if (a == "--max-context") {
            max_context = std::stoll(next("--max-context"));
        } else if (a == "--kv") {
            kv_precision = next("--kv");
        } else if (a == "--kv-host-only") {
            kv_host_only = true;
        } else if (a == "--resident-budget-gib") {
            resident_budget_gib = std::stoi(next("--resident-budget-gib"));
        } else if (a == "--expert-cache") {
            expert_cache = next("--expert-cache");
        } else if (a == "--prefill") {
            prefill = next("--prefill");
        } else if (a == "--spec") {
            spec = std::stoi(next("--spec"));
        } else if (a == "--spec-min-p") {
            spec_min_p = std::stof(next("--spec-min-p"));
        } else if (a == "--pool-workers") {
            pool_workers = std::stoi(next("--pool-workers"));
        } else if (a == "--prompt") {
            prompt = next("--prompt");
        } else if (a == "--tokens") {
            if (!parse_token_list(next("--tokens"), prompt_tokens)) {
                std::fprintf(stderr, "guild-generate: invalid prompt token list\n");
                return 2;
            }
        } else if (a == "--tokens-file") {
            std::string tpath = next("--tokens-file");
            std::ifstream tf(tpath);
            if (!tf.is_open()) {
                std::fprintf(stderr, "guild-generate: cannot open token file\n");
                return 2;
            }
            std::string content((std::istreambuf_iterator<char>(tf)), std::istreambuf_iterator<char>());
            if (!parse_token_list(content, prompt_tokens)) {
                std::fprintf(stderr, "guild-generate: invalid prompt token file\n");
                return 2;
            }
        } else if (a == "--max-new") {
            max_new = std::stoi(next("--max-new"));
        } else if (a == "--temperature") {
            temperature = std::stof(next("--temperature"));
            if (temperature > 0.0f) greedy = false;
        } else if (a == "--top-p") {
            top_p = std::stof(next("--top-p"));
        } else if (a == "--top-k") {
            top_k = std::stoi(next("--top-k"));
        } else if (a == "--seed") {
            seed = std::stoi(next("--seed"));
        } else if (a == "--greedy") {
            greedy = true;
            temperature = 0.0f;
        } else if (a == "--serve") {
            serve_mode = true;
        }
        // Other legacy / engine flags are accepted silently
        else if (a.rfind("--", 0) == 0 && i + 1 < argc && argv[i + 1][0] != '-') {
            // Check if option was followed by a value
            // If unknown option without value, it won't consume
        }
    }

    if (tokenizer_dir.empty() && !pack_dir.empty()) {
        tokenizer_dir = pack_dir + "/tokenizer";
    }

    // Hardware detection
    const auto hw = guild::cli::detect_hardware();

    // Model Descriptor
    guild::model::ModelDescriptor desc = make_default_descriptor(native_path);
    if (!native_path.empty()) {
        try {
            guild::GgufFile gguf(native_path);
            std::string d_err;
            guild::model::ModelDescriptor parsed;
            if (guild::model::ArchetypeRegistry::instance().describe_gguf(gguf, parsed, d_err)) {
                desc = parsed;
            }
        } catch (...) {}
    }

    // Execution Plan
    guild::memory::PlannerOptions popts;
    popts.context_length = max_context;
    popts.kv_format = guild::memory::kv_precision_from_string(kv_precision);
    if (kv_host_only) {
        popts.kv_mode = guild::memory::KvMode::HostOnly;
    }
    if (resident_budget_gib > 0) {
        popts.resident_budget_gib = resident_budget_gib;
    }
    if (expert_cache != "auto") {
        try { popts.gpu_cache_slots = std::stoi(expert_cache); } catch (...) {}
    }
    if (prefill != "auto") {
        try { popts.prefill_chunk = std::stoi(prefill); } catch (...) {}
    }
    popts.spec_tokens = spec;

    auto plan = guild::memory::MemoryPlanner::plan(desc, hw, popts);

    // Model Paths
    ModelPaths paths;
    paths.primary_model_path = native_path;
    paths.pack_dir = pack_dir;
    paths.expert_profile_path = profile_path;
    paths.mtp_dir = mtp_dir;
    paths.tokenizer_dir = tokenizer_dir;

    (void) spec_min_p;
    (void) pool_workers;

    std::string err_msg;
    auto model = GuildModel::load(paths, desc, plan, err_msg);
    if (!model) {
        std::fprintf(stderr, "guild-generate: model load failed: %s\n", err_msg.c_str());
        return 1;
    }

    auto session = model->create_session(plan.context_length);
    if (!session) {
        std::fprintf(stderr, "guild-generate: session creation failed\n");
        return 1;
    }

    if (serve_mode) {
        std::cout << "READY " << plan.context_length << " stop\n" << std::flush;

        std::string line;
        while (std::getline(std::cin, line)) {
            if (line.empty()) continue;
            if (line == "QUIT") {
                break;
            }
            if (line == "STOP") {
                session->cancel();
                continue;
            }
            if (line.rfind("GEN ", 0) == 0) {
                std::istringstream iss(line.substr(4));
                int req_max_new = 4096;
                iss >> req_max_new;
                float req_temp = 0.0f;
                float req_top_p = 1.0f;
                int req_seed = 0;
                std::string tok_str;
                std::string part;
                while (iss >> part) {
                    if (part.rfind("temperature=", 0) == 0) {
                        req_temp = std::stof(part.substr(12));
                    } else if (part.rfind("top_p=", 0) == 0) {
                        req_top_p = std::stof(part.substr(6));
                    } else if (part.rfind("seed=", 0) == 0) {
                        req_seed = std::stoi(part.substr(5));
                    } else {
                        tok_str = part;
                    }
                }

                GenerationRequest req;
                req.max_new_tokens = req_max_new;
                req.temperature = req_temp;
                req.top_p = req_top_p;
                req.seed = req_seed;

                if (!tok_str.empty()) {
                    if (!parse_token_list(tok_str, req.prompt_tokens)) {
                        std::cout << "ERR invalid prompt token list\n" << std::flush;
                        continue;
                    }
                }

                RuntimeTelemetry telem;
                std::string gen_err;
                auto on_tok = [](const TokenEvent& ev) -> bool {
                    std::cout << "T " << ev.token_id << "\n" << std::flush;
                    return true;
                };
                auto on_pref = [](const PrefillProgressEvent& pe) {
                    std::cout << "PP " << pe.pos << " " << pe.total << " "
                              << pe.ms << " " << pe.tok_s << "\n" << std::flush;
                };

                bool ok = session->generate(req, on_tok, telem, gen_err, on_pref);
                if (!ok) {
                    std::cout << "ERR " << gen_err << "\n" << std::flush;
                } else {
                    std::cout << "DONE " << telem.completion_tokens << " " << telem.prompt_tokens << " "
                              << telem.prompt_ms << " " << telem.decode_ms << " " << telem.finish_reason << "\n"
                              << std::flush;
                }
            }
        }
        return 0;
    }

    // Standard CLI generation mode
    GenerationRequest req;
    req.prompt = prompt;
    req.prompt_tokens = prompt_tokens;
    req.max_new_tokens = max_new;
    req.temperature = greedy ? 0.0f : temperature;
    req.top_p = top_p;
    req.top_k = top_k;
    req.seed = seed;

    RuntimeTelemetry telem;
    std::string gen_err;
    auto on_tok = [](const TokenEvent& ev) -> bool {
        std::cout << ev.text << std::flush;
        return true;
    };
    auto on_pref = [](const PrefillProgressEvent& pe) {
        std::cerr << "Prefill chunk " << pe.pos << "/" << pe.total
                  << " (" << pe.tok_s << " tok/s)\n";
    };

    bool ok = session->generate(req, on_tok, telem, gen_err, on_pref);
    std::cout << "\n";
    if (!ok) {
        std::fprintf(stderr, "guild-generate: generation failed: %s\n", gen_err.c_str());
        return 1;
    }

    std::fprintf(stderr, "\nCompleted %d tokens in %.2f ms (%.2f tok/s), prompt %d tokens in %.2f ms (%.2f tok/s)\n",
                 telem.completion_tokens, telem.decode_ms, telem.decode_tok_s,
                 telem.prompt_tokens, telem.prompt_ms, telem.prompt_tok_s);
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "guild-generate: %s\n", e.what());
    return 2;
}
