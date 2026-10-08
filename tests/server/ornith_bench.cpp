#include "guild/models/store.hpp"
#include "guild/models/registry.hpp"
#include "guild/runtime/model.hpp"
#include "guild/runtime/session.hpp"
#include "guild/memory/planner.hpp"
#include "guild/cli/hardware.hpp"
#include "../check.hpp"

#include <cuda_runtime.h>
#include <filesystem>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <sstream>
#include <vector>
#include <numeric>
#include <algorithm>

static size_t get_host_ram_used_bytes() {
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        if (line.rfind("VmRSS:", 0) == 0) {
            long kb = 0;
            if (std::sscanf(line.c_str(), "VmRSS: %ld kB", &kb) == 1) {
                return static_cast<size_t>(kb) * 1024;
            }
        }
    }
    return 0;
}

int main(int argc, char** argv) {
    int64_t target_context = 4096;
    int64_t target_prompt_len = 32;
    int64_t target_gen_tokens = 128;
    int target_runs = 3;
    std::string out_csv;
    std::string out_json;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--context" && i + 1 < argc) target_context = std::stoll(argv[++i]);
        else if (arg == "--prompt-len" && i + 1 < argc) target_prompt_len = std::stoll(argv[++i]);
        else if (arg == "--gen-tokens" && i + 1 < argc) target_gen_tokens = std::stoll(argv[++i]);
        else if (arg == "--runs" && i + 1 < argc) target_runs = std::stoi(argv[++i]);
        else if (arg == "--csv" && i + 1 < argc) out_csv = argv[++i];
        else if (arg == "--json" && i + 1 < argc) out_json = argv[++i];
        else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: ornith_bench [--context N] [--prompt-len N] [--gen-tokens N] [--runs N] [--csv PATH] [--json PATH]\n";
            return 0;
        }
    }

    if (target_prompt_len + target_gen_tokens > target_context) {
        target_context = ((target_prompt_len + target_gen_tokens + 511) / 512) * 512;
    }

    std::cout << "====================================================\n";
    std::cout << "          GUILD BENCHMARK: ORNITH-1.5-35B           \n";
    std::cout << "====================================================\n";
    std::cout << "Configuration:\n";
    std::cout << "  Context Capacity:  " << target_context << " tokens\n";
    std::cout << "  Prompt Tokens:     " << target_prompt_len << " tokens\n";
    std::cout << "  Generation Target: " << target_gen_tokens << " tokens\n";
    std::cout << "  Iterations:        " << target_runs << " runs\n\n";

    guild::models::StoreOptions st_opts;
    guild::models::ModelStore store(st_opts);
    store.init();

    auto m_opt = guild::models::ModelRegistry::instance().resolve("ornith:35b", store);
    CHECK(m_opt.has_value());

    const auto& manifest = *m_opt;
    auto desc = manifest.to_descriptor();
    const auto hw = guild::cli::detect_hardware();

    guild::memory::PlannerOptions popts;
    popts.context_length = target_context;
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
    auto t_load_start = std::chrono::steady_clock::now();
    auto model = guild::runtime::GuildModel::load(paths, desc, plan, err);
    CHECK(model != nullptr);
    auto t_load_end = std::chrono::steady_clock::now();
    double load_s = std::chrono::duration<double>(t_load_end - t_load_start).count();

    size_t free_vram = 0, total_vram = 0;
    cudaMemGetInfo(&free_vram, &total_vram);
    size_t used_vram = total_vram - free_vram;
    size_t host_ram = get_host_ram_used_bytes();

    std::cout << "Model Loaded in " << std::fixed << std::setprecision(2) << load_s << " s\n";
    std::cout << "Initial Memory State:\n";
    std::cout << "  VRAM Used:        " << std::fixed << std::setprecision(2)
              << (double)used_vram / (1024.0 * 1024.0) << " MiB / "
              << (double)total_vram / (1024.0 * 1024.0) << " MiB\n";
    std::cout << "  Host RAM (RSS):   " << std::fixed << std::setprecision(2)
              << (double)host_ram / (1024.0 * 1024.0 * 1024.0) << " GiB\n\n";

    // Build prompt tokens
    std::vector<int32_t> prompt_tokens;
    if (target_prompt_len <= 16) {
        std::string p_str = "<|im_start|>user\nWhat is 2 + 2?<|im_end|>\n<|im_start|>assistant\n";
        std::string tok_err;
        model->tokenizer().encode(p_str, prompt_tokens, true, tok_err);
    } else {
        // Repeat tokens to target length
        std::string base = "The quick brown fox jumps over the lazy dog repeatedly to test transformer attention. ";
        std::vector<int32_t> base_toks;
        std::string tok_err;
        model->tokenizer().encode(base, base_toks, false, tok_err);
        while (static_cast<int64_t>(prompt_tokens.size()) < target_prompt_len) {
            prompt_tokens.insert(prompt_tokens.end(), base_toks.begin(), base_toks.end());
        }
        prompt_tokens.resize(static_cast<size_t>(target_prompt_len));
    }

    // Warmup run (4 tokens)
    std::cout << "Running warmup... " << std::flush;
    {
        auto session = model->create_session(target_context);
        guild::runtime::GenerationRequest req;
        req.prompt_tokens = {846, 198, 3710};
        req.max_new_tokens = 4;
        req.temperature = 0.0f;
        guild::runtime::RuntimeTelemetry telem;
        std::string gen_err;
        CHECK(session->generate(req, nullptr, telem, gen_err));
    }
    std::cout << "done.\n\n";

    // Repeated runs
    std::vector<double> prompt_tok_s_runs;
    std::vector<double> decode_tok_s_runs;
    std::vector<double> ttft_ms_runs;
    std::vector<double> decode_ms_runs;
    guild::runtime::RuntimeTelemetry last_telem;
    size_t peak_vram = used_vram;
    size_t peak_ram = host_ram;

    for (int r = 0; r < target_runs; ++r) {
        std::cout << "--- Run " << (r + 1) << " of " << target_runs << " ---\n";
        auto session = model->create_session(target_context);
        guild::runtime::GenerationRequest req;
        req.prompt_tokens = prompt_tokens;
        req.max_new_tokens = static_cast<int>(target_gen_tokens);
        req.temperature = 0.0f;

        guild::runtime::RuntimeTelemetry telem;
        std::string gen_err;
        auto t_start = std::chrono::steady_clock::now();

        bool first_tok = true;
        auto t_first = t_start;
        bool ok = session->generate(req, [&](const guild::runtime::TokenEvent& ev) {
            if (first_tok) {
                t_first = std::chrono::steady_clock::now();
                first_tok = false;
            }
            return true;
        }, telem, gen_err);

        auto t_end = std::chrono::steady_clock::now();
        CHECK(ok);
        CHECK(telem.completion_tokens > 0);

        double ttft_ms = std::chrono::duration<double, std::milli>(t_first - t_start).count();
        prompt_tok_s_runs.push_back(telem.prompt_tok_s);
        decode_tok_s_runs.push_back(telem.decode_tok_s);
        ttft_ms_runs.push_back(ttft_ms);
        decode_ms_runs.push_back(telem.decode_ms);
        last_telem = telem;

        cudaMemGetInfo(&free_vram, &total_vram);
        size_t cur_vram = total_vram - free_vram;
        size_t cur_ram = get_host_ram_used_bytes();
        if (cur_vram > peak_vram) peak_vram = cur_vram;
        if (cur_ram > peak_ram) peak_ram = cur_ram;

        std::cout << "  Prompt: " << telem.prompt_tokens << " tokens, "
                  << std::fixed << std::setprecision(2) << telem.prompt_ms << " ms ("
                  << telem.prompt_tok_s << " tok/s)\n";
        std::cout << "  TTFT:   " << std::fixed << std::setprecision(2) << ttft_ms << " ms\n";
        std::cout << "  Decode: " << telem.completion_tokens << " tokens, "
                  << std::fixed << std::setprecision(2) << telem.decode_ms << " ms ("
                  << telem.decode_tok_s << " tok/s)\n";
    }

    auto median_of = [](std::vector<double> v) -> double {
        if (v.empty()) return 0.0;
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    };

    double med_prompt_tok_s = median_of(prompt_tok_s_runs);
    double med_decode_tok_s = median_of(decode_tok_s_runs);
    double med_ttft_ms = median_of(ttft_ms_runs);
    double med_decode_ms = median_of(decode_ms_runs);

    // Compute RAM bandwidth: 8 experts * 40 layers * ~1.69 MB = ~540 MB per token
    double bytes_per_token = 40.0 * 8.0 * (512.0 * 2048.0 * 4.5 / 8.0) * 3.0; // 3 matrices: gate, up, down
    double med_bandwidth_gb_s = (med_decode_tok_s * bytes_per_token) / (1024.0 * 1024.0 * 1024.0);

    // KV Cache sizing: 2 kv heads * 256 dim * 2 bytes * 10 layers = 20,480 bytes/token
    int64_t total_tokens_in_ctx = last_telem.prompt_tokens + last_telem.completion_tokens;
    double kv_host_bytes = static_cast<double>(total_tokens_in_ctx * 20480);
    double kv_staging_bytes = static_cast<double>(std::min<int64_t>(target_context, 16384) * 20480);

    std::cout << "\n=================== BENCHMARK SUMMARY ===================\n";
    std::cout << "Model:                 Ornith-1.5-35B Q4_K_M (Dense Causal GQA)\n";
    std::cout << "Context Target:        " << target_context << " tokens\n";
    std::cout << "Prompt Length:         " << last_telem.prompt_tokens << " tokens\n";
    std::cout << "Generation Length:     " << last_telem.completion_tokens << " tokens\n";
    std::cout << "Median Prompt Tok/s:   " << std::fixed << std::setprecision(2) << med_prompt_tok_s << " tok/s\n";
    std::cout << "Median TTFT:           " << std::fixed << std::setprecision(2) << med_ttft_ms << " ms\n";
    std::cout << "Median Decode Tok/s:   " << std::fixed << std::setprecision(2) << med_decode_tok_s << " tok/s\n";
    std::cout << "Est. RAM Bandwidth:    " << std::fixed << std::setprecision(2) << med_bandwidth_gb_s << " GiB/s\n";
    std::cout << "Peak VRAM Used:        " << std::fixed << std::setprecision(2) << (double)peak_vram / (1024.0 * 1024.0) << " MiB\n";
    std::cout << "Peak Host RAM (RSS):   " << std::fixed << std::setprecision(2) << (double)peak_ram / (1024.0 * 1024.0 * 1024.0) << " GiB\n";
    std::cout << "KV Footprint (Host):   " << std::fixed << std::setprecision(2) << kv_host_bytes / (1024.0 * 1024.0) << " MiB\n";
    std::cout << "KV Staging (GPU VRAM): " << std::fixed << std::setprecision(2) << kv_staging_bytes / (1024.0 * 1024.0) << " MiB\n";
    std::cout << "RAM Expert Hits:       " << last_telem.ram_expert_hits << "\n";
    std::cout << "File Expert Reads:     " << last_telem.file_expert_reads << "\n";
    std::cout << "GPU Expert Cache Hits: " << last_telem.gpu_cache_hits << "\n";
    std::cout << "=========================================================\n\n";

    if (!out_csv.empty()) {
        bool write_header = !std::filesystem::exists(out_csv);
        std::ofstream csv(out_csv, std::ios::app);
        if (write_header) {
            csv << "context,prompt_len,gen_tokens,prompt_tok_s,ttft_ms,decode_tok_s,ram_bw_gib_s,peak_vram_mib,peak_ram_gib,kv_host_mib,kv_gpu_mib\n";
        }
        csv << target_context << "," << last_telem.prompt_tokens << "," << last_telem.completion_tokens << ","
            << med_prompt_tok_s << "," << med_ttft_ms << "," << med_decode_tok_s << ","
            << med_bandwidth_gb_s << "," << ((double)peak_vram / (1024.0 * 1024.0)) << ","
            << ((double)peak_ram / (1024.0 * 1024.0 * 1024.0)) << ","
            << (kv_host_bytes / (1024.0 * 1024.0)) << "," << (kv_staging_bytes / (1024.0 * 1024.0)) << "\n";
    }

    if (!out_json.empty()) {
        std::ofstream jf(out_json);
        jf << "{\n"
           << "  \"model\": \"Ornith-1.5-35B-Q4_K_M\",\n"
           << "  \"context_capacity\": " << target_context << ",\n"
           << "  \"prompt_tokens\": " << last_telem.prompt_tokens << ",\n"
           << "  \"completion_tokens\": " << last_telem.completion_tokens << ",\n"
           << "  \"median_prompt_tok_s\": " << med_prompt_tok_s << ",\n"
           << "  \"median_decode_tok_s\": " << med_decode_tok_s << ",\n"
           << "  \"median_ttft_ms\": " << med_ttft_ms << ",\n"
           << "  \"ram_bandwidth_gib_s\": " << med_bandwidth_gb_s << ",\n"
           << "  \"peak_vram_mib\": " << ((double)peak_vram / (1024.0 * 1024.0)) << ",\n"
           << "  \"peak_ram_gib\": " << ((double)peak_ram / (1024.0 * 1024.0 * 1024.0)) << ",\n"
           << "  \"kv_host_mib\": " << (kv_host_bytes / (1024.0 * 1024.0)) << ",\n"
           << "  \"kv_gpu_staging_mib\": " << (kv_staging_bytes / (1024.0 * 1024.0)) << ",\n"
           << "  \"runs\": " << target_runs << "\n"
           << "}\n";
    }

    return 0;
}
