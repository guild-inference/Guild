#include "guild/models/store.hpp"
#include "guild/models/registry.hpp"
#include "guild/runtime/model.hpp"
#include "guild/runtime/session.hpp"
#include "guild/memory/planner.hpp"
#include "guild/cli/hardware.hpp"

#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <vector>

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

int main() {
    std::cout << "====================================================\n";
    std::cout << "          GUILD BENCHMARK: ORNITH-1.5-35B           \n";
    std::cout << "====================================================\n\n";

    guild::models::StoreOptions st_opts;
    guild::models::ModelStore store(st_opts);
    store.init();

    auto m_opt = guild::models::ModelRegistry::instance().resolve("ornith:35b", store);
    if (!m_opt.has_value()) {
        std::cerr << "ornith:35b not found\n";
        return 1;
    }

    const auto& manifest = *m_opt;
    auto desc = manifest.to_descriptor();
    const auto hw = guild::cli::detect_hardware();

    guild::memory::PlannerOptions popts;
    popts.context_length = 2048;
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

    size_t free_vram = 0, total_vram = 0;
    cudaMemGetInfo(&free_vram, &total_vram);
    size_t used_vram = total_vram - free_vram;
    size_t host_ram = get_host_ram_used_bytes();

    std::cout << "Initial Memory State:\n";
    std::cout << "  VRAM Used: " << std::fixed << std::setprecision(2)
              << (double)used_vram / (1024.0 * 1024.0) << " MiB / "
              << (double)total_vram / (1024.0 * 1024.0) << " MiB\n";
    std::cout << "  Host RAM (RSS): " << std::fixed << std::setprecision(2)
              << (double)host_ram / (1024.0 * 1024.0 * 1024.0) << " GiB\n\n";

    // 1. Warmup
    std::string warmup_prompt = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n";
    {
        auto session = model->create_session();
        guild::runtime::GenerationRequest req;
        req.prompt = warmup_prompt;
        req.max_new_tokens = 4;
        req.temperature = 0.0f;
        guild::runtime::RuntimeTelemetry telem;
        std::string gen_err;
        session->generate(req, [](const guild::runtime::TokenEvent&) { return true; }, telem, gen_err);
    }

    // 2. Decode Benchmark (64 tokens)
    std::cout << "--- Decode Throughput Benchmark (64 tokens) ---\n";
    std::string dec_prompt = "<|im_start|>user\nCount from 1 to 50 in words.<|im_end|>\n<|im_start|>assistant\n";
    guild::runtime::RuntimeTelemetry dec_telem;
    {
        auto session = model->create_session();
        guild::runtime::GenerationRequest req;
        req.prompt = dec_prompt;
        req.max_new_tokens = 64;
        req.temperature = 0.0f;
        std::string gen_err;
        std::string out_text;
        auto t0 = std::chrono::steady_clock::now();
        session->generate(req, [&](const guild::runtime::TokenEvent& ev) {
            out_text += ev.text;
            return true;
        }, dec_telem, gen_err);
        auto t1 = std::chrono::steady_clock::now();
        double wall_s = std::chrono::duration<double>(t1 - t0).count();

        cudaMemGetInfo(&free_vram, &total_vram);
        used_vram = total_vram - free_vram;
        host_ram = get_host_ram_used_bytes();

        double ram_gb_per_sec = 0.0;
        if (dec_telem.decode_ms > 0) {
            // Each expert in Ornith Q4_K is ~1.8 MB. 8 active experts * 40 layers = 320 expert passes per token.
            // 320 * 1.8 MB ~ 576 MB of expert parameters touched per decode token.
            double bytes_touched = (double)dec_telem.ram_expert_hits * (512 * 2048 * 4.5 / 8.0); // approx expert bytes
            ram_gb_per_sec = (bytes_touched / (1024.0 * 1024.0 * 1024.0)) / (dec_telem.decode_ms / 1000.0);
        }

        std::cout << "  Tokens Generated: " << dec_telem.completion_tokens << "\n";
        std::cout << "  Prompt Tokens:    " << dec_telem.prompt_tokens << " (in " << dec_telem.prompt_ms << " ms)\n";
        std::cout << "  Prompt Throughput: " << std::fixed << std::setprecision(2) << dec_telem.prompt_tok_s << " tok/s\n";
        std::cout << "  Decode Time:      " << std::fixed << std::setprecision(2) << dec_telem.decode_ms << " ms\n";
        std::cout << "  Decode Throughput:" << std::fixed << std::setprecision(2) << dec_telem.decode_tok_s << " tok/s\n";
        std::cout << "  RAM Expert Hits:  " << dec_telem.ram_expert_hits << "\n";
        std::cout << "  GPU Cache Hits:   " << dec_telem.gpu_cache_hits << "\n";
        std::cout << "  File Reads:       " << dec_telem.file_expert_reads << "\n";
        std::cout << "  Est. RAM Bandwidth:" << std::fixed << std::setprecision(2) << ram_gb_per_sec << " GiB/s\n";
        std::cout << "  VRAM Used:        " << (double)used_vram / (1024.0 * 1024.0) << " MiB\n";
        std::cout << "  Host RAM Used:    " << (double)host_ram / (1024.0 * 1024.0 * 1024.0) << " GiB\n\n";
    }

    // 3. Prompt Prefill Benchmark (longer prompt)
    std::cout << "--- Prompt Processing Benchmark ---\n";
    std::string long_p = "<|im_start|>user\n";
    for (int i = 0; i < 8; ++i) {
        long_p += "The quick brown fox jumps over the lazy dog repeatedly to test transformer prefill ingestion throughput. ";
    }
    long_p += "<|im_end|>\n<|im_start|>assistant\n";

    guild::runtime::RuntimeTelemetry pf_telem;
    {
        auto session = model->create_session();
        guild::runtime::GenerationRequest req;
        req.prompt = long_p;
        req.max_new_tokens = 1;
        req.temperature = 0.0f;
        std::string gen_err;
        session->generate(req, [](const guild::runtime::TokenEvent&) { return true; }, pf_telem, gen_err);

        std::cout << "  Prompt Tokens:    " << pf_telem.prompt_tokens << "\n";
        std::cout << "  Prompt Time:      " << std::fixed << std::setprecision(2) << pf_telem.prompt_ms << " ms\n";
        std::cout << "  Prompt Throughput: " << std::fixed << std::setprecision(2) << pf_telem.prompt_tok_s << " tok/s\n";
    }

    return 0;
}
