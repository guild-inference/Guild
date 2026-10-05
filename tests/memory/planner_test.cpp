#include "guild/memory/planner.hpp"
#include "guild/core/layout.hpp"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <iostream>

#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "planner_test assertion failed at line %d: %s\n", __LINE__, #cond); \
        return 1; \
    } \
} while (0)

using namespace guild::memory;
using namespace guild::model;
using namespace guild::hardware;

ModelDescriptor create_test_qwen_descriptor() {
    ModelDescriptor desc;
    desc.name = "Qwen3.8-Flash-Next-UD-IQ4_XS";
    desc.archetype = ModelArchetype::Qwen4Exp;
    desc.arch_name = "qwen4exp";

    desc.attn.n_embd = 2560;
    desc.attn.n_layers = 48;
    desc.attn.n_heads = 24;
    desc.attn.n_kv_heads = 2;
    desc.attn.head_dim = 256;
    desc.attn.context_length = 262144;
    desc.attn.vocab_size = 151936;
    desc.attn.pattern = AttentionPattern::HybridGDN;
    desc.attn.full_attn_interval = 4;

    desc.moe.n_routed_experts = 512;
    desc.moe.k_active_experts = 10;
    desc.moe.expert_dim_ff = 640;
    desc.moe.n_shared_experts = 1;
    desc.moe.shared_dim_ff = 2560;
    desc.moe.expert_blob_bytes = 2421813; // UD-IQ4_XS: 24576 * 2421813 = 59,518,476,288 B (~55.43 GiB)

    return desc;
}

int main() {
    const auto desc = create_test_qwen_descriptor();

    // =========================================================================
    // Test 1: Synthetic 8 GiB VRAM / 377 GiB RAM chooses host-only KV for 262K FP16
    // =========================================================================
    {
        HardwareInfo hw;
        hw.ram_total_bytes = 377ULL * 1024ULL * 1024ULL * 1024ULL; // 377 GiB
        hw.gpu_vram_bytes = 8ULL * 1024ULL * 1024ULL * 1024ULL;    // 8.0 GiB
        hw.has_cuda = true;
        hw.sync_gib();

        PlannerOptions opts;
        opts.context_length = 262144;
        opts.kv_format = KvPrecision::FP16;

        ExecutionPlan plan = MemoryPlanner::plan(desc, hw, opts);

        CHECK(plan.kv_mode == KvMode::HostOnly);
        CHECK(plan.host_only_kv == true);
        CHECK(plan.full_host_kv_bytes == 6442450944ULL); // 6.00 GiB
        CHECK(plan.kv_staging_bytes == 33685504ULL);     // 32.125 MiB
        CHECK(plan.routed_experts_in_ram == 24576);
        CHECK(plan.routed_experts_on_file == 0);
        CHECK(plan.routed_experts_in_gpu == 0);
        CHECK(plan.prefill_chunk == 512);
        CHECK(plan.mtp_spec_tokens == 4);

        PlanValidation val = MemoryPlanner::validate(plan, hw, desc);
        CHECK(val.ok());

        std::cout << "[Test 1 Passed] 8 GiB / 377 GiB correctly selected host-only KV (6.00 GiB host, 32.1 MiB staging)\n";
    }

    // =========================================================================
    // Test 2: 56 GiB host expert capacity results in zero file-tier experts for 55.43 GiB model
    // =========================================================================
    {
        HardwareInfo hw;
        hw.ram_total_bytes = 128ULL * 1024ULL * 1024ULL * 1024ULL;
        hw.gpu_vram_bytes = 8ULL * 1024ULL * 1024ULL * 1024ULL;
        hw.has_cuda = true;
        hw.sync_gib();

        PlannerOptions opts;
        opts.resident_budget_gib = 56; // Explicit 56 GiB budget

        ExecutionPlan plan = MemoryPlanner::plan(desc, hw, opts);
        CHECK(plan.routed_expert_total_bytes == 59518476288ULL);
        CHECK(plan.routed_experts_in_ram == 24576);
        CHECK(plan.routed_experts_on_file == 0);
        CHECK(plan.routed_experts_file_bytes == 0);

        std::cout << "[Test 2 Passed] 56 GiB budget results in exactly 0 file-tier experts for 55.43 GiB model\n";
    }

    // =========================================================================
    // Test 3: Insufficient host RAM results in deterministic RAM/file split
    // =========================================================================
    {
        HardwareInfo hw;
        hw.ram_total_bytes = 64ULL * 1024ULL * 1024ULL * 1024ULL;
        hw.gpu_vram_bytes = 8ULL * 1024ULL * 1024ULL * 1024ULL;
        hw.has_cuda = true;
        hw.sync_gib();

        PlannerOptions opts;
        opts.resident_budget_gib = 32; // 32 GiB budget (less than 55.43 GiB)

        ExecutionPlan plan = MemoryPlanner::plan(desc, hw, opts);

        const uint64_t budget_bytes = 32ULL * 1024ULL * 1024ULL * 1024ULL;
        const int64_t expected_ram_slots = budget_bytes / 2421813ULL; // 14,187
        const int64_t expected_file_slots = 24576 - expected_ram_slots; // 10,389

        CHECK(plan.routed_experts_in_ram == expected_ram_slots);
        CHECK(plan.routed_experts_on_file == expected_file_slots);
        CHECK(plan.routed_experts_in_ram + plan.routed_experts_on_file == 24576);
        CHECK(plan.routed_experts_ram_bytes + plan.routed_experts_file_bytes == plan.routed_expert_total_bytes);

        std::cout << "[Test 3 Passed] 32 GiB budget produced deterministic split: "
                  << plan.routed_experts_in_ram << " RAM / " << plan.routed_experts_on_file << " file\n";
    }

    // =========================================================================
    // Test 4: Large GPU chooses resident KV where preferable
    // =========================================================================
    {
        HardwareInfo hw;
        hw.ram_total_bytes = 512ULL * 1024ULL * 1024ULL * 1024ULL; // 512 GiB
        hw.gpu_vram_bytes = 80ULL * 1024ULL * 1024ULL * 1024ULL;   // 80 GiB A100
        hw.has_cuda = true;
        hw.sync_gib();

        PlannerOptions opts;
        opts.context_length = 262144;
        opts.kv_format = KvPrecision::FP16;

        ExecutionPlan plan = MemoryPlanner::plan(desc, hw, opts);

        CHECK(plan.kv_mode == KvMode::Resident);
        CHECK(plan.host_only_kv == false);
        CHECK(plan.persistent_gpu_kv_bytes == 6442450944ULL);
        CHECK(plan.kv_staging_bytes == 0);
        CHECK(plan.prefill_chunk == 2048);
        CHECK(plan.routed_experts_in_gpu > 0); // Holds expert cache on 80 GiB card

        PlanValidation val = MemoryPlanner::validate(plan, hw, desc);
        CHECK(val.ok());

        std::cout << "[Test 4 Passed] 80 GiB GPU chose resident KV and allocated "
                  << plan.routed_experts_in_gpu << " GPU expert cache slots\n";
    }

    // =========================================================================
    // Test 5: Explicit FP16 remains FP16
    // =========================================================================
    {
        HardwareInfo hw;
        hw.ram_total_bytes = 64ULL * 1024ULL * 1024ULL * 1024ULL;
        hw.gpu_vram_bytes = 8ULL * 1024ULL * 1024ULL * 1024ULL;
        hw.sync_gib();

        PlannerOptions opts;
        opts.kv_format = KvPrecision::FP16;

        ExecutionPlan plan = MemoryPlanner::plan(desc, hw, opts);
        CHECK(plan.kv_format == KvPrecision::FP16);

        std::cout << "[Test 5 Passed] Explicit FP16 preserved under tight VRAM\n";
    }

    // =========================================================================
    // Test 6: Explicit context remains unchanged
    // =========================================================================
    {
        HardwareInfo hw;
        hw.ram_total_bytes = 64ULL * 1024ULL * 1024ULL * 1024ULL;
        hw.gpu_vram_bytes = 8ULL * 1024ULL * 1024ULL * 1024ULL;
        hw.sync_gib();

        PlannerOptions opts1;
        opts1.context_length = 131072;
        ExecutionPlan plan1 = MemoryPlanner::plan(desc, hw, opts1);
        CHECK(plan1.context_length == 131072);

        PlannerOptions opts2;
        opts2.context_length = 262144;
        ExecutionPlan plan2 = MemoryPlanner::plan(desc, hw, opts2);
        CHECK(plan2.context_length == 262144);

        std::cout << "[Test 6 Passed] Explicit context length (131072 and 262144) preserved\n";
    }

    // =========================================================================
    // Test 7: Planner never emits a plan whose prompt scratch cannot allocate
    // =========================================================================
    {
        HardwareInfo hw;
        hw.ram_total_bytes = 16ULL * 1024ULL * 1024ULL * 1024ULL;
        // Severe VRAM limit: 4.0 GiB (dense is 3.75 GiB, leaving no room for 512 prefill scratch + reserve)
        hw.gpu_vram_bytes = 4ULL * 1024ULL * 1024ULL * 1024ULL;
        hw.sync_gib();

        PlannerOptions opts;
        opts.context_length = 262144;

        ExecutionPlan plan = MemoryPlanner::plan(desc, hw, opts);
        PlanValidation val = MemoryPlanner::validate(plan, hw, desc);

        // Validation MUST fail because prompt scratch cannot allocate
        CHECK(!val.ok());
        CHECK(!val.valid);
        CHECK(!val.errors.empty());

        std::cout << "[Test 7 Passed] Severely constrained GPU correctly rejected by validation:\n"
                  << "  Error: " << val.errors[0] << "\n";
    }

    // =========================================================================
    // Test 8: JSON serialization is stable and testable
    // =========================================================================
    {
        HardwareInfo hw;
        hw.ram_total_bytes = 377ULL * 1024ULL * 1024ULL * 1024ULL;
        hw.gpu_vram_bytes = 8ULL * 1024ULL * 1024ULL * 1024ULL;
        hw.sync_gib();

        ExecutionPlan plan = MemoryPlanner::plan(desc, hw);
        std::string json = plan.to_json_string();

        CHECK(json.find("\"context_length\": 262144") != std::string::npos);
        CHECK(json.find("\"kv_format\": \"FP16\"") != std::string::npos);
        CHECK(json.find("\"kv_mode\": \"host-only\"") != std::string::npos);
        CHECK(json.find("\"host_only_kv\": true") != std::string::npos);
        CHECK(json.find("\"kv_staging_bytes\": 33685504") != std::string::npos);
        CHECK(json.find("\"routed_experts_in_ram\": 24576") != std::string::npos);
        CHECK(json.find("\"routed_experts_on_file\": 0") != std::string::npos);
        CHECK(json.find("\"prefill_chunk\": 512") != std::string::npos);

        std::cout << "[Test 8 Passed] JSON output validated:\n" << json << "\n";
    }

    // =========================================================================
    // Test 9: Existing qwen4exp execution behavior remains unchanged
    // =========================================================================
    {
        guild::core::ModelGeometry g;
        g.apply_descriptor(desc);

        CHECK(g.n_layers == 48);
        CHECK(g.qsa_interval == 4);
        CHECK(g.n_qsa_layers() == 12);
        CHECK(g.n_gdn_layers() == 36);
        CHECK(g.n_expert == 512);
        CHECK(g.n_ff == 640);
        CHECK(g.n_embd == 2560);
        CHECK(g.head_dim == 256);
        CHECK(g.n_head_kv == 2);
        CHECK(g.qwen4exp.ssm_state_size == 128);
        CHECK(g.qwen4exp.ssm_conv_channels == 10240);

        std::cout << "[Test 9 Passed] ModelGeometry layout synchronized perfectly with descriptor\n";
    }

    std::cout << "\nAll 9 Memory Planner tests PASSED successfully!\n";
    return 0;
}
