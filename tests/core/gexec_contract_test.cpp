// tests/core/gexec_contract_test.cpp - Phase 4 GEXEC contract, invalidation rules,
// expert gate/up row interleaving, autotune lock table, and heterogeneous placement tests.
#include "guild/gexec/gexec_contract.hpp"

#include <chrono>
#include <cstring>
#include <iostream>
#include <vector>

using namespace guild::gexec;

namespace {

int g_passed = 0;
int g_failed = 0;

void check(bool cond, const char* msg, int line) {
    if (cond) {
        ++g_passed;
    } else {
        ++g_failed;
        std::cerr << "[FAIL] line " << line << ": " << msg << "\n";
    }
}
#define CHECK(c) check((c), #c, __LINE__)

GExecArtifactIdentity make_base_identity() {
    GExecArtifactIdentity id;
    id.format_version = kGExecVersionV1;
    id.gpack_manifest_sha256 = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    id.core_abi_version = 1;
    id.module_abi_version = 1;
    id.runner_id = "guild.runner.qwen3_5_moe.v1";
    id.runner_version = 100;
    id.bound_packages = {
        {"guild.kernel.cpu.native", "1.0.0", "aaaa1111"},
        {"guild.kernel.cuda.moe", "1.0.0", "bbbb2222"}
    };
    id.hardware = GExecHardwareFingerprint::detect();
    id.kv_format = "fp16";
    id.kv_mode = "host_only";
    id.max_context = 32768;
    id.prefill_batch = 64;
    id.mtp_spec_tokens = 0;
    return id;
}

void test_identity_and_invalidation() {
    const auto base = make_base_identity();
    const std::string key1 = base.compute_cache_key();
    CHECK(key1.size() == 64);

    // Order of bound_packages must not change cache key
    auto reordered = base;
    std::swap(reordered.bound_packages[0], reordered.bound_packages[1]);
    CHECK(reordered.compute_cache_key() == key1);

    // JSON round-trip
    GExecArtifactIdentity parsed;
    std::string err;
    CHECK(GExecArtifactIdentity::from_json(base.to_json(), parsed, err));
    CHECK(parsed.compute_cache_key() == key1);

    std::string reason;
    CHECK(GExecArtifactIdentity::validate(parsed, base, reason) == GExecInvalidationReason::None);

    // 1. Format version mismatch
    {
        auto m = base;
        m.format_version = 2;
        CHECK(GExecArtifactIdentity::validate(m, base, reason) == GExecInvalidationReason::MagicOrVersionMismatch);
    }
    // 2. GPACK manifest SHA-256 mismatch
    {
        auto m = base;
        m.gpack_manifest_sha256[0] = 'f';
        CHECK(GExecArtifactIdentity::validate(m, base, reason) == GExecInvalidationReason::GPackIdentityMismatch);
    }
    // 3. Core ABI mismatch
    {
        auto m = base;
        m.core_abi_version = 2;
        CHECK(GExecArtifactIdentity::validate(m, base, reason) == GExecInvalidationReason::CoreAbiMismatch);
    }
    // 4. Runner identity/version mismatch
    {
        auto m = base;
        m.runner_version = 101;
        CHECK(GExecArtifactIdentity::validate(m, base, reason) == GExecInvalidationReason::RunnerIdentityMismatch);
    }
    // 5. Bound kernel/module digest mismatch
    {
        auto m = base;
        m.bound_packages[0].payload_sha256 = "cccc3333";
        CHECK(GExecArtifactIdentity::validate(m, base, reason) == GExecInvalidationReason::ModuleOrKernelDigestMismatch);
    }
    // 6. CPU ISA mismatch
    {
        auto m = base;
        m.hardware.cpu_feature_flags ^= 1u;
        CHECK(GExecArtifactIdentity::validate(m, base, reason) == GExecInvalidationReason::CpuIsaMismatch);
    }
    // 7. NUMA topology mismatch
    {
        auto m = base;
        m.hardware.numa_nodes += 1;
        CHECK(GExecArtifactIdentity::validate(m, base, reason) == GExecInvalidationReason::NumaTopologyMismatch);
    }
    // 8. GPU architecture mismatch
    {
        auto m = base;
        m.hardware.gpu_cc_major += 1;
        CHECK(GExecArtifactIdentity::validate(m, base, reason) == GExecInvalidationReason::GpuArchitectureMismatch);
    }
    // 9. CUDA runtime/driver mismatch
    {
        auto m = base;
        m.hardware.cuda_runtime_version += 10;
        CHECK(GExecArtifactIdentity::validate(m, base, reason) == GExecInvalidationReason::CudaRuntimeOrDriverMismatch);
    }
    // 10. Execution plan mismatch
    {
        auto m = base;
        m.prefill_batch = 128;
        CHECK(GExecArtifactIdentity::validate(m, base, reason) == GExecInvalidationReason::ExecutionPlanMismatch);
    }
}

void test_expert_gate_up_interleaving() {
    // Simulate an Ornith IQ3_XXS / Q2_K expert blob: n_ff = 512, gate_row = 784 B, up_row = 784 B, down = 430080 B
    const size_t n_ff = 512;
    const size_t gate_row = 784;
    const size_t up_row = 784;
    const size_t down_bytes = 430080;
    const size_t total_bytes = n_ff * (gate_row + up_row) + down_bytes;

    std::vector<uint8_t> orig(total_bytes);
    for (size_t i = 0; i < total_bytes; ++i) {
        orig[i] = static_cast<uint8_t>((i * 31 + 7) & 0xFF);
    }

    std::vector<uint8_t> packed(total_bytes, 0);
    std::vector<uint8_t> unpacked(total_bytes, 0);

    const auto t0 = std::chrono::steady_clock::now();
    pack_expert_gate_up_interleaved(orig.data(), packed.data(), n_ff, gate_row, up_row, down_bytes);
    const double pack_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();

    unpack_expert_gate_up_interleaved(packed.data(), unpacked.data(), n_ff, gate_row, up_row, down_bytes);
    CHECK(orig == unpacked);

    // Verify row 0 and row 1 layout in packed blob: [gate_0 | up_0 | gate_1 | up_1 ...]
    CHECK(std::memcmp(packed.data(), orig.data(), gate_row) == 0);
    CHECK(std::memcmp(packed.data() + gate_row, orig.data() + n_ff * gate_row, up_row) == 0);
    CHECK(std::memcmp(packed.data() + gate_row + up_row, orig.data() + gate_row, gate_row) == 0);

    std::cout << "[GEXEC] Expert gate/up row interleave (" << (total_bytes >> 10) << " KiB blob): "
              << pack_us << " us (100% lossless round-trip)\n";
}

void test_autotune_and_heterogeneous_plan() {
    GExecAutotuneTable table;
    table.identity = make_base_identity();
    GExecKernelTuneRecord rec;
    rec.op_kind = 10;
    rec.operation_name = "moe_expert_gemv";
    rec.phase = GExecWorkloadPhase::Decode;
    rec.backend = 1;
    rec.quant_type = 18;
    rec.m_tokens = 1;
    rec.n_cols = 512;
    rec.k_cols = 2048;
    rec.selected_impl_id = "guild.cpu.avx2.iq3_xxs";
    rec.selected_package_id = "builtin";
    rec.fallback_impl_id = "guild.cpu.scalar";
    rec.median_latency_us = 41.2;
    rec.candidate_count = 2;
    table.records.push_back(rec);

    const std::string tjson = table.to_json();
    GExecAutotuneTable loaded_table;
    std::string err;
    CHECK(GExecAutotuneTable::from_json(tjson, loaded_table, err));
    const auto* found = loaded_table.find(10, "moe_expert_gemv", GExecWorkloadPhase::Decode, 18, 1);
    CHECK(found != nullptr);
    CHECK(found->selected_impl_id == "guild.cpu.avx2.iq3_xxs");

    // Tamper with JSON -> digest verification must fail closed
    std::string tampered = tjson;
    const auto pos = tampered.find("41.2");
    if (pos != std::string::npos) tampered[pos] = '9';
    GExecAutotuneTable bad_table;
    CHECK(!GExecAutotuneTable::from_json(tampered, bad_table, err));

    // Heterogeneous plan round-trip + tamper detection
    GExecHeterogeneousPlan hplan;
    hplan.identity = make_base_identity();
    hplan.numa_policy = "interleave";
    hplan.use_thp_hugepages = true;
    hplan.measured_formats = 2;
    hplan.calibrated_cpu_ms = 0.85;
    hplan.calibrated_gpu_ms = 0.42;
    hplan.calibrated_copy_ms = 1.90;
    hplan.gpu_layers = {1, 1, 1, 1};
    hplan.pcie_numerators = {0, 0, 0, 0};
    hplan.expert_cache_slots = 2496;

    const std::string hjson = hplan.to_json();
    GExecHeterogeneousPlan loaded_hplan;
    CHECK(GExecHeterogeneousPlan::from_json(hjson, loaded_hplan, err));
    CHECK(loaded_hplan.expert_cache_slots == 2496);
    CHECK(loaded_hplan.numa_policy == "interleave");

    std::string bad_hjson = hjson;
    const auto hpos = bad_hjson.find("2496");
    if (hpos != std::string::npos) bad_hjson[hpos] = '1';
    CHECK(!GExecHeterogeneousPlan::from_json(bad_hjson, loaded_hplan, err));

    // Compilation key determinism
    GExecCompilationKey ck;
    ck.cuda_runtime_version = 12080;
    ck.cuda_driver_version = 13000;
    ck.gpu_cc_major = 6;
    ck.gpu_cc_minor = 1;
    ck.kernel_package_sha256 = "deadbeef";
    CHECK(ck.compute_key().size() == 64);
    CHECK(ck.compute_key() == ck.compute_key());
}

} // namespace

int main() {
    test_identity_and_invalidation();
    test_expert_gate_up_interleaving();
    test_autotune_and_heterogeneous_plan();
    std::cout << "gexec_contract_test: " << g_passed << " passed, " << g_failed << " failed\n";
    return g_failed == 0 ? 0 : 1;
}
