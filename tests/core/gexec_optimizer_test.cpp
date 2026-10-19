// Copyright (c) Guild Inference Contributors. MIT License.
// Phase 5: Comprehensive GEXEC v1 Binary Container, Expert Kernel Parity,
// Single-Barrier Pool Pipeline, Autotuner, Checkpoint/Resume, Corruption, and Concurrency Tests.

#include "guild/artifact/gpack.hpp"
#include "guild/gexec/gexec_contract.hpp"
#include "guild/gexec/gexec_optimizer.hpp"
#include "guild/kernels/cpu/gexec_expert.hpp"
#include "guild/kernels/cpu/pool.hpp"
#include "guild/kernels/registry.hpp"
#include "guild/models/sha256.hpp"
#include "guild/server/json.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <thread>
#include <tuple>
#include <vector>

#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;
using namespace guild;
using namespace guild::gexec;

namespace {

int g_passed = 0;
int g_failed = 0;

void check(bool cond, const char* msg, int line, const std::string& extra = "") {
    if (cond) {
        ++g_passed;
    } else {
        ++g_failed;
        std::cerr << "[FAIL] line " << line << ": " << msg;
        if (!extra.empty())
            std::cerr << " (" << extra << ")";
        std::cerr << "\n";
    }
}
#define CHECK(c) check((c), #c, __LINE__)
#define CHECK_MSG(c, m) check((c), #c, __LINE__, (m))

void fill_synthetic_blob(const guild_cpu_expert_format_v1_t& fmt, int seed, std::vector<uint8_t>& blob) {
    blob.assign(static_cast<size_t>(fmt.bytes), 0);
    auto fill_rows = [&](uint8_t* base, int type, int64_t rows, int64_t cols, uint64_t row_bytes, int role_seed) {
        const int qk = (type == 7 || type == 8 || type == 20) ? 32 : 256;
        const size_t nblocks = static_cast<size_t>(cols / qk);
        const size_t bbytes = static_cast<size_t>(row_bytes) / nblocks;
        for (int64_t r = 0; r < rows; ++r) {
            uint8_t* rptr = base + static_cast<size_t>(r) * static_cast<size_t>(row_bytes);
            for (size_t b = 0; b < nblocks; ++b) {
                uint8_t* blk = rptr + b * bbytes;
                const uint16_t d_fp16 = static_cast<uint16_t>(0x1800u + ((r * 173 + b * 937 + role_seed * 67) & 0xfff));
                const uint16_t m_fp16 = static_cast<uint16_t>(0x1000u + ((r * 313 + b * 271) & 0xfff));
                uint32_t st = static_cast<uint32_t>(seed * 131 + role_seed * 17 + r * 31 + b * 7 + 1);
                for (size_t i = 0; i < bbytes; ++i) {
                    st = st * 1664525u + 1013904223u;
                    blk[i] = static_cast<uint8_t>(st >> 16);
                }
                if (type == 12 || type == 7) {
                    std::memcpy(blk + 0, &d_fp16, 2);
                    std::memcpy(blk + 2, &m_fp16, 2);
                } else if (type == 14) {
                    std::memcpy(blk + 208, &d_fp16, 2);
                } else {
                    std::memcpy(blk + 0, &d_fp16, 2);
                }
            }
        }
    };
    fill_rows(blob.data(), fmt.gu_type, fmt.n_ff, fmt.n_embd, fmt.gu_row, 1);
    fill_rows(blob.data() + fmt.up_off, fmt.gu_type, fmt.n_ff, fmt.n_embd, fmt.gu_row, 2);
    fill_rows(blob.data() + fmt.down_off, fmt.d_type, fmt.n_embd, fmt.n_ff, fmt.d_row, 3);
}

// 1. Test all 11 quantized formats across all 4 GEXEC CPU expert providers and nt = 1..4
void test_all_gexec_cpu_expert_kernels(int64_t n_embd, int64_t n_ff) {
    const auto caps = kernels::query_host_device_capabilities();
    const auto ref_bind = kernels::KernelRegistry::instance().find_by_id("cpu.expert_pool.split_multi_native");
    CHECK(static_cast<bool>(ref_bind));
    if (!ref_bind || !ref_bind->next_ext)
        return;
    const auto* ref_tbl = static_cast<const guild_cpu_backend_ext_v2_t*>(ref_bind->next_ext);

    struct ProviderEntry {
        const char* impl_id;
        const guild_cpu_backend_ext_v2_t* tbl;
        bool interleaved;
        bool needs_avx512;
    };
    const std::vector<ProviderEntry> providers = {
        {kernels::cpu::kGExecFusedAvx2ImplId, kernels::cpu::guild_cpu_gexec_fused_avx2_table_v2(), false, false},
        {kernels::cpu::kGExecInterleavedAvx2ImplId, kernels::cpu::guild_cpu_gexec_interleaved_avx2_table_v2(), true,
         false},
        {kernels::cpu::kGExecFusedAvx512ImplId, kernels::cpu::guild_cpu_gexec_fused_avx512_table_v2(), false, true},
        {kernels::cpu::kGExecInterleavedAvx512ImplId, kernels::cpu::guild_cpu_gexec_interleaved_avx512_table_v2(), true,
         true},
    };

    const uint32_t avx512_mask = GUILD_ISA_AVX512F | GUILD_ISA_AVX512BW | GUILD_ISA_AVX512DQ | GUILD_ISA_AVX512VL;
    const bool host_has_avx512 = ((caps.cpu_isa_flags & avx512_mask) == avx512_mask);

    // Every (gu_type, d_type) pair used across Ornith-1.5-35B and Qwen3.8-Flash-Next
    const std::vector<std::pair<int32_t, int32_t>> type_pairs = {
        {12, 12}, // Q4_K + Q4_K
        {12, 14}, // Q4_K + Q6_K
        {12, 20}, // Q4_K + IQ4_NL (Ornith-1.5-35B primary format)
        {14, 14}, // Q6_K + Q6_K
        {8, 8},   // Q8_0 + Q8_0
        {7, 7},   // Q5_1 + Q5_1
        {23, 20}, // IQ4_XS + IQ4_NL (Qwen3.8-Flash-Next primary format)
        {21, 20}, // IQ3_S + IQ4_NL
        {16, 20}, // IQ2_XXS + IQ4_NL
        {17, 20}, // IQ2_XS + IQ4_NL
        {18, 20}, // IQ3_XXS + IQ4_NL
        {22, 20}, // IQ2_S + IQ4_NL
    };

    constexpr int max_tokens = 8;
    std::vector<float> x_in(static_cast<size_t>(max_tokens * n_embd));
    for (size_t i = 0; i < x_in.size(); ++i) {
        x_in[i] = std::sin(static_cast<float>(i) * 0.07f) * 0.75f;
    }

    for (const auto& [gu_type, d_type] : type_pairs) {
        guild_cpu_expert_format_v1_t fmt{};
        guild_error_info_t einfo{};
        CHECK(ref_tbl->format(gu_type, d_type, n_embd, n_ff, &fmt, &einfo) == GUILD_STATUS_OK);

        std::vector<uint8_t> blob_contig;
        fill_synthetic_blob(fmt, gu_type * 37 + d_type, blob_contig);
        std::vector<uint8_t> blob_inter(blob_contig.size());
        pack_expert_gate_up_interleaved(blob_contig.data(), blob_inter.data(), static_cast<size_t>(fmt.n_ff),
                                        static_cast<size_t>(fmt.gu_row), static_cast<size_t>(fmt.gu_row),
                                        static_cast<size_t>(fmt.bytes - fmt.down_off));

        alignas(64) uint8_t qact[max_tokens][kernels::cpu::kNativeActBytes] = {};
        const void* act_ptrs[max_tokens] = {};
        for (int t = 0; t < max_tokens; ++t) {
            act_ptrs[t] = qact[t];
            ref_tbl->native.quant_act(&fmt, x_in.data() + t * n_embd, qact[t]);
        }

        for (int nt = 1; nt <= max_tokens; ++nt) {
            std::vector<std::vector<float>> ref_ff(static_cast<size_t>(nt),
                                                   std::vector<float>(static_cast<size_t>(n_ff), 0.0f));
            float* ref_ff_ptrs[max_tokens] = {};
            for (int t = 0; t < nt; ++t)
                ref_ff_ptrs[t] = ref_ff[static_cast<size_t>(t)].data();
            ref_tbl->native.gate_up(&fmt, blob_contig.data(), act_ptrs, nt, ref_ff_ptrs, 0, static_cast<int>(n_ff));

            alignas(64) uint8_t ref_hq[max_tokens][kernels::cpu::kNativeHBytes] = {};
            const void* ref_hq_ptrs[max_tokens] = {};
            for (int t = 0; t < nt; ++t) {
                ref_hq_ptrs[t] = ref_hq[t];
                ref_tbl->native.quant_h(&fmt, ref_ff[static_cast<size_t>(t)].data(), ref_hq[t]);
            }

            std::vector<std::vector<float>> ref_out(static_cast<size_t>(nt),
                                                    std::vector<float>(static_cast<size_t>(n_embd), 0.0f));
            float* ref_out_ptrs[max_tokens] = {};
            for (int t = 0; t < nt; ++t)
                ref_out_ptrs[t] = ref_out[static_cast<size_t>(t)].data();
            ref_tbl->native.down(&fmt, blob_contig.data(), ref_hq_ptrs, nt, ref_out_ptrs, 0, static_cast<int>(n_embd));

            for (const auto& prov : providers) {
                if (prov.needs_avx512 && !host_has_avx512)
                    continue;
                const uint8_t* wblob = prov.interleaved ? blob_inter.data() : blob_contig.data();

                std::vector<std::vector<float>> cand_ff(static_cast<size_t>(nt),
                                                        std::vector<float>(static_cast<size_t>(n_ff), 0.0f));
                float* cand_ff_ptrs[max_tokens] = {};
                for (int t = 0; t < nt; ++t)
                    cand_ff_ptrs[t] = cand_ff[static_cast<size_t>(t)].data();
                for (const auto& [a, z] : {std::pair<int, int>{0, 1}, {1, int(n_ff) - 1}, {int(n_ff) - 1, int(n_ff)}})
                    prov.tbl->native.gate_up(&fmt, wblob, act_ptrs, nt, cand_ff_ptrs, a, z);

                alignas(64) uint8_t cand_hq[max_tokens][kernels::cpu::kNativeHBytes] = {};
                bool quant_exact = true;
                for (int t = 0; t < nt; ++t) {
                    prov.tbl->native.quant_h(&fmt, cand_ff[static_cast<size_t>(t)].data(), cand_hq[t]);
                    quant_exact = quant_exact && std::memcmp(cand_hq[t], ref_hq[t], fmt.h_bytes) == 0;
                }

                std::vector<std::vector<float>> cand_out(static_cast<size_t>(nt),
                                                         std::vector<float>(static_cast<size_t>(n_embd), 0.0f));
                float* cand_out_ptrs[max_tokens] = {};
                for (int t = 0; t < nt; ++t)
                    cand_out_ptrs[t] = cand_out[static_cast<size_t>(t)].data();
                for (const auto& [a, z] :
                     {std::pair<int, int>{0, 1}, {1, int(n_embd) - 1}, {int(n_embd) - 1, int(n_embd)}})
                    prov.tbl->native.down(&fmt, wblob, ref_hq_ptrs, nt, cand_out_ptrs, a, z);

                // Verify bitwise parity on gate_up and down!
                bool gu_exact = true;
                bool down_exact = true;
                for (int t = 0; t < nt; ++t) {
                    if (std::memcmp(cand_ff[static_cast<size_t>(t)].data(), ref_ff[static_cast<size_t>(t)].data(),
                                    static_cast<size_t>(n_ff) * sizeof(float)) != 0) {
                        gu_exact = false;
                    }
                    if (std::memcmp(cand_out[static_cast<size_t>(t)].data(), ref_out[static_cast<size_t>(t)].data(),
                                    static_cast<size_t>(n_embd) * sizeof(float)) != 0) {
                        down_exact = false;
                    }
                }
                const std::string tag = std::string(prov.impl_id) + " gu=" + std::to_string(gu_type) +
                                        " d=" + std::to_string(d_type) + " nt=" + std::to_string(nt);
                CHECK_MSG(gu_exact, "gate_up bitwise parity: " + tag);
                CHECK_MSG(quant_exact, "quant_h bitwise parity: " + tag);
                CHECK_MSG(down_exact, "down bitwise parity: " + tag);
            }
        }
    }

    // Also test AVX2 vectorized combine + non-finite detection
    {
        const int k = 8, n = 259; // non-multiple of 8 tail coverage
        std::vector<float> parts(static_cast<size_t>(k * n));
        std::vector<float> weights(static_cast<size_t>(k));
        for (int r = 0; r < k; ++r)
            weights[static_cast<size_t>(r)] = 0.1f * static_cast<float>(r + 1);
        for (size_t i = 0; i < parts.size(); ++i)
            parts[i] = std::cos(static_cast<float>(i) * 0.05f);

        std::vector<float> out_ref(static_cast<size_t>(n), 0.0f);
        std::vector<float> out_gexec(static_cast<size_t>(n), 0.0f);
        const auto* gtbl = kernels::cpu::guild_cpu_gexec_fused_avx2_table_v2();
        CHECK(ref_tbl->combine(parts.data(), weights.data(), out_ref.data(), k, n) == GUILD_STATUS_OK);
        CHECK(gtbl->combine(parts.data(), weights.data(), out_gexec.data(), k, n) == GUILD_STATUS_OK);
        double max_diff = 0.0;
        for (int d = 0; d < n; ++d) {
            max_diff = std::max(max_diff, static_cast<double>(std::fabs(out_ref[static_cast<size_t>(d)] -
                                                                        out_gexec[static_cast<size_t>(d)])));
        }
        CHECK(max_diff < 1e-6);

        // Inject NaN -> must return GUILD_STATUS_ERR_NONFINITE_OUTPUT
        parts[13] = std::numeric_limits<float>::quiet_NaN();
        CHECK(gtbl->combine(parts.data(), weights.data(), out_gexec.data(), k, n) == GUILD_STATUS_ERR_NONFINITE_OUTPUT);
    }
}

// 2. Test ExpertPool mode_ == 11 single-barrier fused pipeline + in-worker L1 finiteness check
void test_expert_pool_fused_pipeline() {
    const auto* gtbl = kernels::cpu::guild_cpu_gexec_interleaved_avx2_table_v2();
    const auto* ftbl = kernels::cpu::guild_cpu_gexec_fused_avx2_table_v2();

    guild_cpu_expert_format_v1_t fmt{};
    CHECK(gtbl->format(12, 20, 512, 256, &fmt, nullptr) == GUILD_STATUS_OK);

    const int k = 3; // Exercise sliced execution below the whole-expert threshold (n < threads).
    std::vector<std::vector<uint8_t>> blobs_contig(static_cast<size_t>(k));
    std::vector<std::vector<uint8_t>> blobs_inter(static_cast<size_t>(k));
    for (int e = 0; e < k; ++e) {
        fill_synthetic_blob(fmt, 900 + e, blobs_contig[static_cast<size_t>(e)]);
        blobs_inter[static_cast<size_t>(e)].resize(blobs_contig[static_cast<size_t>(e)].size());
        pack_expert_gate_up_interleaved(blobs_contig[static_cast<size_t>(e)].data(),
                                        blobs_inter[static_cast<size_t>(e)].data(), static_cast<size_t>(fmt.n_ff),
                                        static_cast<size_t>(fmt.gu_row), static_cast<size_t>(fmt.gu_row),
                                        static_cast<size_t>(fmt.bytes - fmt.down_off));
    }

    std::vector<float> x_in(static_cast<size_t>(2 * fmt.n_embd));
    for (size_t i = 0; i < x_in.size(); ++i)
        x_in[i] = std::sin(static_cast<float>(i) * 0.03f);
    alignas(64) uint8_t qact0[kernels::cpu::kNativeActBytes] = {};
    alignas(64) uint8_t qact1[kernels::cpu::kNativeActBytes] = {};
    ftbl->native.quant_act(&fmt, x_in.data(), qact0);
    ftbl->native.quant_act(&fmt, x_in.data() + fmt.n_embd, qact1);

    kernels::cpu::ExpertPool pool(/*n_workers=*/4, /*pin=*/false);

    std::vector<std::vector<float>> out_base0(static_cast<size_t>(k),
                                              std::vector<float>(static_cast<size_t>(fmt.n_embd), 0.0f));
    std::vector<std::vector<float>> out_base1(static_cast<size_t>(k),
                                              std::vector<float>(static_cast<size_t>(fmt.n_embd), 0.0f));
    std::vector<std::vector<float>> out_fused0(static_cast<size_t>(k),
                                               std::vector<float>(static_cast<size_t>(fmt.n_embd), 0.0f));
    std::vector<std::vector<float>> out_fused1(static_cast<size_t>(k),
                                               std::vector<float>(static_cast<size_t>(fmt.n_embd), 0.0f));

    std::vector<kernels::cpu::ExpertJobMulti> jobs_base(static_cast<size_t>(k));
    std::vector<kernels::cpu::ExpertJobMulti> jobs_fused(static_cast<size_t>(k));
    for (int e = 0; e < k; ++e) {
        jobs_base[static_cast<size_t>(e)].blob = blobs_contig[static_cast<size_t>(e)].data();
        jobs_base[static_cast<size_t>(e)].nt = 2;
        jobs_base[static_cast<size_t>(e)].nact[0] = qact0;
        jobs_base[static_cast<size_t>(e)].nact[1] = qact1;
        jobs_base[static_cast<size_t>(e)].out[0] = out_base0[static_cast<size_t>(e)].data();
        jobs_base[static_cast<size_t>(e)].out[1] = out_base1[static_cast<size_t>(e)].data();

        jobs_fused[static_cast<size_t>(e)].blob = blobs_inter[static_cast<size_t>(e)].data();
        jobs_fused[static_cast<size_t>(e)].nt = 2;
        jobs_fused[static_cast<size_t>(e)].nact[0] = qact0;
        jobs_fused[static_cast<size_t>(e)].nact[1] = qact1;
        jobs_fused[static_cast<size_t>(e)].out[0] = out_fused0[static_cast<size_t>(e)].data();
        jobs_fused[static_cast<size_t>(e)].out[1] = out_fused1[static_cast<size_t>(e)].data();
    }

    // Run 2-barrier mode_ == 7
    pool.bind_backend_kernels(*ftbl);
    pool.set_fused_pipeline(false);
    pool.run_split_multi_native(fmt, jobs_base.data(), k);
    CHECK(!pool.last_run_all_finite()); // mode_ == 7 does not set last_run_verified_finite_

    // Run 1-barrier fused mode_ == 11
    pool.bind_backend_kernels(*gtbl);
    pool.set_fused_pipeline(true);
    pool.run_split_multi_native(fmt, jobs_fused.data(), k);
    CHECK(pool.last_run_all_finite());

    for (int e = 0; e < k; ++e) {
        CHECK(std::memcmp(out_base0[static_cast<size_t>(e)].data(), out_fused0[static_cast<size_t>(e)].data(),
                          static_cast<size_t>(fmt.n_embd) * sizeof(float)) == 0);
        CHECK(std::memcmp(out_base1[static_cast<size_t>(e)].data(), out_fused1[static_cast<size_t>(e)].data(),
                          static_cast<size_t>(fmt.n_embd) * sizeof(float)) == 0);
    }
}

// 3. Test GExecBuilder / GExecReader binary container, alignment, section SHA-256, and corruption detection
void test_gexec_binary_container_and_corruption(const fs::path& work_dir) {
    GExecArtifactIdentity id;
    id.format_version = kGExecVersionV1;
    id.gpack_manifest_sha256 = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    id.core_abi_version = GUILD_KERNEL_ABI_VERSION_V1;
    id.module_abi_version = 1;
    id.runner_id = "guild.runner.qwen3_5_moe.v1";
    id.runner_version = 100;
    id.bound_packages = {{"builtin.guild-kernels-cpu", "5.0.0", "aaaa1111"}};
    id.hardware = GExecHardwareFingerprint::detect();
    id.kv_format = "fp16";
    id.kv_mode = "host_only";
    id.max_context = 4096;
    id.prefill_batch = 64;
    id.mtp_spec_tokens = 0;

    GExecAutotuneTable at;
    at.identity = id;
    GExecKernelTuneRecord rec;
    rec.op_kind = GUILD_OP_BATCHED_EXPERT_EXEC;
    rec.operation_name = "batched_expert_exec";
    rec.phase = GExecWorkloadPhase::Decode;
    rec.backend = GUILD_BACKEND_CPU;
    rec.quant_type = 12;
    rec.m_tokens = 1;
    rec.n_cols = 512;
    rec.k_cols = 2048;
    rec.selected_impl_id = kernels::cpu::kGExecInterleavedAvx512ImplId;
    rec.selected_package_id = "builtin.guild-kernels-cpu";
    rec.selected_impl_version = kernels::cpu::kGExecExpertImplVersion;
    rec.fallback_impl_id = "cpu.expert_pool.split_multi_native";
    rec.median_latency_us = 41.2;
    rec.candidate_count = 5;
    at.records.push_back(rec);

    GExecHeterogeneousPlan hp;
    hp.identity = id;
    hp.numa_policy = "interleave";
    hp.use_thp_hugepages = true;
    hp.measured_formats = 1;
    hp.calibrated_cpu_ms = 12.5;
    hp.gpu_layers = {1, 1};
    hp.pcie_numerators = {0, 0};

    GExecWorkspaceRecipe wr;
    wr.dense_arena_bytes = 128ull << 20;
    wr.native_dense_arena_bytes = 64ull << 20;
    wr.session_device_bytes = 32ull << 20;
    wr.session_host_kv_bytes = 256ull << 20;
    wr.verifier_arena_bytes = 16ull << 20;
    wr.prefill_arena_bytes = 64ull << 20;
    wr.capture_window_tokens = {1, 2};
    wr.capture_commit_graph = true;

    GExecCompilationProvenance prov;
    prov.key.cuda_runtime_version = 12080;
    prov.key.cuda_driver_version = 12080;
    prov.key.gpu_cc_major = 6;
    prov.key.gpu_cc_minor = 1;
    prov.key.kernel_package_sha256 = "aaaa1111";
    prov.compiler_id = "guild-gexec-v1";
    prov.guild_version = "0.5.0";
    prov.git_commit = "phase5";
    prov.created_unix_seconds = 1700000000;
    prov.optimization_mode = "auto";
    prov.optimization_wall_ms = 185.0;

    GExecExpertPackMeta pmeta;
    pmeta.layout_kind = ExpertPackLayoutKind::GateUpRowInterleaved;
    pmeta.embedded_blob_present = true;
    pmeta.fused_single_barrier_pipeline = true;
    pmeta.n_layers = 2;
    pmeta.n_expert = 4;
    pmeta.total_expert_bytes = 8192;

    GExecDiagnosticLog dlog;
    dlog.baseline_decode_ms = 55.0;
    dlog.optimized_decode_ms = 39.5;
    dlog.speedup_ratio = 55.0 / 39.5;
    dlog.break_even_tokens = 12.0;
    dlog.notes = {"unit test diagnostic note"};
    dlog.rejected_candidates.push_back({GUILD_OP_BATCHED_EXPERT_EXEC, "batched_expert_exec",
                                        "cpu.expert_pool.split_multi_native", "microbenchmark", "slower than GEXEC",
                                        0.0, 62.0});

    std::vector<uint8_t> fake_experts(8192, 0x5Au);

    const fs::path gexec_path = work_dir / "container_test.gexec";
    GExecBuilder builder;
    builder.set_identity(id);
    builder.set_autotune_table(at);
    builder.set_heterogeneous_plan(hp);
    builder.set_workspace_recipe(wr);
    builder.set_provenance(prov);
    builder.set_expert_pack_meta(pmeta);
    builder.set_diagnostic_log(dlog);
    // A blob replacement must clear the old streaming callback and byte count.
    builder.set_packed_experts_stream(3, [](uint64_t, uint8_t*, size_t, std::string&) { return false; });
    builder.set_packed_experts_blob(fake_experts, 4096);

    std::string err;
    CHECK_MSG(builder.write_file(gexec_path.string(), err), err);

    GExecReader reader;
    GExecInvalidationReason reason = GExecInvalidationReason::None;
    CHECK_MSG(reader.open(gexec_path.string(), err, &reason, /*verify_expert_blob_digest=*/true), err);
    CHECK(reason == GExecInvalidationReason::None);
    CHECK(reader.is_open());
    CHECK(!reader.is_partial());
    CHECK(reader.sections().size() == 8);
    CHECK(reader.validate_against(id, err, &reason));
    CHECK(reader.verify_all_digests(err));

    // Verify 4096-byte page alignment of PackedExpertsBlob
    const auto* blob_sec = reader.find_section(GExecSectionKind::PackedExpertsBlob);
    CHECK(blob_sec != nullptr);
    if (blob_sec) {
        CHECK((blob_sec->offset % 4096u) == 0);
        CHECK(blob_sec->byte_size == 8192u);
    }

    // Read raw file bytes to test corruption detection on every section and header field
    std::ifstream raw_ifs(gexec_path, std::ios::binary);
    std::vector<uint8_t> raw((std::istreambuf_iterator<char>(raw_ifs)), std::istreambuf_iterator<char>());
    CHECK(raw.size() == reader.header().total_file_bytes);

    auto write_corrupt = [&](const std::string& name, const std::vector<uint8_t>& data) -> std::string {
        const fs::path cp = work_dir / name;
        std::ofstream ofs(cp, std::ios::binary | std::ios::trunc);
        ofs.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
        return cp.string();
    };

    // 3a. Bad magic
    {
        auto c = raw;
        c[0] ^= 0xFF;
        GExecReader r;
        CHECK(!r.open(write_corrupt("bad_magic.gexec", c), err, &reason));
        CHECK(reason == GExecInvalidationReason::MagicOrVersionMismatch);
    }
    // 3b. Bad version
    {
        auto c = raw;
        c[4] = 99;
        GExecReader r;
        CHECK(!r.open(write_corrupt("bad_ver.gexec", c), err, &reason));
        CHECK(reason == GExecInvalidationReason::MagicOrVersionMismatch);
    }
    // 3c. Bad endianness tag
    {
        auto c = raw;
        std::swap(c[6], c[7]);
        GExecReader r;
        CHECK(!r.open(write_corrupt("bad_endian.gexec", c), err, &reason));
        CHECK(reason == GExecInvalidationReason::MagicOrVersionMismatch);
    }
    // 3d. Truncated header (< 64 bytes)
    {
        std::vector<uint8_t> c(raw.begin(), raw.begin() + 30);
        GExecReader r;
        CHECK(!r.open(write_corrupt("trunc_hdr.gexec", c), err, &reason));
        CHECK(reason == GExecInvalidationReason::CorruptedDigest);
    }
    // 3e. Truncated file payload
    {
        std::vector<uint8_t> c(raw.begin(), raw.begin() + raw.size() - 17);
        GExecReader r;
        CHECK(!r.open(write_corrupt("trunc_body.gexec", c), err, &reason));
        CHECK(reason == GExecInvalidationReason::CorruptedDigest);
    }
    // 3f. Corrupted section table entry
    {
        auto c = raw;
        c[64 + 8] ^= 0x01; // flip bit in section 0 offset
        GExecReader r;
        CHECK(!r.open(write_corrupt("bad_sectbl.gexec", c), err, &reason));
        CHECK(reason == GExecInvalidationReason::CorruptedDigest);
    }
    // 3g. Flip a byte inside EVERY single section payload and verify SHA-256 catches it!
    for (const auto& sec : reader.sections()) {
        if (sec.byte_size == 0)
            continue;
        auto c = raw;
        c[static_cast<size_t>(sec.offset + sec.byte_size / 2)] ^= 0x40;
        GExecReader r;
        const std::string cname =
            std::string("corrupt_sec_") + section_kind_to_string(static_cast<GExecSectionKind>(sec.kind)) + ".gexec";
        CHECK(!r.open(write_corrupt(cname, c), err, &reason, /*verify_expert_blob_digest=*/true));
        CHECK(reason == GExecInvalidationReason::CorruptedDigest);
    }
}

// Helper to create a minimal valid safetensors model and convert it to a .gpack container
std::string create_test_gpack(const fs::path& work_dir) {
    const fs::path st_dir = work_dir / "st_model";
    fs::create_directories(st_dir);
    {
        std::ofstream cfg(st_dir / "config.json");
        cfg << R"({
  "architecture": "qwen3_5_moe",
  "hidden_size": 16,
  "num_hidden_layers": 2,
  "num_attention_heads": 2,
  "num_key_value_heads": 2,
  "head_dim": 8,
  "vocab_size": 32,
  "max_position_embeddings": 512,
  "num_experts": 4,
  "num_experts_per_tok": 2,
  "moe_intermediate_size": 8,
  "full_attention_interval": 2
})";
    }
    std::vector<std::tuple<std::string, std::string, std::vector<int64_t>, std::vector<uint8_t>>> tensors;
    auto add_t = [&](const std::string& name, const std::string& dt, std::vector<int64_t> sh, size_t b) {
        tensors.emplace_back(name, dt, std::move(sh), std::vector<uint8_t>(b, 1));
    };
    add_t("token_embd.weight", "BF16", {32, 16}, 32 * 16 * 2);
    add_t("output_norm.weight", "F32", {16}, 16 * 4);
    add_t("output.weight", "BF16", {32, 16}, 32 * 16 * 2);
    for (int l = 0; l < 2; ++l) {
        const std::string p = "blk." + std::to_string(l) + ".";
        add_t(p + "attn_norm.weight", "F32", {16}, 16 * 4);
        add_t(p + "ffn_gate_inp.weight", "BF16", {4, 16}, 4 * 16 * 2);
        add_t(p + "ffn_gate_exps.weight", "BF16", {4, 8, 16}, 4 * 8 * 16 * 2);
        add_t(p + "ffn_up_exps.weight", "BF16", {4, 8, 16}, 4 * 8 * 16 * 2);
        add_t(p + "ffn_down_exps.weight", "BF16", {4, 16, 8}, 4 * 16 * 8 * 2);
    }

    guild::server::json::JsonValue hroot;
    uint64_t off = 0;
    for (const auto& [name, dtype, shape, data] : tensors) {
        guild::server::json::JsonValue tdesc;
        tdesc["dtype"] = dtype;
        guild::server::json::JsonValue sh(std::vector<guild::server::json::JsonValue>{});
        for (int64_t d : shape)
            sh.arr_val.push_back(guild::server::json::JsonValue(d));
        tdesc["shape"] = std::move(sh);
        guild::server::json::JsonValue offs(std::vector<guild::server::json::JsonValue>{
            guild::server::json::JsonValue(static_cast<int64_t>(off)),
            guild::server::json::JsonValue(static_cast<int64_t>(off + data.size()))});
        tdesc["data_offsets"] = std::move(offs);
        hroot[name] = std::move(tdesc);
        off += data.size();
    }
    const std::string hstr = hroot.dump();
    const uint64_t hlen = hstr.size();
    {
        std::ofstream out(st_dir / "model.safetensors", std::ios::binary);
        out.write(reinterpret_cast<const char*>(&hlen), 8);
        out.write(hstr.data(), static_cast<std::streamsize>(hstr.size()));
        for (const auto& [name, dtype, shape, data] : tensors) {
            out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
        }
    }

    const fs::path gpack_path = work_dir / "test_model.gpack";
    artifact::GPackBuildOptions bopts;
    bopts.primary_input_path = st_dir.string();
    bopts.output_path = gpack_path.string();
    bopts.model_name = "test-moe";
    artifact::GPackBuildReport brep;
    std::string err;
    const bool built = artifact::GPackBuilder::build(bopts, brep, err);
    CHECK_MSG(built, err);
    if (!built)
        return {};
    return gpack_path.string();
}

// 4. Test optimize_gpack modes, checkpoint/resume, section regeneration, lock contention, concurrency, and cache-clean
void test_optimizer_lifecycle_and_concurrency(const fs::path& work_dir) {
    const std::string gpack_path = create_test_gpack(work_dir);
    if (gpack_path.empty())
        return;
    const fs::path cache_dir = work_dir / ".gexec_cache";

    // 4a. Skip mode
    {
        GExecOptimizeOptions opts;
        opts.gpack_path = gpack_path;
        opts.cache_dir = cache_dir.string();
        opts.mode = GExecOptMode::Skip;
        const auto res = optimize_gpack(opts);
        CHECK(res.ok && res.skipped);
    }

    // 4b. Simulated interruption after pass 2 -> writes partial artifact + .progress.json checkpoint
    const fs::path interrupted_gexec = cache_dir / "interrupted.gexec";
    {
        GExecOptimizeOptions opts;
        opts.gpack_path = gpack_path;
        opts.output_gexec_path = interrupted_gexec.string();
        opts.mode = GExecOptMode::Quick;
        opts.stop_after_pass = 2;
        const auto res = optimize_gpack(opts);
        CHECK_MSG(res.ok, res.error);
        CHECK(res.partial_artifact);
        CHECK(fs::exists(interrupted_gexec));
        CHECK(fs::exists(interrupted_gexec.string() + ".progress.json"));

        GExecReader r;
        std::string err;
        CHECK(r.open(interrupted_gexec.string(), err));
        CHECK(r.is_partial());
    }

    // 4c. Resume interrupted optimization -> loads .progress.json and finishes passes 3..5
    {
        GExecOptimizeOptions opts;
        opts.gpack_path = gpack_path;
        opts.output_gexec_path = interrupted_gexec.string();
        opts.mode = GExecOptMode::Quick;
        opts.stop_after_pass = -1;
        const auto res = optimize_gpack(opts);
        CHECK_MSG(res.ok, res.error);
        CHECK(res.resumed_from_checkpoint);
        CHECK(!res.partial_artifact);
        CHECK(!fs::exists(interrupted_gexec.string() + ".progress.json"));
        CHECK(!res.selected_cpu_expert_impl_id.empty() ||
              res.expert_layout_kind == ExpertPackLayoutKind::PortableContiguous);

        GExecReader r;
        std::string err;
        CHECK(r.open(interrupted_gexec.string(), err));
        CHECK(!r.is_partial());
        CHECK(r.has_section(GExecSectionKind::AutotuneTable));
        CHECK(r.has_section(GExecSectionKind::HeterogeneousPlan));
        CHECK(r.has_section(GExecSectionKind::WorkspaceRecipe));
        CHECK(r.has_section(GExecSectionKind::CompilationProvenance));
        CHECK(r.has_section(GExecSectionKind::DiagnosticLog));
    }

    // 4d. Full Auto optimization into default cache path + resolve_gexec_for_gpack
    std::string auto_gexec_path;
    {
        GExecOptimizeOptions opts;
        opts.gpack_path = gpack_path;
        opts.cache_dir = cache_dir.string();
        opts.mode = GExecOptMode::Auto;
        opts.budget.embed_packed_experts = true;
        const auto res = optimize_gpack(opts);
        CHECK_MSG(res.ok, res.error);
        CHECK(!res.partial_artifact);
        CHECK(res.artifact_bytes > 512);
        auto_gexec_path = res.output_path;

        artifact::GPackReader gpack;
        std::string gerr;
        CHECK(gpack.open(gpack_path, gerr, false));
        const auto cur_id = build_current_gexec_identity(gpack);
        CHECK(build_current_gexec_identity(gpack, "FP16", "host-only").compute_cache_key() ==
              cur_id.compute_cache_key());

        GExecReader resolved;
        std::string fb_reason;
        GExecInvalidationReason rcode = GExecInvalidationReason::None;
        CHECK(resolve_gexec_for_gpack(gpack_path, auto_gexec_path, cur_id, resolved, fb_reason, &rcode));
        CHECK(rcode == GExecInvalidationReason::None);
        CHECK(resolved.has_section(GExecSectionKind::PackedExpertsBlob));

        // Verify invalidation fallback when max_context changes
        auto mismatch_id = cur_id;
        mismatch_id.max_context = 65536;
        GExecReader rejected;
        CHECK(!resolve_gexec_for_gpack(gpack_path, auto_gexec_path, mismatch_id, rejected, fb_reason, &rcode));
        CHECK(rcode == GExecInvalidationReason::ExecutionPlanMismatch);
        CHECK(!fb_reason.empty());
    }

    // 4e. Section-selective regeneration (WorkspaceRecipe and HeterogeneousPlan)
    {
        std::string err;
        CHECK_MSG(regenerate_gexec_section(auto_gexec_path, gpack_path, GExecSectionKind::WorkspaceRecipe, err), err);
        CHECK_MSG(regenerate_gexec_section(auto_gexec_path, gpack_path, GExecSectionKind::HeterogeneousPlan, err), err);
        GExecReader r;
        CHECK(r.open(auto_gexec_path, err, nullptr, true));
        CHECK(r.verify_all_digests(err));

        GExecOptimizeOptions opts;
        opts.gpack_path = gpack_path;
        opts.output_gexec_path = auto_gexec_path;
        opts.budget.only_section = GExecSectionKind::WorkspaceRecipe;
        opts.stop_after_pass = 1; // Selective regeneration must not restart the pipeline.
        opts.budget.max_memory_bytes = 1;
        const auto memory_limited = optimize_gpack(opts);
        CHECK(!memory_limited.ok && memory_limited.error.find("memory budget") != std::string::npos);
        opts.budget.max_memory_bytes = 8ull << 20;
        opts.budget.max_disk_bytes = 1;
        const auto disk_limited = optimize_gpack(opts);
        CHECK(!disk_limited.ok && disk_limited.error.find("disk budget") != std::string::npos);
        opts.budget.max_disk_bytes = 0;
        const auto selective = optimize_gpack(opts);
        CHECK_MSG(selective.ok && !selective.partial_artifact, selective.error);
        GExecReader updated;
        CHECK(updated.open(auto_gexec_path, err, nullptr, true));
        for (const auto& section : r.sections()) {
            if (section.kind == static_cast<uint32_t>(GExecSectionKind::WorkspaceRecipe))
                continue;
            const auto* after = updated.find_section(static_cast<GExecSectionKind>(section.kind));
            CHECK(after && std::memcmp(section.sha256, after->sha256, sizeof(section.sha256)) == 0);
        }
    }

    // 4f. Advisory lock contention test
#if !defined(_WIN32)
    {
        const std::string lock_file = auto_gexec_path + ".lock";
        const int lfd = ::open(lock_file.c_str(), O_RDWR | O_CREAT, 0644);
        CHECK(lfd >= 0);
        CHECK(::flock(lfd, LOCK_EX | LOCK_NB) == 0);

        GExecOptimizeOptions opts;
        opts.gpack_path = gpack_path;
        opts.output_gexec_path = auto_gexec_path;
        opts.mode = GExecOptMode::Quick;
        const auto res = optimize_gpack(opts);
        CHECK(!res.ok);
        CHECK(res.error.find("lock") != std::string::npos);

        (void) ::flock(lfd, LOCK_UN);
        ::close(lfd);
        fs::remove(lock_file);
    }
#endif

    // 4g. Concurrent multi-thread GExecReader while atomic GExecBuilder::write_file updates the file
    {
        GExecReader base_r;
        std::string err;
        CHECK(base_r.open(auto_gexec_path, err));
        GExecBuilder writer_builder;
        CHECK(writer_builder.import_from(base_r, err));
        base_r.close();

        std::atomic<bool> stop{false};
        std::atomic<bool> ok{true};
        std::vector<std::thread> readers;
        for (int t = 0; t < 6; ++t) {
            readers.emplace_back([&]() {
                while (!stop.load(std::memory_order_relaxed)) {
                    GExecReader tr;
                    std::string terr;
                    if (!tr.open(auto_gexec_path, terr, nullptr, true) || !tr.verify_all_digests(terr)) {
                        ok.store(false);
                        break;
                    }
                }
            });
        }
        for (int w = 0; w < 10; ++w) {
            std::string werr;
            if (!writer_builder.write_file(auto_gexec_path, werr)) {
                ok.store(false);
                break;
            }
        }
        stop.store(true);
        for (auto& th : readers)
            th.join();
        CHECK(ok.load());
    }

    // 4h. Cache cleanup test: create a corrupt .gexec and a stray .tmp file, verify clean_gexec_cache removes them
    // while keeping the valid `auto_gexec_path`
    {
        const fs::path corrupt_file = cache_dir / "corrupt_entry.gexec";
        {
            std::ofstream ofs(corrupt_file, std::ios::binary);
            ofs << "NOT_A_VALID_GEXEC_FILE";
        }
        const fs::path stray_tmp = cache_dir / "stray.gexec.tmp.9999";
        {
            std::ofstream ofs(stray_tmp, std::ios::binary);
            ofs << "temporary";
        }

        artifact::GPackReader gpack;
        std::string gerr;
        CHECK(gpack.open(gpack_path, gerr, false));
        const auto cur_id = build_current_gexec_identity(gpack);

        const auto clean_res = clean_gexec_cache(cache_dir.string(), /*remove_all=*/false, &cur_id);
        CHECK(clean_res.ok);
        CHECK(clean_res.removed_files >= 2);
        CHECK(clean_res.kept_files >= 1);
        CHECK(!fs::exists(corrupt_file));
        CHECK(!fs::exists(stray_tmp));
        CHECK(fs::exists(auto_gexec_path));
    }
}

} // namespace

int main() {
    const fs::path work_dir = fs::current_path() / "gexec_optimizer_test_tmp";
    std::error_code ec;
    fs::remove_all(work_dir, ec);
    fs::create_directories(work_dir, ec);

    test_all_gexec_cpu_expert_kernels(512, 256);
    test_all_gexec_cpu_expert_kernels(2048, 512);
    test_expert_pool_fused_pipeline();
    test_gexec_binary_container_and_corruption(work_dir);
    test_optimizer_lifecycle_and_concurrency(work_dir);

    fs::remove_all(work_dir, ec);

    std::cout << "[gexec_optimizer_test] passed=" << g_passed << " failed=" << g_failed << "\n";
    return g_failed == 0 ? 0 : 1;
}
