#include "guild/core/expert_source.hpp"
#include "guild/artifact/gpack.hpp"
#include "guild/gexec/gexec_contract.hpp"
#include "guild/kernels/cpu/expert_layout.hpp"
#include "guild/kernels/cpu/gexec_expert.hpp"
#include "guild/memory/planner.hpp"
#include "guild/model/model_descriptor.hpp"
#include "guild/models/registry.hpp"
#include "guild/models/store.hpp"
#include "guild/runtime/model.hpp"
#include "guild/runtime/session.hpp"
#include "../check.hpp"

#include <filesystem>
#include <unistd.h>

#if defined(GUILD_ENABLE_CUDA) || defined(GUILD_ENABLE_HIP)
#include "guild/cli/hardware.hpp"
#include "guild/core/execution_state.hpp"
#include "guild/core/layer.hpp"
#include "guild/core/native_head.hpp"
#include "guild/core/session.hpp"
#include "guild/kernels/mrope.hpp"
#include "guild/kernels/native_gdn.hpp"
#include "guild/kernels/native_moe.hpp"
#include "guild/kernels/native_qsa.hpp"
#include "guild/kernels/native_qsa_indexer.hpp"
#include "guild/kernels/native_rope.hpp"
#include "guild/kernels/native_router.hpp"
#include "guild/kernels/ple.hpp"
#include "guild/kernels/shared_expert.hpp"
#include "../../src/runtime/model_impl.hpp"
#endif

#include <atomic>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace {

void test_scoped_expert_layout_concurrency() {
    std::cout << "[multi_model_isolation_test] 1. ScopedExpertLayout concurrent thread isolation...\n";
    using guild::kernels::cpu::expert_layout;
    using guild::kernels::cpu::expert_layout_set_gate_up_interleaved;
    using guild::kernels::cpu::ExpertLayout;
    using guild::kernels::cpu::ScopedExpertLayout;

    std::atomic<bool> start_flag{false};
    std::vector<std::thread> threads;
    for (int t = 0; t < 6; ++t) {
        threads.emplace_back([t, &start_flag]() {
            while (!start_flag.load(std::memory_order_acquire)) {}
            for (int iter = 0; iter < 500; ++iter) {
                ExpertLayout local{};
                local.native = (t % 2 == 0);
                local.n_layers = 32 + t * 4;
                local.n_expert = 128 * (t + 1);
                local.max_blob = 1024ull * (t + 1);
                local.total = uint64_t(local.n_layers) * uint64_t(local.n_expert) * local.max_blob;

                ScopedExpertLayout scope(&local);
                const bool want_interleaved = ((iter + t) % 2 == 1);
                expert_layout_set_gate_up_interleaved(want_interleaved);

                const ExpertLayout& active = expert_layout();
                CHECK(&active == &local);
                CHECK(active.native == (t % 2 == 0));
                CHECK(active.n_layers == 32 + t * 4);
                CHECK(active.n_expert == 128 * (t + 1));
                CHECK(active.max_blob == 1024ull * (t + 1));
                CHECK(active.gate_up_interleaved == want_interleaved);
            }
        });
    }
    start_flag.store(true, std::memory_order_release);
    for (auto& th : threads) {
        th.join();
    }
    std::cout << "  -> PASSED\n";
}

#if defined(GUILD_ENABLE_CUDA) || defined(GUILD_ENABLE_HIP)

void test_model_execution_state_concurrency() {
    std::cout << "[multi_model_isolation_test] 2. ModelExecutionState concurrent scope isolation...\n";
    using namespace guild;

    // Set distinct process defaults first and verify failed/scoped mutations never leak to defaults.
    core::layer_set_shared_early(true);
    core::qsa_set_kv_resident(0);
    core::qsa_set_kv_host_only(false);
    core::qsa_set_kv_int8(false);
    core::qsa_set_kv_q4(false);
    core::qsa_set_kv_hybrid(false);
    core::session_set_kv_share_staging(-1);

    std::atomic<bool> go{false};
    std::vector<std::thread> workers;
    for (int t = 0; t < 6; ++t) {
        workers.emplace_back([t, &go]() {
            auto state = std::make_shared<core::ModelExecutionState>(core::ModelExecutionState::snapshot_current());
            const core::NativeEmbed* fake_embed =
                reinterpret_cast<const core::NativeEmbed*>(uintptr_t(0x10000 + (t + 1) * 64));
            while (!go.load(std::memory_order_acquire)) {}
            for (int iter = 0; iter < 300; ++iter) {
                core::ScopedModelExecution scope(state);
                const bool odd = ((t + iter) % 2 == 1);
                const int64_t want_res = (t + 1) * 4096 + iter;

                core::layer_set_shared_early(odd);
                core::layer_set_fused_gr(!odd);
                core::layer_set_native_bf16(odd);
                core::qsa_set_kv_resident(want_res);
                core::qsa_set_kv_host_only(odd);
                core::qsa_set_kv_int8(!odd);
                core::qsa_set_kv_q4(odd);
                core::qsa_set_kv_hybrid(!odd);
                core::session_set_kv_share_staging(odd ? 1 : 0);
                core::set_native_embed(fake_embed);
                kernels::cpu::expert_layout_set_gate_up_interleaved(odd);

                kernels::RopeScaling rs{};
                rs.type = odd ? kernels::RopeScalingType::Linear : kernels::RopeScalingType::YaRN;
                rs.freq_base = 100000.0f * float(t + 1);
                rs.factor = float(t + 2);
                kernels::rope_scaling_set(rs);

                kernels::ple_set_native_bf16(odd);
                kernels::shared_expert_set_native_bf16(!odd);
                kernels::native_router_set_enabled(odd);
                kernels::native_moe_combine_set_enabled(!odd);
                kernels::native_qsa_set_enabled(odd);
                kernels::native_rope_set_enabled(!odd);
                kernels::native_gdn_set_enabled(odd);

                // Verify nested scope is a fast no-op and preserves state
                {
                    core::ScopedModelExecution nested(state.get());
                    CHECK(core::layer_shared_early() == odd);
                    CHECK(core::layer_fused_gr() == !odd);
                    CHECK(core::qsa_kv_resident() == want_res);
                    CHECK(core::qsa_kv_host_only() == odd);
                    CHECK(core::qsa_kv_int8() == !odd);
                    CHECK(core::qsa_kv_q4() == odd);
                    CHECK(core::qsa_kv_hybrid() == !odd);
                    CHECK(core::native_embed() == fake_embed);
                    CHECK(kernels::cpu::expert_layout().gate_up_interleaved == odd);
                    CHECK(kernels::rope_scaling().type == rs.type);
                    CHECK(kernels::rope_scaling().freq_base == rs.freq_base);
                    CHECK(kernels::rope_scaling().factor == rs.factor);
                    CHECK(kernels::ple_native_bf16_enabled() == odd);
                    CHECK(kernels::shared_expert_native_bf16_enabled() == !odd);
                    CHECK(kernels::native_router_enabled() == odd);
                    CHECK(kernels::native_moe_combine_enabled() == !odd);
                    CHECK(kernels::native_qsa_enabled() == odd);
                    CHECK(kernels::native_rope_enabled() == !odd);
                    CHECK(kernels::native_gdn_enabled() == odd);
                }
            }
        });
    }
    go.store(true, std::memory_order_release);
    for (auto& w : workers) {
        w.join();
    }

    // Verify process defaults were never mutated by any scoped model configuration
    CHECK(core::active_model_execution_state() == nullptr);
    CHECK(core::layer_shared_early() == true);
    CHECK(core::qsa_kv_resident() == 0);
    CHECK(core::qsa_kv_host_only() == false);
    CHECK(core::qsa_kv_int8() == false);
    CHECK(core::qsa_kv_q4() == false);
    CHECK(core::qsa_kv_hybrid() == false);
    std::cout << "  -> PASSED\n";
}

std::vector<int32_t> run_greedy(guild::runtime::GuildSession& session, const std::string& prompt, int max_tokens) {
    guild::runtime::GenerationRequest req;
    req.prompt = prompt;
    req.max_new_tokens = max_tokens;
    req.temperature = 0.0f;

    std::vector<int32_t> ids;
    guild::runtime::RuntimeTelemetry telem;
    std::string err;
    bool ok = session.generate(
        req,
        [&ids](const guild::runtime::TokenEvent& ev) {
            ids.push_back(ev.token_id);
            return true;
        },
        telem, err);
    if (!ok) {
        std::cerr << "[multi_model_isolation_test] generate failed: " << err << "\n";
    }
    CHECK(ok);
    CHECK(static_cast<int>(ids.size()) == max_tokens);
    return ids;
}

void test_real_multi_model_coexistence() {
    std::cout << "[multi_model_isolation_test] 3. Real multi-model coexistence, concurrency, and teardown...\n";
    using namespace guild;

    models::StoreOptions st_opts;
    models::ModelStore store(st_opts);
    store.init();

    auto m_opt = models::ModelRegistry::instance().resolve("ornith:35b", store);
    if (!m_opt.has_value()) {
        std::cout << "  -> SKIPPED (ornith:35b not present in local ModelStore)\n";
        return;
    }

    const auto& manifest = *m_opt;
    const auto hw = cli::detect_hardware();

    std::string native_model;
    const auto* prim = manifest.find_file_by_role("primary");
    if (!prim)
        prim = manifest.find_file_by_role("shard");
    if (prim)
        native_model = prim->local_path;

    std::string pack_dir = manifest.metadata.at("pack_dir");
    std::string tokenizer_dir = pack_dir + "/tokenizer";

    runtime::ModelPaths pathsA;
    pathsA.primary_model_path = native_model;
    pathsA.pack_dir = pack_dir;
    pathsA.tokenizer_dir = tokenizer_dir;

    // Configure Model A: 2K context, Resident FP16 KV, shared_early=true, standard RoPE (freq_base=10,000,000)
    auto descA = manifest.to_descriptor();
    memory::PlannerOptions poptsA;
    poptsA.context_length = 2048;
    poptsA.kv_mode = memory::KvMode::Resident;
    poptsA.kv_format = memory::KvPrecision::FP16;
    auto planA = memory::MemoryPlanner::plan(descA, hw, poptsA);
    planA.routed_experts_in_gpu = 0;
    planA.expert_ram_budget_bytes = 0;
    planA.routed_expert_total_bytes = 1;
    planA.routed_experts_ram_bytes = 0;
    planA.expert_cache_auto = false;
    planA.pool_workers = 4;
    planA.shared_early = true;
    planA.kv_resident = 0;

    // Configure Model B: 4K context, HostOnly INT8 KV, shared_early=false, aggressive Linear RoPE scaling
    // (freq_base=1.01, factor=1000.0) AND a GEXEC artifact selecting interleaved gate/up rows (when model.gpack
    // exists), proving that any cross-model configuration or GEXEC layout leak between A and B is immediately caught!
    runtime::ModelPaths pathsB = pathsA;
    auto descB = manifest.to_descriptor();
    descB.rope.type = "linear";
    descB.rope.factor = 1.0f;
    descB.rope.freq_base = 1.05f;
    memory::PlannerOptions poptsB;
    poptsB.context_length = 4096;
    poptsB.kv_mode = memory::KvMode::HostOnly;
    poptsB.kv_format = memory::KvPrecision::INT8;
    auto planB = memory::MemoryPlanner::plan(descB, hw, poptsB);
    planB.routed_experts_in_gpu = 0;
    planB.pcie_expert_fraction = 0.0;
    planB.expert_ram_budget_bytes = 0;
    planB.routed_expert_total_bytes = 1;
    planB.routed_experts_ram_bytes = 0;
    planB.expert_cache_auto = false;
    planB.pool_workers = 4;
    planB.shared_early = false;
    planB.kv_resident = 20480;

    const std::filesystem::path gpack_ext = "/mnt/models-ssd/Strata-data/gpacks/ornith-1.5-35b-ext.gpack";
    const std::filesystem::path tmp_gexec =
        std::filesystem::temp_directory_path() / ("guild-multi-model-B-" + std::to_string(getpid()) + ".gexec");
    bool have_gexec_b = false;
    if (std::filesystem::is_regular_file(gpack_ext)) {
        artifact::GPackReader gpack_reader;
        std::string gerr;
        if (gpack_reader.open(gpack_ext.string(), gerr, false)) {
            const int64_t pf_batch =
                planB.prefill_batch > 0 ? planB.prefill_batch : (planB.prefill_chunk > 0 ? planB.prefill_chunk : 512);
            const auto gid = gexec::build_current_gexec_identity(gpack_reader, "int8", "host_only", 4096, pf_batch,
                                                                 planB.mtp_spec_tokens);
            gexec::GExecAutotuneTable at{};
            at.identity = gid;
            gexec::GExecKernelTuneRecord rec{};
            rec.op_kind = GUILD_OP_BATCHED_EXPERT_EXEC;
            rec.selected_impl_id = kernels::cpu::kGExecInterleavedAvx2ImplId;
            rec.selected_impl_version = kernels::cpu::kGExecExpertImplVersion;
            at.records.push_back(rec);

            gexec::GExecExpertPackMeta pmeta{};
            pmeta.layout_kind = gexec::ExpertPackLayoutKind::GateUpRowInterleaved;
            pmeta.embedded_blob_present = false;
            pmeta.fused_single_barrier_pipeline = true;
            pmeta.n_layers = static_cast<int64_t>(gpack_reader.expert_layers().size());
            pmeta.n_expert = gpack_reader.expert_count_per_layer();
            pmeta.total_expert_bytes = gpack_reader.total_expert_bytes();
            for (const auto& layer : gpack_reader.expert_layers()) {
                gexec::GExecExpertPackSpec spec{};
                spec.gu_type = layer.gu_type;
                spec.d_type = layer.d_type;
                spec.n_embd = gpack_reader.descriptor().attn.n_embd;
                spec.n_ff = gpack_reader.descriptor().moe.expert_dim_ff;
                spec.layout_kind = gexec::ExpertPackLayoutKind::GateUpRowInterleaved;
                pmeta.layer_specs.push_back(spec);
            }

            gexec::GExecBuilder builder;
            builder.set_identity(gid);
            builder.set_autotune_table(at);
            builder.set_expert_pack_meta(pmeta);
            if (builder.write_file(tmp_gexec.string(), gerr)) {
                pathsB.primary_model_path = gpack_ext.string();
                pathsB.gexec_path = tmp_gexec.string();
                have_gexec_b = true;
            }
        }
    }

    const std::string prompt = "Explain why process-wide global variables break multi-model inference engines:";
    constexpr int kTokens = 6;

    // Step 3.1: Load Model A and establish isolated reference output
    std::string errA;
    auto modelA = runtime::GuildModel::load(pathsA, descA, planA, errA);
    CHECK(modelA != nullptr);
    auto sessionA = modelA->create_session(2048);
    CHECK(sessionA != nullptr);
    const std::vector<int32_t> refA = run_greedy(*sessionA, prompt, kTokens);
    std::cout << "  Model A isolated reference tokens:";
    for (int32_t id : refA)
        std::cout << " " << id;
    std::cout << "\n";

    // Step 3.2: Load Model B in the same process while Model A and sessionA remain active
    std::string errB;
    auto modelB = runtime::GuildModel::load(pathsB, descB, planB, errB);
    CHECK(modelB != nullptr);
    auto sessionB = modelB->create_session(4096);
    CHECK(sessionB != nullptr);

    // Verify per-model execution states are completely distinct in memory
    CHECK(modelA->impl()->exec_state != modelB->impl()->exec_state);
    CHECK(modelA->impl()->exec_state->shared_early == true);
    CHECK(modelB->impl()->exec_state->shared_early == false);
    CHECK(modelA->impl()->exec_state->kv_host_only == false);
    CHECK(modelB->impl()->exec_state->kv_host_only == true);
    CHECK(modelA->impl()->exec_state->kv_int8 == false);
    CHECK(modelB->impl()->exec_state->kv_int8 == true);
    CHECK(modelA->impl()->exec_state->kv_resident == 0);
    CHECK(modelB->impl()->exec_state->kv_resident == 20480);
    CHECK(modelA->impl()->exec_state->native_embed == &modelA->impl()->native_embed);
    CHECK(modelB->impl()->exec_state->native_embed == &modelB->impl()->native_embed);
    CHECK(modelA->impl()->exec_state->cuda_attn.rope_scaling.type == kernels::RopeScalingType::None);
    CHECK(modelB->impl()->exec_state->cuda_attn.rope_scaling.type == kernels::RopeScalingType::Linear);
    CHECK(modelB->impl()->exec_state->cuda_attn.rope_scaling.factor == 1.0);
    CHECK(modelB->impl()->exec_state->cuda_attn.rope_scaling.freq_base == 1.05f);
    CHECK(modelA->impl()->exec_state->expert_layout.gate_up_interleaved == false);
    CHECK(modelA->impl()->src.gate_up_interleaved() == false);
    if (have_gexec_b) {
        CHECK(modelB->startup_profile().gexec_active == true);
        CHECK(modelB->impl()->exec_state->expert_layout.gate_up_interleaved == true);
        CHECK(modelB->impl()->src.gate_up_interleaved() == true);
        std::cout << "  -> GEXEC interleaved Model B coexisting with portable Model A verified\n";
    }

    // Step 3.3: Execute Model B and verify its trajectory differs from Model A due to RoPE config
    const std::vector<int32_t> refB = run_greedy(*sessionB, prompt, kTokens);
    std::cout << "  Model B reference tokens:         ";
    for (int32_t id : refB)
        std::cout << " " << id;
    std::cout << "\n";
    CHECK(refA != refB);

    // Step 3.4: Re-execute Model A after Model B loaded and ran — must match refA token-for-token!
    const std::vector<int32_t> afterB_A = run_greedy(*sessionA, prompt, kTokens);
    CHECK(afterB_A == refA);
    std::cout << "  -> Sequential coexistence A -> B -> A verified (exact token match)\n";

    // Step 3.5: Failed load & VRAM resource exhaustion recovery while Model A and Model B are resident
    {
        std::string fail_err;
        auto bad_plan = planA;
        bad_plan.kv_mode = memory::KvMode::Streaming;
        bad_plan.kv_resident = 0;
        auto failed_model1 = runtime::GuildModel::load(pathsA, descA, bad_plan, fail_err);
        CHECK(failed_model1 == nullptr);
        CHECK(!fail_err.empty());

        // Attempt to allocate a 256K-context Resident FP16 KV buffer (>13 GiB VRAM) when <3.5 GiB VRAM remains free
        std::string oom_err;
        auto oom_plan = planA;
        oom_plan.context_length = 262144;
        oom_plan.kv_mode = memory::KvMode::Resident;
        oom_plan.kv_format = memory::KvPrecision::FP16;
        auto failed_model2 = runtime::GuildModel::load(pathsA, descA, oom_plan, oom_err);
        CHECK(failed_model2 == nullptr);
        CHECK(!oom_err.empty());
        std::cout << "  -> Resource exhaustion diagnostic verified: '" << oom_err << "'\n";

        // Verify both Model A and Model B are completely uncorrupted after failed loads
        CHECK(run_greedy(*sessionA, prompt, kTokens) == refA);
        CHECK(run_greedy(*sessionB, prompt, kTokens) == refB);
    }

    // Step 3.6: Concurrent execution of Model A and Model B on separate threads
    {
        std::vector<int32_t> concA, concB;
        std::thread thA([&]() { concA = run_greedy(*sessionA, prompt, kTokens); });
        std::thread thB([&]() { concB = run_greedy(*sessionB, prompt, kTokens); });
        thA.join();
        thB.join();
        CHECK(concA == refA);
        CHECK(concB == refB);
        std::cout << "  -> Concurrent execution of Model A and Model B verified (exact token match)\n";
    }

    // Step 3.7: Asymmetric teardown — destroy modelA handle while sessionA outlives it (sessionA must fail
    // cleanly without crashing), then destroy sessionA (releasing Model A's Impl) and verify Model B remains intact.
    {
        modelA.reset();
        runtime::GenerationRequest req{};
        req.prompt = prompt;
        req.max_new_tokens = 2;
        runtime::RuntimeTelemetry telem{};
        std::string stopped_err;
        CHECK(!sessionA->generate(req, [](const runtime::TokenEvent&) { return true; }, telem, stopped_err));
        CHECK(!stopped_err.empty());
        sessionA.reset(); // triggers GuildModel::Impl::~Impl() for Model A while Model B is still alive
        CHECK(run_greedy(*sessionB, prompt, kTokens) == refB);
        std::cout << "  -> Asymmetric teardown (sessionA outlives modelA handle; Model A destroyed before Model B) "
                     "verified\n";
    }
    sessionB.reset();
    modelB.reset();
    std::error_code ec;
    std::filesystem::remove(tmp_gexec, ec);
}

#endif

} // namespace

int main() {
    test_scoped_expert_layout_concurrency();
#if defined(GUILD_ENABLE_CUDA) || defined(GUILD_ENABLE_HIP)
    test_model_execution_state_concurrency();
    test_real_multi_model_coexistence();
#endif
    std::cout << "[multi_model_isolation_test] ALL TESTS PASSED\n";
    return 0;
}
