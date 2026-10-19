// tests/core/ornith_gkernel_plugin_test.cpp
// End-to-End Real-Model Verification of External .gkernel Dynamic Plugin Execution,
// Numerical Bitwise Parity, Active-Session Lease Protection, and Fallback on Ornith-1.5-35B.
#include "../check.hpp"
#include "../signing_fixture.hpp"
#include "guild/abi/gkernel_package.hpp"
#include "guild/cli/hardware.hpp"
#include "guild/kernels/registry.hpp"
#include "guild/memory/planner.hpp"
#include "guild/models/registry.hpp"
#include "guild/models/sha256.hpp"
#include "guild/models/store.hpp"
#include "guild/runtime/model.hpp"
#include "guild/runtime/session.hpp"
#include "../../src/runtime/model_impl.hpp"

#if !defined(_WIN32)
#include <dlfcn.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using namespace guild;

int main() {
#if defined(_WIN32) || !defined(GUILD_TEST_EXT_PREFILL_SO)
    return 0;
#else
    std::cout << "[ornith_gkernel_plugin_test] Starting E2E external .gkernel inference verification...\n";

    const fs::path temp_root = fs::temp_directory_path() / "guild_ornith_gkernel_e2e";
    fs::remove_all(temp_root);
    fs::create_directories(temp_root / "pkg_src" / "lib");
    setenv("GUILD_DATA_DIR", (temp_root / "data").c_str(), 1);

    std::string err;
    CHECK(abi::GKernelPackageManager::instance().init(err));
    abi::trust_test_signing_key(abi::GKernelPackageManager::instance().security_policy());

    models::StoreOptions st_opts;
    models::ModelStore store(st_opts);
    store.init();

    auto m_opt = models::ModelRegistry::instance().resolve("ornith:35b", store);
    CHECK(m_opt.has_value());
    const auto& manifest = *m_opt;
    auto desc = manifest.to_descriptor();
    const auto hw = cli::detect_hardware();

    memory::PlannerOptions popts;
    popts.context_length = 2048;
    auto plan = memory::MemoryPlanner::plan(desc, hw, popts);

    std::string native_model;
    const auto* prim = manifest.find_file_by_role("primary");
    if (!prim) prim = manifest.find_file_by_role("shard");
    if (prim) native_model = prim->local_path;
    const std::string pack_dir = manifest.metadata.at("pack_dir");

    runtime::ModelPaths paths;
    paths.primary_model_path = native_model;
    paths.pack_dir = pack_dir;
    paths.tokenizer_dir = pack_dir + "/tokenizer";

    // Construct prompt without printing raw special tokens
    const std::string im_s = std::string("<|") + "im_start|>";
    const std::string im_e = std::string("<|") + "im_end|>";
    const std::string prompt = im_s + "user\nWhat is the capital of France? Answer in one word." + im_e + "\n" + im_s + "assistant\n";

    std::vector<float> baseline_logits;
    std::vector<int32_t> baseline_tokens;

    // -------------------------------------------------------------------------
    // Phase A: Baseline run using built-in kernels (`gpu.moe.router_topk_native`)
    // -------------------------------------------------------------------------
    std::cerr << "[ornith_gkernel_plugin_test] Phase A: loading model...\n";
    auto model = runtime::GuildModel::load(paths, desc, plan, err);
    CHECK(model != nullptr);
    CHECK(model->impl() && model->impl()->batched_prefill);
    {
        const auto& bound_ids = model->impl()->batched_prefill->bound_impl_ids();
        CHECK(std::find(bound_ids.begin(), bound_ids.end(), "gpu.moe.router_topk_native") != bound_ids.end());

        auto session = model->create_session(2048);
        CHECK(session != nullptr);

        runtime::GenerationRequest req;
        req.prompt = prompt;
        req.max_new_tokens = 1;
        req.temperature = 0.0f;
        runtime::RuntimeTelemetry telem;
        CHECK(session->generate(req, [&](const runtime::TokenEvent& ev) {
            baseline_tokens.push_back(ev.token_id);
            return true;
        }, telem, err));
        CHECK(baseline_tokens.size() == 1);

        const int64_t n_vocab = model->descriptor().attn.vocab_size > 0 ? model->descriptor().attn.vocab_size : 248320;
        baseline_logits.resize(static_cast<size_t>(n_vocab));
        CHECK(model->impl()->ver && model->impl()->ver->copy_logits(0, baseline_logits.data()));

        // Also run an 8-token greedy continuation
        baseline_tokens.clear();
        auto session8 = model->create_session(2048);
        req.max_new_tokens = 8;
        CHECK(session8->generate(req, [&](const runtime::TokenEvent& ev) {
            baseline_tokens.push_back(ev.token_id);
            return true;
        }, telem, err));
        CHECK(!baseline_tokens.empty());
        std::cerr << "[ornith_gkernel_plugin_test] Phase A complete: baseline_tokens=" << baseline_tokens.size() << "\n";
    }

    // -------------------------------------------------------------------------
    // Phase B: Build, sign, and install external .gkernel archive, then re-bind & run Ornith
    // -------------------------------------------------------------------------
    const fs::path pkg_src = temp_root / "pkg_src";
    const fs::path dst_so = pkg_src / "lib" / "libguild_ext_ornith_prefill.so";
    fs::copy_file(GUILD_TEST_EXT_PREFILL_SO, dst_so, fs::copy_options::overwrite_existing);

    abi::GKernelMetadata meta;
    meta.format_version = 1;
    meta.package_kind = "gkernel";
    meta.package_id = "ext.guild-kernels-ornith-prefill";
    meta.scope = "hybrid";
    meta.version = "1.0.0";
    meta.publisher_id = abi::kOfficialGuildPublisherId;
    meta.entrypoint_symbol = GUILD_PLUGIN_ENTRY_SYMBOL_V1;
    meta.library_path = "lib/libguild_ext_ornith_prefill.so";
    meta.abi_major = 1;
    meta.abi_minor_min = 0;
    meta.abi_minor_max = 0;
    meta.supported_backends = GUILD_BACKEND_CPU | GUILD_BACKEND_CUDA;
    meta.provided_ops = {"ext.cpu.finite_check.v1", "ext.cuda.moe.router_topk_bf16"};
    abi::ArtifactHashEntry art;
    art.relative_path = "lib/libguild_ext_ornith_prefill.so";
    art.byte_size = fs::file_size(dst_so);
    art.sha256 = models::Sha256::hash_file(dst_so.string());
    art.role = "shared_library";
    meta.artifacts.push_back(art);
    meta.provenance.spdx_license = "MIT";
    meta.provenance.source_repo = "https://github.com/guild-inference/Guild";
    meta.provenance.source_revision = "9d77159000000000000000000000000000000000";
    CHECK(meta.sign_with_seed_hex(abi::kOfficialGuildReleaseKeyId, abi::kOfficialGuildDevSeedHex, err));
    {
        std::ofstream mf(pkg_src / ".gkmetadata", std::ios::binary | std::ios::trunc);
        const std::string j = meta.to_json_string();
        mf.write(j.data(), static_cast<std::streamsize>(j.size()));
    }
    const fs::path bundle_path = temp_root / "ext_ornith_prefill.gkernel";
    CHECK(abi::create_gkernel_bundle(pkg_src.string(), bundle_path.string(), err) == GUILD_STATUS_OK);

    std::cerr << "[ornith_gkernel_plugin_test] Phase B: installing external .gkernel bundle...\n";
    CHECK(abi::GKernelPackageManager::instance().install_package(bundle_path.string(), true, err) == GUILD_STATUS_OK);

    {
        auto* m_impl = model->impl();
        const int64_t batch = m_impl->batched_prefill->batch_size();
        m_impl->batched_prefill.reset();
        m_impl->batched_prefill = std::make_unique<runtime::BatchedPrefill>();
        CHECK(m_impl->batched_prefill->init(
            m_impl->wt, m_impl->g, m_impl->ss, *m_impl->srcp, *m_impl->pool,
            m_impl->main_stream, err, batch, m_impl->xcache.valid() ? &m_impl->xcache : nullptr));
        for (const auto& pkg : m_impl->batched_prefill->bound_package_ids()) {
            m_impl->kernel_lease.add_package(pkg);
        }

        // 1. Verify external plugin's CUDA kernel was selected & bound for BatchedPrefill!
        const auto& bound_ids = m_impl->batched_prefill->bound_impl_ids();
        const auto& bound_pkgs = m_impl->batched_prefill->bound_package_ids();
        CHECK(std::find(bound_ids.begin(), bound_ids.end(), "ext.cuda.moe.router_topk_bf16") != bound_ids.end());
        CHECK(std::find(bound_pkgs.begin(), bound_pkgs.end(), "ext.guild-kernels-ornith-prefill") != bound_pkgs.end());

        // 2. Verify active model holds per-package lease preventing disable/remove while in flight
        CHECK(kernels::KernelRegistry::instance().active_package_leases("ext.guild-kernels-ornith-prefill") >= 1);
        CHECK(abi::GKernelPackageManager::instance().disable_package("ext.guild-kernels-ornith-prefill", err) ==
              GUILD_STATUS_ERR_ACTIVE_SESSION_BUSY);
        CHECK(abi::GKernelPackageManager::instance().remove_package("ext.guild-kernels-ornith-prefill", err) ==
              GUILD_STATUS_ERR_ACTIVE_SESSION_BUSY);

        // Lookup instrumentation counter in the loaded plugin
        abi::InstalledPackageRecord rec;
        CHECK(abi::GKernelPackageManager::instance().get_package("ext.guild-kernels-ornith-prefill", rec));
        const fs::path staged_so = fs::path(rec.staged_dir) / "lib" / "libguild_ext_ornith_prefill.so";
        void* h = ::dlopen(staged_so.c_str(), RTLD_NOLOAD | RTLD_NOW);
        CHECK(h != nullptr);
        auto get_calls = reinterpret_cast<int64_t (*)(void)>(::dlsym(h, "guild_ext_ornith_router_call_count"));
        auto get_rows = reinterpret_cast<int64_t (*)(void)>(::dlsym(h, "guild_ext_ornith_router_row_count"));
        auto reset_counters = reinterpret_cast<void (*)(void)>(::dlsym(h, "guild_ext_ornith_reset_counters"));
        CHECK(get_calls != nullptr && get_rows != nullptr && reset_counters != nullptr);
        reset_counters();

        // 3. Run 1-token prefill + logits check
        auto session = model->create_session(2048);
        CHECK(session != nullptr);
        runtime::GenerationRequest req;
        req.prompt = prompt;
        req.max_new_tokens = 1;
        req.temperature = 0.0f;
        runtime::RuntimeTelemetry telem;
        std::vector<int32_t> plugin_tokens;
        CHECK(session->generate(req, [&](const runtime::TokenEvent& ev) {
            plugin_tokens.push_back(ev.token_id);
            return true;
        }, telem, err));

        // Verify external CUDA kernel executed across all 40 layers of Ornith!
        const int64_t calls_after_1 = get_calls();
        const int64_t rows_after_1 = get_rows();
        std::cerr << "[ornith_gkernel_plugin_test] External CUDA router calls=" << calls_after_1
                  << ", token_rows=" << rows_after_1 << "\n";
        CHECK(calls_after_1 == 40);
        CHECK(rows_after_1 == 40 * (static_cast<int64_t>(telem.prompt_tokens) - 1));
        ::dlclose(h);

        std::vector<float> plugin_logits(baseline_logits.size());
        CHECK(m_impl->ver && m_impl->ver->copy_logits(0, plugin_logits.data()));
        CHECK(std::memcmp(baseline_logits.data(), plugin_logits.data(),
                          baseline_logits.size() * sizeof(float)) == 0);

        // 4. Run 8-token greedy continuation and verify token-for-token parity
        plugin_tokens.clear();
        auto session8 = model->create_session(2048);
        req.max_new_tokens = 8;
        CHECK(session8->generate(req, [&](const runtime::TokenEvent& ev) {
            plugin_tokens.push_back(ev.token_id);
            return true;
        }, telem, err));
        CHECK(plugin_tokens == baseline_tokens);
        session8.reset();
        session.reset();

        // 5. Multi-session (8 concurrent sessions) + concurrent package inspection/unload-attempt stress
        {
            std::vector<std::unique_ptr<runtime::GuildSession>> sessions;
            for (int i = 0; i < 8; ++i) {
                auto s = model->create_session(2048);
                CHECK(s != nullptr);
                sessions.push_back(std::move(s));
            }
            std::atomic<bool> inf_done{false};
            std::thread pkg_worker([&]() {
                while (!inf_done.load(std::memory_order_relaxed)) {
                    std::string perr;
                    CHECK(abi::GKernelPackageManager::instance().disable_package(
                              "ext.guild-kernels-ornith-prefill", perr) == GUILD_STATUS_ERR_ACTIVE_SESSION_BUSY);
                    CHECK(abi::GKernelPackageManager::instance().remove_package(
                              "ext.guild-kernels-ornith-prefill", perr) == GUILD_STATUS_ERR_ACTIVE_SESSION_BUSY);
                    auto pkgs = abi::GKernelPackageManager::instance().list_packages();
                    CHECK(!pkgs.empty());
                }
            });
            std::vector<std::thread> session_workers;
            for (int i = 0; i < 8; ++i) {
                session_workers.emplace_back([&, i]() {
                    runtime::GenerationRequest sreq;
                    sreq.prompt = prompt;
                    sreq.max_new_tokens = 8;
                    sreq.temperature = 0.0f;
                    runtime::RuntimeTelemetry stelem;
                    std::string serr;
                    std::vector<int32_t> stokens;
                    CHECK(sessions[static_cast<size_t>(i)]->generate(sreq, [&](const runtime::TokenEvent& ev) {
                        stokens.push_back(ev.token_id);
                        return true;
                    }, stelem, serr));
                    CHECK(stokens == baseline_tokens);
                });
            }
            for (auto& sw : session_workers) sw.join();
            inf_done.store(true);
            pkg_worker.join();
        }

        // Release model's kernel lease & batched_prefill before disabling/unloading package
        m_impl->batched_prefill.reset();
        m_impl->kernel_lease = kernels::SessionKernelLease{};
    }

    // -------------------------------------------------------------------------
    // Phase C: Disable external .gkernel, verify fallback inference, then remove package
    // -------------------------------------------------------------------------
    {
        kernels::ResolvedPrefillKernels fallback_table{};
        CHECK(kernels::KernelRegistry::instance().active_package_leases("ext.guild-kernels-ornith-prefill") == 0);
        CHECK(abi::GKernelPackageManager::instance().disable_package("ext.guild-kernels-ornith-prefill", err) == GUILD_STATUS_OK);

        CHECK(kernels::KernelRegistry::instance().resolve_prefill_kernels(
            kernels::query_host_device_capabilities(), fallback_table, err));
        CHECK(std::find(fallback_table.bound_impl_ids.begin(), fallback_table.bound_impl_ids.end(),
                        "gpu.moe.router_topk_native") != fallback_table.bound_impl_ids.end());
    }

    {
        auto* m_impl = model->impl();
        m_impl->batched_prefill = std::make_unique<runtime::BatchedPrefill>();
        CHECK(m_impl->batched_prefill->init(
            m_impl->wt, m_impl->g, m_impl->ss, *m_impl->srcp, *m_impl->pool,
            m_impl->main_stream, err, plan.prefill_batch, m_impl->xcache.valid() ? &m_impl->xcache : nullptr));
        const auto& bound_ids = m_impl->batched_prefill->bound_impl_ids();
        CHECK(std::find(bound_ids.begin(), bound_ids.end(), "gpu.moe.router_topk_native") != bound_ids.end());

        std::vector<int32_t> fallback_tokens;
        auto session_fb = model->create_session(2048);
        runtime::GenerationRequest req;
        req.prompt = prompt;
        req.max_new_tokens = 8;
        req.temperature = 0.0f;
        runtime::RuntimeTelemetry telem;
        CHECK(session_fb->generate(req, [&](const runtime::TokenEvent& ev) {
            fallback_tokens.push_back(ev.token_id);
            return true;
        }, telem, err));
        CHECK(fallback_tokens == baseline_tokens);
    }

    // Phase D: Verify orphaned GuildSession handle outlives GuildModel without use-after-free
    auto orphan_session = model->create_session(2048);
    CHECK(orphan_session != nullptr);
    model.reset();
    {
        runtime::GenerationRequest oreq;
        oreq.prompt = prompt;
        oreq.max_new_tokens = 4;
        runtime::RuntimeTelemetry otelem;
        std::string oerr;
        CHECK(!orphan_session->generate(oreq, nullptr, otelem, oerr));
        CHECK(oerr.find("not ready") != std::string::npos);
        orphan_session->reset();
    }
    orphan_session.reset();
    CHECK(abi::GKernelPackageManager::instance().remove_package("ext.guild-kernels-ornith-prefill", err) == GUILD_STATUS_OK);

    fs::remove_all(temp_root);
    std::cout << "[ornith_gkernel_plugin_test] PASS: external .gkernel executed in real Ornith inference with bitwise parity and clean fallback.\n";
    return 0;
#endif
}
