// Independently installed packages; run once for reference and again in a fresh
// process with GUILD_DATA_DIR pointing at the signed external package store.
#include "../check.hpp"
#include "../../src/runtime/model_impl.hpp"
#include "guild/models/store.hpp"
#include "guild/models/registry.hpp"
#include "guild/abi/gkernel_package.hpp"
#include <fstream>
#include <filesystem>
#include <cstring>
#include <iostream>
#include <dlfcn.h>

int main(int argc, char** argv) {
    CHECK(argc == 4);
    const std::string mode = argv[1], file = argv[2], expected = argv[3];
    guild::models::ModelStore store;
    store.init();
    const auto manifest = guild::models::ModelRegistry::instance().resolve("ornith:35b", store);
    CHECK(manifest);
    guild::runtime::ModelPaths paths;
    const auto* primary = manifest->find_file_by_role("primary");
    if (!primary)
        primary = manifest->find_file_by_role("shard");
    CHECK(primary);
    paths.primary_model_path = primary->local_path;
    paths.pack_dir = manifest->metadata.at("pack_dir");
    paths.tokenizer_dir = paths.pack_dir + "/tokenizer";
    guild::memory::ExecutionPlan plan;
    plan.context_length = 32768;
    plan.prefill_batch = 256;
    plan.mtp_spec_tokens = 0;
    std::string err;
    auto model = guild::runtime::GuildModel::load(paths, manifest->to_descriptor(), plan, err);
    if (!model)
        std::cerr << err << '\n';
    CHECK(model);
    auto* impl = model->impl();
    CHECK(impl->runner && expected == impl->runner->package_id());
    CHECK(impl->g.hc == 1 && impl->g.ssm_v_heads == 32 && impl->g.qsa_interval == 4);
    CHECK(impl->batched_prefill && impl->config.kv_host_only);
    if (std::getenv("GUILD_EXPECT_BUNDLED_CPU")) {
        CHECK(impl->native_cpu_package == "builtin.guild-kernels-cpu");
        const auto& ids = impl->batched_prefill->bound_impl_ids();
        CHECK(std::find(ids.begin(), ids.end(), "gpu.moe.router_topk_native") != ids.end());
    }
    if (std::getenv("GUILD_EXPECT_CPU_PACKAGE") && !std::getenv("GUILD_TEST_NO_CPU_REFERENCE")) {
        const auto op = guild::kernels::KernelRegistry::instance().find_by_id("external.cpu.native_expert.v1");
        CHECK(op && op->next_ext);
        const auto* table = static_cast<const guild_cpu_expert_ext_v1_t*>(op->next_ext);
        const auto& format = guild::kernels::cpu::expert_layout().fmt[0];
        // A complete external backend supplies format metadata itself, so Core's
        // private GGML reference instance may not have initialized yet.
        guild::kernels::cpu::NativeFmt reference_format;
        CHECK(guild::kernels::cpu::native_fmt(format.gu_type, format.d_type, format.n_embd, format.n_ff,
                                              reference_format, err));
        const uint8_t* blob = impl->src.blob(0, 0);
        CHECK(blob);
        std::vector<float> x(size_t(format.n_embd), 0.125f), a(size_t(format.n_ff)), b(a.size());
        std::vector<uint8_t> qa(size_t(format.act_bytes)), qb(qa.size());
        guild::kernels::cpu::native_quant_act(format, x.data(), qa.data());
        table->quant_act(&format, x.data(), qb.data());
        CHECK(qa == qb);
        const void* input[] = {qa.data()};
        float* out_a[] = {a.data()};
        float* out_b[] = {b.data()};
        guild::kernels::cpu::native_gu_rows(format, blob, input, 1, out_a, 0, int(format.n_ff));
        table->gate_up(&format, blob, input, 1, out_b, 0, int(format.n_ff));
        float max_difference = 0;
        size_t mismatch = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            if (a[i] != b[i])
                ++mismatch;
            max_difference = std::max(max_difference, std::abs(a[i] - b[i]));
        }
        std::cout << "CPU format=" << format.gu_type << " act=" << format.gu_act << " first=" << a[0] << "/" << b[0]
                  << " single expert mismatch=" << mismatch << " max_abs=" << max_difference << '\n';
        CHECK(std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
    }
    std::vector<int32_t> base;
    CHECK(model->tokenizer().encode(
        "The quick brown fox jumps over the lazy dog repeatedly to test transformer attention. ", base, false, err));
    guild::runtime::GenerationRequest request;
    for (size_t i = 0; i < 512; ++i)
        request.prompt_tokens.push_back(base[i % base.size()]);
    request.max_new_tokens = 8;
    request.temperature = 0;
    std::vector<int32_t> tokens;
    std::vector<float> logits;
    auto session = model->create_session();
    guild::runtime::RuntimeTelemetry telemetry;
    CHECK(session->generate(
        request,
        [&](const guild::runtime::TokenEvent& event) {
            if (tokens.empty()) {
                logits.resize(size_t(impl->ver->vocab()));
                CHECK(impl->ver->copy_logits(0, logits.data()));
            }
            tokens.push_back(event.token_id);
            return true;
        },
        telemetry, err));
    CHECK(tokens.size() == 8 && telemetry.file_expert_reads == 0);
    CHECK(impl->ver->wait_commit(err));
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    const size_t recurrent_floats =
        size_t(impl->ss.gdn_alloc) *
        (size_t(impl->g.ssm_state_size) * size_t(impl->g.ssm_v_heads) * size_t(impl->g.ssm_state_size) +
         size_t(impl->g.ssm_conv_channels) * size_t(impl->g.ssm_d_conv - 1));
    std::vector<uint8_t> state(recurrent_floats * sizeof(float));
    CHECK(cudaMemcpy(state.data(), impl->ss.gdn_state, state.size(), cudaMemcpyDeviceToHost) == cudaSuccess);
    const size_t kv_bytes = 512 * size_t(impl->g.n_head_kv) * size_t(impl->g.head_dim) * sizeof(uint16_t);
    for (int64_t i = 0; i < impl->ss.qsa_alloc; ++i) {
        const auto& kv = impl->ss.qsa_states[impl->ss.qsa_ord0 + i];
        CHECK(kv.kv_mode == 3 && kv.host.present());
        for (const auto* pool : {kv.host.k_pool, kv.host.v_pool}) {
            const size_t offset = state.size();
            state.resize(offset + kv_bytes);
            CHECK(cudaMemcpy(state.data() + offset, pool, kv_bytes, cudaMemcpyDefault) == cudaSuccess);
        }
    }
    if (const char* cuda_package = std::getenv("GUILD_EXPECT_CUDA_PACKAGE")) {
        const auto& packages = impl->batched_prefill->bound_package_ids();
        CHECK(std::find(packages.begin(), packages.end(), cuda_package) != packages.end());
        CHECK(impl->cuda_dense && impl->cuda_dense->package_id == cuda_package);
        CHECK(impl->cuda_attn && impl->cuda_attn->package_id == cuda_package);
        CHECK(impl->cuda_moe && impl->cuda_moe->package_id == cuda_package);
        guild::abi::InstalledPackageRecord record;
        CHECK(guild::abi::GKernelPackageManager::instance().get_package(cuda_package, record));
        const auto library = std::filesystem::path(record.staged_dir) / record.metadata.library_path;
        void* h = dlopen(library.c_str(), RTLD_NOW | RTLD_NOLOAD);
        CHECK(h);
        auto count = reinterpret_cast<int64_t (*)()>(dlsym(h, "guild_ext_ornith_router_call_count"));
        CHECK(count && count() > 0);
        if (std::getenv("GUILD_CUDA_INSTRUMENT")) {
            auto dense_count = reinterpret_cast<uint64_t (*)(uint32_t)>(dlsym(h, "guild_cuda_dense_call_count_v1"));
            CHECK(dense_count && dense_count(0) > 0 && dense_count(1) > 0 && dense_count(2) > 0);
            auto attn_count = reinterpret_cast<uint64_t (*)(uint32_t)>(dlsym(h, "guild_cuda_attn_call_count_v1"));
            CHECK(attn_count && attn_count(1) > 0 && attn_count(2) > 0 && attn_count(3) > 0 && attn_count(4) > 0 &&
                  attn_count(6) > 0);
            auto moe_count = reinterpret_cast<uint64_t (*)(uint32_t)>(dlsym(h, "guild_cuda_moe_call_count_v1"));
            CHECK(moe_count && moe_count(0) > 0 && moe_count(1) > 0 && moe_count(3) > 0 && moe_count(4) > 0 &&
                  moe_count(5) > 0 && moe_count(6) > 0);
            // Ornith has hc=1 and native weights: S-form GEMV and Qwen4Exp PLE/GR are inactive.
        }
        dlclose(h);
        std::cout << "external CUDA router, dense, attention and MoE execution verified\n";
    }
    if (const char* cpu_package = std::getenv("GUILD_EXPECT_CPU_PACKAGE")) {
        CHECK(impl->native_cpu_package == cpu_package);
        guild::abi::InstalledPackageRecord record;
        CHECK(guild::abi::GKernelPackageManager::instance().get_package(cpu_package, record));
        if (const char* version = std::getenv("GUILD_EXPECT_CPU_VERSION")) {
            CHECK(record.active_version == version);
            unsigned major = 0, minor = 0, patch = 0;
            CHECK(std::sscanf(version, "%u.%u.%u", &major, &minor, &patch) == 3);
            const auto operation =
                guild::kernels::KernelRegistry::instance().find_by_id("external.cpu.native_expert.v1");
            CHECK(operation && operation->impl_version == GUILD_MAKE_ABI_VERSION(major, minor, patch));
        }
        const auto library = std::filesystem::path(record.staged_dir) / record.metadata.library_path;
        void* h = dlopen(library.c_str(), RTLD_NOW | RTLD_NOLOAD);
        CHECK(h);
        auto count = reinterpret_cast<uint64_t (*)()>(dlsym(h, "guild_cpu_gate_up_calls"));
        CHECK(count && count() > 0);
        dlclose(h);
        std::cout << "external CPU gate/up execution verified\n";
    }
    if (mode == "write") {
        std::ofstream out(file, std::ios::binary);
        CHECK(out.write(reinterpret_cast<const char*>(tokens.data()), tokens.size() * sizeof(int32_t)));
        CHECK(out.write(reinterpret_cast<const char*>(logits.data()), logits.size() * sizeof(float)));
        CHECK(out.write(reinterpret_cast<const char*>(state.data()), state.size()));
    } else {
        CHECK(mode == "compare");
        std::ifstream in(file, std::ios::binary);
        std::vector<int32_t> reference(tokens.size());
        std::vector<float> reference_logits(logits.size());
        CHECK(in.read(reinterpret_cast<char*>(reference.data()), reference.size() * sizeof(int32_t)));
        CHECK(in.read(reinterpret_cast<char*>(reference_logits.data()), reference_logits.size() * sizeof(float)));
        CHECK(reference == tokens);
        CHECK(std::memcmp(logits.data(), reference_logits.data(), logits.size() * sizeof(float)) == 0);
        std::vector<uint8_t> reference_state(state.size());
        CHECK(in.read(reinterpret_cast<char*>(reference_state.data()), reference_state.size()));
        CHECK(reference_state == state);
    }
    if (expected == "org.guild.runners.qwen") {
        CHECK(impl->desc.runner_model);
        CHECK(guild::abi::GKernelPackageManager::instance().disable_package(expected, err) ==
              GUILD_STATUS_ERR_ACTIVE_SESSION_BUSY);
    }
    session.reset();
    model.reset();
    if (expected == "org.guild.runners.qwen")
        CHECK(guild::abi::GKernelPackageManager::instance().disable_package(expected, err) == GUILD_STATUS_OK);
    std::cout << "runner package=" << expected << ": first-token logits and eight greedy tokens verified\n";
}
