#include "../check.hpp"
#include "../../src/runtime/model_impl.hpp"
#include "guild/abi/gkernel_package.hpp"
#include "guild/models/store.hpp"
#include "guild/models/registry.hpp"
#include <sys/resource.h>
#include <algorithm>
#include <atomic>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>
#if defined(__linux__)
#include <dlfcn.h>
#endif

using namespace guild;
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point a) {
    return std::chrono::duration<double, std::milli>(Clock::now() - a).count();
}
double percentile(std::vector<double> v, double p) {
    if (v.empty())
        return 0.0;
    std::sort(v.begin(), v.end());
    const double idx = (p / 100.0) * double(v.size() - 1);
    const size_t lo = static_cast<size_t>(idx);
    const size_t hi = std::min(v.size() - 1, lo + 1);
    const double frac = idx - double(lo);
    return v[lo] * (1.0 - frac) + v[hi] * frac;
}
int main(int argc, char** argv) try {
    int prompt = 512, generated = 128, runs = 3, context = 32768, only_batch = 0, n_sessions = 1, spec_tokens = 0;
    double pcie_frac = -1.0;
    std::string csv, write_ids, dump, activity, json_summary, startup_json, model_arg = "ornith:35b", gpack_path,
                                                                            gexec_path;
    std::string activity_library = "./build-cuda12/libguild_activity.so";
    bool sequential = false, full_warmup = false, concurrent_pkg_ops = false, ignore_eos = false, disable_gexec = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto value = [&]() {
            if (i + 1 >= argc)
                throw std::invalid_argument("missing argument");
            return std::string(argv[++i]);
        };
        if (a == "--prompt")
            prompt = std::stoi(value());
        else if (a == "--generate")
            generated = std::stoi(value());
        else if (a == "--runs")
            runs = std::stoi(value());
        else if (a == "--context")
            context = std::stoi(value());
        else if (a == "--batch")
            only_batch = std::stoi(value());
        else if (a == "--sessions")
            n_sessions = std::stoi(value());
        else if (a == "--spec")
            spec_tokens = std::stoi(value());
        else if (a == "--pcie-frac")
            pcie_frac = std::stod(value());
        else if (a == "--ignore-eos")
            ignore_eos = true;
        else if (a == "--concurrent-pkg-ops")
            concurrent_pkg_ops = true;
        else if (a == "--json-summary")
            json_summary = value();
        else if (a == "--startup-json")
            startup_json = value();
        else if (a == "--model")
            model_arg = value();
        else if (a == "--gpack")
            gpack_path = value();
        else if (a == "--gexec")
            gexec_path = value();
        else if (a == "--disable-gexec")
            disable_gexec = true;
        else if (a == "--activity")
            activity = value();
        else if (a == "--activity-library")
            activity_library = value();
        else if (a == "--full-warmup")
            full_warmup = true;
        else if (a == "--csv")
            csv = value();
        else if (a == "--write-ids")
            write_ids = value();
        else if (a == "--dump")
            dump = value();
        else if (a == "--sequential")
            sequential = true;
        else
            throw std::invalid_argument("unknown argument: " + a);
    }
    CHECK(prompt > 0 && generated > 0 && runs > 0 && n_sessions > 0 && context >= prompt + generated);
    models::ModelStore store;
    store.init();
    auto manifest = models::ModelRegistry::instance().resolve(model_arg, store);
    CHECK(manifest);
    runtime::ModelPaths paths;
    const auto* primary = manifest->find_file_by_role("primary");
    if (!primary)
        primary = manifest->find_file_by_role("shard");
    std::string default_primary = primary ? primary->local_path : "";
    if (default_primary.empty() && manifest->name == "qwen3.8-flash-next")
        default_primary =
            "/mnt/models-ssd/Strata-data/models/unsloth-UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf";
    paths.primary_model_path = gpack_path.empty() ? default_primary : gpack_path;
    paths.gexec_path = gexec_path;
    if (disable_gexec)
        setenv("GUILD_DISABLE_GEXEC", "1", 1);
    else
        unsetenv("GUILD_DISABLE_GEXEC");
    paths.pack_dir = manifest->metadata.count("pack_dir") ? manifest->metadata.at("pack_dir")
                                                          : "/mnt/models-ssd/Strata-data/packs/unsloth-ud-iq4_xs";
    paths.tokenizer_dir = paths.pack_dir + "/tokenizer";
    if (manifest->metadata.count("expert_profile"))
        paths.expert_profile_path = manifest->metadata.at("expert_profile");
    else if (manifest->name == "qwen3.8-flash-next")
        paths.expert_profile_path = "/home/ubuntu/Guild/data/expert-profile.bin";
    if (spec_tokens > 0) {
        if (manifest->metadata.count("mtp_dir"))
            paths.mtp_dir = manifest->metadata.at("mtp_dir");
        else if (manifest->name == "qwen3.8-flash-next")
            paths.mtp_dir = "/mnt/models-ssd/Strata-data/mtp/rt";
    }
    memory::ExecutionPlan plan;
    plan.context_length = context;
    plan.prefill_batch = 512;
    plan.mtp_spec_tokens = spec_tokens;
    plan.pcie_expert_fraction = pcie_frac;
    std::string err;
    const auto load_start = Clock::now();
    auto model = runtime::GuildModel::load(paths, manifest->to_descriptor(), plan, err);
    if (!model)
        throw std::runtime_error(err);
    const double load_ms = elapsed(load_start);
    auto* m = model->impl();
    if (ignore_eos)
        m->config.eos_token_ids.clear();
    if (model_arg == "ornith:35b")
        CHECK(!m->mtp && m->batched_prefill);
    std::vector<int32_t> base;
    CHECK(model->tokenizer().encode(
        "The quick brown fox jumps over the lazy dog repeatedly to test transformer attention. ", base, false, err));
    std::vector<int32_t> ids(prompt);
    for (int i = 0; i < prompt; ++i)
        ids[i] = base[i % base.size()];
    if (!write_ids.empty()) {
        std::ofstream f(write_ids);
        CHECK(f);
        for (auto i : ids)
            f << i << ' ';
        CHECK(f);
    }
    std::ofstream file;
    if (!csv.empty()) {
        file.open(csv);
        CHECK(file);
    }
    std::ostream& out = csv.empty() ? std::cout : file;
    out << "batch,run,prompt_tokens,prefill_tokens,generated_tokens,prefill_ms,decode_ms,wall_ms,ttft_ms,prefill_tps,"
           "decode_tps,load_ms,vram_sample_bytes,host_peak_bytes,host_kv_allocated_bytes,kv_staging_bytes,prefill_"
           "scratch_bytes,expert_groups,expert_weight_bytes,activation_transfer_bytes,cpu_gu_ms,cpu_quant_ms,cpu_down_"
           "ms,cpu_group_ms,host_wait_ms,gpu_dense_ms,gpu_gdn_ms,gpu_attention_ms,file_reads,gpu_groups,finish\n"
        << std::setprecision(12);
    const std::vector<int> batches = sequential   ? std::vector<int>{0}
                                     : only_batch ? std::vector<int>{only_batch}
                                                  : std::vector<int>{8, 16, 32, 64, 128, 256, 512};
    std::vector<int32_t> expected;
    void (*activity_start)(const char*) = nullptr;
    void (*activity_mark)(const char*) = nullptr;
    void (*activity_stop)() = nullptr;
    if (!activity.empty()) {
#if defined(__linux__)
        auto* lib = dlopen(activity_library.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!lib)
            throw std::runtime_error(dlerror());
        activity_start = reinterpret_cast<decltype(activity_start)>(dlsym(lib, "guild_activity_start"));
        activity_mark = reinterpret_cast<decltype(activity_mark)>(dlsym(lib, "guild_activity_mark"));
        activity_stop = reinterpret_cast<decltype(activity_stop)>(dlsym(lib, "guild_activity_stop"));
        CHECK(activity_start && activity_mark && activity_stop);
        CHECK(runs == 1 && batches.size() == 1);
#else
        throw std::runtime_error("activity collection requires Linux and optional CUPTI library");
#endif
    }
    for (int batch : batches) {
        if (sequential)
            setenv("GUILD_PREFILL_SEQUENTIAL", "1", 1);
        else
            unsetenv("GUILD_PREFILL_SEQUENTIAL");
        if (batch && m->batched_prefill)
            m->batched_prefill->set_batch_size(batch);
        runtime::GenerationRequest warm;
        warm.prompt_tokens = {9419, 11, 220};
        warm.max_new_tokens = 4;
        if (full_warmup) {
            warm.prompt_tokens = ids;
            warm.max_new_tokens = generated;
        }
        runtime::RuntimeTelemetry warm_t;
        auto session = model->create_session();
        std::vector<int32_t> warm_out;
        CHECK(session->generate(
            warm,
            [&](const runtime::TokenEvent& ev) {
                warm_out.push_back(ev.token_id);
                return true;
            },
            warm_t, err));
        if (full_warmup && expected.empty())
            expected = warm_out;

        if (n_sessions == 1 && !concurrent_pkg_ops && json_summary.empty()) {
            for (int r = 0; r < runs; ++r) {
                CHECK(m->ver->wait_commit(err));
                CHECK(cudaDeviceSynchronize() == cudaSuccess);
                if (activity_start) {
                    activity_start(activity.c_str());
                    activity_mark("prefill_and_first_token");
                }
                runtime::GenerationRequest req;
                req.prompt_tokens = ids;
                req.max_new_tokens = generated;
                runtime::RuntimeTelemetry t;
                std::vector<int32_t> output;
                const auto file0 = m->src.file_reads();
                const auto start = Clock::now();
                double first_ms = 0;
                CHECK(session->generate(
                    req,
                    [&](const runtime::TokenEvent& ev) {
                        if (output.empty()) {
                            first_ms = elapsed(start);
                            if (activity_mark)
                                activity_mark("decode");
                            if (!dump.empty()) {
                                std::vector<float> logits(model->descriptor().attn.vocab_size);
                                CHECK(m->ver->copy_logits(0, logits.data()));
                                std::ofstream f(dump + ".f32", std::ios::binary);
                                CHECK(f.write(reinterpret_cast<const char*>(logits.data()), logits.size() * 4));
                            }
                        }
                        output.push_back(ev.token_id);
                        return true;
                    },
                    t, err));
                CHECK(m->ver->wait_commit(err));
                CHECK(cudaDeviceSynchronize() == cudaSuccess);
                const double wall = elapsed(start);
                if (activity_stop) {
                    activity_mark("end");
                    activity_stop();
                }
                if (spec_tokens == 0)
                    CHECK(t.completion_tokens == int(output.size()) && t.drafts_offered == 0 && t.drafts_accepted == 0);
                else
                    CHECK(t.completion_tokens == int(output.size()));
                // Early EOS is recorded, never relabeled as the requested output length.
                if (expected.empty())
                    expected = output;
                else
                    CHECK(expected == output);
                if (!dump.empty()) {
                    std::ofstream f(dump + ".ids");
                    for (auto id : output)
                        f << id << ' ';
                    CHECK(f);
                }
                size_t free = 0, total = 0;
                CHECK(cudaMemGetInfo(&free, &total) == cudaSuccess);
                rusage usage{};
                CHECK(getrusage(RUSAGE_SELF, &usage) == 0);
                const auto& st = m->ss.qsa_states[m->ss.qsa_primary()];
                const uint64_t stage = uint64_t(st.n_slots) * 4 * m->g.n_head_kv * m->g.head_dim * 4;
                const runtime::BatchedPrefillStats p =
                    (sequential || !m->batched_prefill) ? runtime::BatchedPrefillStats{} : m->batched_prefill->stats();
                out << batch << ',' << r << ',' << t.prompt_tokens << ',' << prompt - 1 << ',' << t.completion_tokens
                    << ',' << t.prompt_ms << ',' << t.decode_ms << ',' << wall << ',' << first_ms << ','
                    << t.prompt_tok_s << ',' << t.decode_tok_s << ',' << load_ms << ',' << total - free << ','
                    << uint64_t(usage.ru_maxrss) * 1024 << ',' << core::qsa_kv_host_bytes() << ',' << stage << ','
                    << p.device_bytes << ',' << p.expert_groups << ',' << p.expert_bytes << ',' << p.activation_bytes
                    << ',' << p.ms_cpu_gu << ',' << p.ms_cpu_quant << ',' << p.ms_cpu_down << ',' << p.cpu_group_ms
                    << ',' << p.host_wait_ms << ',' << p.gpu_dense_ms << ',' << p.gpu_gdn_ms << ','
                    << p.gpu_attention_ms << ',' << m->src.file_reads() - file0 << ',' << p.gpu_groups << ','
                    << t.finish_reason << '\n';
                out.flush();
                std::cerr << "batch=" << batch << " prompt=" << prompt << " generated=" << t.completion_tokens
                          << " pp=" << t.prompt_tok_s << " tg=" << t.decode_tok_s << '\n';
                if (!startup_json.empty() && r == 0) {
                    const auto& sp = model->startup_profile();
                    std::ofstream sf(startup_json);
                    CHECK(sf);
                    sf << std::setprecision(10) << "{\n"
                       << "  \"container_format\": \"" << sp.container_format << "\",\n"
                       << "  \"load_ms\": " << load_ms << ",\n"
                       << "  \"startup\": {\n"
                       << "    \"package_load_ms\": " << sp.package_load_ms << ",\n"
                       << "    \"model_open_parse_ms\": " << sp.model_open_parse_ms << ",\n"
                       << "    \"runner_and_plan_ms\": " << sp.runner_and_plan_ms << ",\n"
                       << "    \"cuda_init_ms\": " << sp.cuda_init_ms << ",\n"
                       << "    \"kernel_provider_resolve_ms\": " << sp.kernel_provider_resolve_ms << ",\n"
                       << "    \"quant_layout_prep_ms\": " << sp.quant_layout_prep_ms << ",\n"
                       << "    \"embed_load_ms\": " << sp.embed_load_ms << ",\n"
                       << "    \"dense_index_and_pack_ms\": " << sp.dense_index_and_pack_ms << ",\n"
                       << "    \"native_dense_load_ms\": " << sp.native_dense_load_ms << ",\n"
                       << "    \"native_head_load_ms\": " << sp.native_head_load_ms << ",\n"
                       << "    \"expert_source_open_ms\": " << sp.expert_source_open_ms << ",\n"
                       << "    \"expert_calibration_pre_ms\": " << sp.expert_calibration_pre_ms << ",\n"
                       << "    \"expert_cache_init_ms\": " << sp.expert_cache_init_ms << ",\n"
                       << "    \"numa_expert_pin_ms\": " << sp.numa_expert_pin_ms << ",\n"
                       << "    \"kv_session_init_ms\": " << sp.kv_session_init_ms << ",\n"
                       << "    \"ple_bind_ms\": " << sp.ple_bind_ms << ",\n"
                       << "    \"mtp_init_ms\": " << sp.mtp_init_ms << ",\n"
                       << "    \"verifier_init_ms\": " << sp.verifier_init_ms << ",\n"
                       << "    \"prefill_init_ms\": " << sp.prefill_init_ms << ",\n"
                       << "    \"expert_calibration_post_ms\": " << sp.expert_calibration_post_ms << ",\n"
                       << "    \"graph_capture_startup_ms\": " << sp.graph_capture_startup_ms << ",\n"
                       << "    \"tokenizer_load_ms\": " << sp.tokenizer_load_ms << ",\n"
                       << "    \"total_load_ms\": " << sp.total_load_ms << ",\n"
                       << "    \"container_opens\": " << sp.container_opens << ",\n"
                       << "    \"index_parses\": " << sp.index_parses << ",\n"
                       << "    \"dense_device_allocs\": " << sp.dense_device_allocs << ",\n"
                       << "    \"expert_copy_calls\": " << sp.expert_copy_calls << ",\n"
                       << "    \"dense_vram_bytes\": " << sp.dense_vram_bytes << ",\n"
                       << "    \"expert_vram_bytes\": " << sp.expert_vram_bytes << ",\n"
                       << "    \"expert_ram_bytes\": " << sp.expert_ram_bytes << ",\n"
                       << "    \"kv_host_bytes\": " << sp.kv_host_bytes << ",\n"
                       << "    \"kv_device_bytes\": " << sp.kv_device_bytes << ",\n"
                       << "    \"expert_zero_copy\": " << (sp.expert_zero_copy ? "true" : "false") << ",\n"
                       << "    \"expert_numa_interleaved\": " << (sp.expert_numa_interleaved ? "true" : "false")
                       << ",\n"
                       << "    \"expert_page_locked\": " << (sp.expert_page_locked ? "true" : "false") << ",\n"
                       << "    \"gexec_active\": " << (sp.gexec_active ? "true" : "false") << ",\n"
                       << "    \"gexec_path\": \"" << sp.gexec_path << "\",\n"
                       << "    \"gexec_cache_key\": \"" << sp.gexec_cache_key << "\",\n"
                       << "    \"gexec_cpu_expert_impl\": \"" << sp.gexec_cpu_expert_impl << "\",\n"
                       << "    \"gexec_expert_layout\": \"" << sp.gexec_expert_layout << "\",\n"
                       << "    \"gexec_fallback_reason\": \"" << sp.gexec_fallback_reason << "\"\n"
                       << "  },\n"
                       << "  \"first_inference\": {\n"
                       << "    \"prompt_ms\": " << warm_t.prompt_ms << ",\n"
                       << "    \"decode_ms\": " << warm_t.decode_ms << ",\n"
                       << "    \"ttft_ms\": " << warm_t.ttft_ms << ",\n"
                       << "    \"first_window_capture_ms\": " << warm_t.first_window_capture_ms << "\n"
                       << "  },\n"
                       << "  \"steady_state\": {\n"
                       << "    \"prompt_tokens\": " << t.prompt_tokens << ",\n"
                       << "    \"completion_tokens\": " << t.completion_tokens << ",\n"
                       << "    \"prompt_ms\": " << t.prompt_ms << ",\n"
                       << "    \"decode_ms\": " << t.decode_ms << ",\n"
                       << "    \"ttft_ms\": " << first_ms << ",\n"
                       << "    \"prefill_tps\": " << t.prompt_tok_s << ",\n"
                       << "    \"decode_tps\": " << t.decode_tok_s << ",\n"
                       << "    \"prefill_embed_ms\": " << p.ms_embed << ",\n"
                       << "    \"prefill_gpu_dense_ms\": " << p.gpu_dense_ms << ",\n"
                       << "    \"prefill_gpu_attn_ms\": " << p.gpu_attention_ms << ",\n"
                       << "    \"prefill_gpu_gdn_ms\": " << p.gpu_gdn_ms << ",\n"
                       << "    \"prefill_host_wait_ms\": " << p.host_wait_ms << ",\n"
                       << "    \"prefill_cpu_group_ms\": " << p.cpu_group_ms << ",\n"
                       << "    \"prefill_cpu_quant_ms\": " << p.ms_cpu_quant << ",\n"
                       << "    \"prefill_cpu_gu_ms\": " << p.ms_cpu_gu << ",\n"
                       << "    \"prefill_cpu_down_ms\": " << p.ms_cpu_down << ",\n"
                       << "    \"decode_verify_ms\": " << t.decode_verify_ms << ",\n"
                       << "    \"decode_mtp_draft_ms\": " << t.decode_mtp_draft_ms << ",\n"
                       << "    \"decode_host_wait_ms\": " << t.decode_host_wait_ms << ",\n"
                       << "    \"decode_cpu_pool_ms\": " << t.decode_cpu_pool_ms << ",\n"
                       << "    \"decode_cpu_validation_ms\": " << t.decode_cpu_validation_ms << ",\n"
                       << "    \"decode_cpu_quant_ms\": " << t.decode_cpu_quant_ms << ",\n"
                       << "    \"decode_cpu_gu_ms\": " << t.decode_cpu_gu_ms << ",\n"
                       << "    \"decode_cpu_down_ms\": " << t.decode_cpu_down_ms << ",\n"
                       << "    \"decode_ple_ms\": " << t.decode_ple_ms << ",\n"
                       << "    \"decode_commit_ms\": " << t.decode_commit_ms << ",\n"
                       << "    \"decode_logit_validation_ms\": " << t.decode_logit_validation_ms << ",\n"
                       << "    \"decode_windows\": " << t.decode_windows << ",\n"
                       << "    \"vram_sample_bytes\": " << (total - free) << ",\n"
                       << "    \"host_peak_bytes\": " << (uint64_t(usage.ru_maxrss) * 1024ULL) << "\n"
                       << "  }\n"
                       << "}\n";
                }
            }
        } else {
            std::vector<std::unique_ptr<runtime::GuildSession>> sessions_vec;
            sessions_vec.push_back(std::move(session));
            for (int s = 1; s < n_sessions; ++s) {
                auto ns = model->create_session(context);
                CHECK(ns != nullptr);
                sessions_vec.push_back(std::move(ns));
            }
            if (expected.empty()) {
                runtime::GenerationRequest ref_req;
                ref_req.prompt_tokens = ids;
                ref_req.max_new_tokens = generated;
                runtime::RuntimeTelemetry ref_t;
                CHECK(sessions_vec[0]->generate(
                    ref_req,
                    [&](const runtime::TokenEvent& ev) {
                        expected.push_back(ev.token_id);
                        return true;
                    },
                    ref_t, err));
            }
            std::atomic<bool> stop_pkg{false};
            std::atomic<int64_t> pkg_list_ops{0};
            std::atomic<int64_t> busy_rejections{0};
            std::atomic<int64_t> max_kernel_leases{0};
            std::atomic<int64_t> max_runner_leases{0};
            std::thread pkg_thread;
            if (concurrent_pkg_ops) {
                pkg_thread = std::thread([&]() {
                    auto& mgr = abi::GKernelPackageManager::instance();
                    auto& kreg = kernels::KernelRegistry::instance();
                    auto& rreg = model::ModelRunnerRegistry::instance();
                    while (!stop_pkg.load(std::memory_order_relaxed)) {
                        auto pkgs = mgr.list_packages();
                        pkg_list_ops.fetch_add(1, std::memory_order_relaxed);
                        int64_t k_sum = 0, r_sum = 0;
                        for (const auto& pkg : pkgs) {
                            k_sum += kreg.active_package_leases(pkg.package_id);
                            r_sum += rreg.active_package_leases(pkg.package_id);
                            if (pkg.enabled && pkg.loaded_in_process) {
                                std::string perr;
                                if (mgr.disable_package(pkg.package_id, perr) == GUILD_STATUS_ERR_ACTIVE_SESSION_BUSY) {
                                    busy_rejections.fetch_add(1, std::memory_order_relaxed);
                                }
                                if (mgr.remove_package(pkg.package_id, perr) == GUILD_STATUS_ERR_ACTIVE_SESSION_BUSY) {
                                    busy_rejections.fetch_add(1, std::memory_order_relaxed);
                                }
                            }
                        }
                        for (const char* builtin_pkg : {"builtin.guild-kernels-cpu", "builtin.guild-kernels-cuda",
                                                        "builtin.guild-kernels-ornith"}) {
                            k_sum += kreg.active_package_leases(builtin_pkg);
                            r_sum += rreg.active_package_leases(builtin_pkg);
                        }
                        int64_t prev_k = max_kernel_leases.load(std::memory_order_relaxed);
                        while (k_sum > prev_k && !max_kernel_leases.compare_exchange_weak(prev_k, k_sum)) {}
                        int64_t prev_r = max_runner_leases.load(std::memory_order_relaxed);
                        while (r_sum > prev_r && !max_runner_leases.compare_exchange_weak(prev_r, r_sum)) {}
                    }
                });
            }
            std::vector<double> all_prefill_tps, all_decode_tps, all_ttft_ms, all_wall_ms, round_wall_ms;
            std::mutex rec_mu;
            for (int r = 0; r < runs; ++r) {
                const auto round_start = Clock::now();
                std::vector<std::thread> workers;
                for (int s = 0; s < n_sessions; ++s) {
                    workers.emplace_back([&, s, r]() {
                        runtime::GenerationRequest req;
                        req.prompt_tokens = ids;
                        req.max_new_tokens = generated;
                        runtime::RuntimeTelemetry t;
                        std::vector<int32_t> output;
                        const auto start = Clock::now();
                        double first_ms = 0;
                        std::string lerr;
                        CHECK(sessions_vec[static_cast<size_t>(s)]->generate(
                            req,
                            [&](const runtime::TokenEvent& ev) {
                                if (output.empty())
                                    first_ms = elapsed(start);
                                output.push_back(ev.token_id);
                                return true;
                            },
                            t, lerr));
                        const double wall = elapsed(start);
                        CHECK(output == expected);
                        std::lock_guard<std::mutex> lk(rec_mu);
                        all_prefill_tps.push_back(t.prompt_tok_s);
                        all_decode_tps.push_back(t.decode_tok_s);
                        all_ttft_ms.push_back(first_ms);
                        all_wall_ms.push_back(wall);
                        size_t free = 0, total = 0;
                        CHECK(cudaMemGetInfo(&free, &total) == cudaSuccess);
                        rusage usage{};
                        CHECK(getrusage(RUSAGE_SELF, &usage) == 0);
                        const auto& st = m->ss.qsa_states[m->ss.qsa_primary()];
                        const uint64_t stage = uint64_t(st.n_slots) * 4 * m->g.n_head_kv * m->g.head_dim * 4;
                        const runtime::BatchedPrefillStats p =
                            sequential ? runtime::BatchedPrefillStats{} : m->batched_prefill->stats();
                        out << batch << ',' << (r * n_sessions + s) << ',' << t.prompt_tokens << ',' << prompt - 1
                            << ',' << t.completion_tokens << ',' << t.prompt_ms << ',' << t.decode_ms << ',' << wall
                            << ',' << first_ms << ',' << t.prompt_tok_s << ',' << t.decode_tok_s << ',' << load_ms
                            << ',' << total - free << ',' << uint64_t(usage.ru_maxrss) * 1024 << ','
                            << core::qsa_kv_host_bytes() << ',' << stage << ',' << p.device_bytes << ','
                            << p.expert_groups << ',' << p.expert_bytes << ',' << p.activation_bytes << ','
                            << p.ms_cpu_gu << ',' << p.ms_cpu_quant << ',' << p.ms_cpu_down << ',' << p.cpu_group_ms
                            << ',' << p.host_wait_ms << ',' << p.gpu_dense_ms << ',' << p.gpu_gdn_ms << ','
                            << p.gpu_attention_ms << ',' << 0 << ',' << p.gpu_groups << ',' << t.finish_reason << '\n';
                    });
                }
                for (auto& th : workers)
                    th.join();
                round_wall_ms.push_back(elapsed(round_start));
            }
            stop_pkg.store(true);
            if (pkg_thread.joinable())
                pkg_thread.join();
            if (!json_summary.empty()) {
                size_t free = 0, total = 0;
                CHECK(cudaMemGetInfo(&free, &total) == cudaSuccess);
                rusage usage{};
                CHECK(getrusage(RUSAGE_SELF, &usage) == 0);
                const auto& st = m->ss.qsa_states[m->ss.qsa_primary()];
                const uint64_t single_pool_stage = uint64_t(st.n_slots) * 4 * m->g.n_head_kv * m->g.head_dim * 4;
                const bool shares = (m->ss.qsa_alloc > 1) ? m->ss.qsa_states[m->ss.qsa_ord0 + 1].shares_staging : false;
                const uint64_t total_stage = (shares ? 1ULL : uint64_t(m->ss.qsa_alloc)) * single_pool_stage;
                std::ofstream jf(json_summary);
                CHECK(jf);
                jf << std::setprecision(10) << "{\n"
                   << "  \"sessions\": " << n_sessions << ",\n"
                   << "  \"runs\": " << runs << ",\n"
                   << "  \"total_requests\": " << (n_sessions * runs) << ",\n"
                   << "  \"prompt_tokens\": " << prompt << ",\n"
                   << "  \"generated_tokens\": " << generated << ",\n"
                   << "  \"context_length\": " << context << ",\n"
                   << "  \"prefill_batch\": " << batch << ",\n"
                   << "  \"prefill_tps_p50\": " << percentile(all_prefill_tps, 50.0) << ",\n"
                   << "  \"prefill_tps_p95\": " << percentile(all_prefill_tps, 95.0) << ",\n"
                   << "  \"decode_tps_p50\": " << percentile(all_decode_tps, 50.0) << ",\n"
                   << "  \"decode_tps_p95\": " << percentile(all_decode_tps, 95.0) << ",\n"
                   << "  \"ttft_ms_p50\": " << percentile(all_ttft_ms, 50.0) << ",\n"
                   << "  \"ttft_ms_p95\": " << percentile(all_ttft_ms, 95.0) << ",\n"
                   << "  \"request_latency_ms_p50\": " << percentile(all_wall_ms, 50.0) << ",\n"
                   << "  \"request_latency_ms_p95\": " << percentile(all_wall_ms, 95.0) << ",\n"
                   << "  \"round_wall_ms_p50\": " << percentile(round_wall_ms, 50.0) << ",\n"
                   << "  \"vram_sample_bytes\": " << (total - free) << ",\n"
                   << "  \"host_peak_rss_bytes\": " << (uint64_t(usage.ru_maxrss) * 1024ULL) << ",\n"
                   << "  \"host_kv_allocated_bytes\": " << core::qsa_kv_host_bytes() << ",\n"
                   << "  \"kv_staging_pool_bytes\": " << single_pool_stage << ",\n"
                   << "  \"kv_staging_total_bytes\": " << total_stage << ",\n"
                   << "  \"kv_shares_staging\": " << (shares ? "true" : "false") << ",\n"
                   << "  \"concurrent_pkg_list_ops\": " << pkg_list_ops.load() << ",\n"
                   << "  \"deferred_unload_rejections\": " << busy_rejections.load() << ",\n"
                   << "  \"max_observed_kernel_leases\": " << max_kernel_leases.load() << ",\n"
                   << "  \"max_observed_runner_leases\": " << max_runner_leases.load() << ",\n"
                   << "  \"deterministic_token_parity\": true\n"
                   << "}\n";
            }
        }
    }
    return 0;
} catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
}
