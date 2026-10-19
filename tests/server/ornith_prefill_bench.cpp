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
using Clock=std::chrono::steady_clock;
double elapsed(Clock::time_point a){return std::chrono::duration<double,std::milli>(Clock::now()-a).count();}
double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const double idx = (p / 100.0) * double(v.size() - 1);
    const size_t lo = static_cast<size_t>(idx);
    const size_t hi = std::min(v.size() - 1, lo + 1);
    const double frac = idx - double(lo);
    return v[lo] * (1.0 - frac) + v[hi] * frac;
}
int main(int argc,char**argv) try {
    int prompt=512,generated=128,runs=3,context=32768,only_batch=0,n_sessions=1;
    std::string csv,write_ids,dump,activity,json_summary;
    std::string activity_library="./build-cuda12/libguild_activity.so";
    bool sequential=false,full_warmup=false,concurrent_pkg_ops=false;
    for(int i=1;i<argc;++i) {
        const std::string a=argv[i];auto value=[&](){if(i+1>=argc)throw std::invalid_argument("missing argument");return std::string(argv[++i]);};
        if(a=="--prompt")prompt=std::stoi(value());else if(a=="--generate")generated=std::stoi(value());
        else if(a=="--runs")runs=std::stoi(value());else if(a=="--context")context=std::stoi(value());
        else if(a=="--batch")only_batch=std::stoi(value());
        else if(a=="--sessions")n_sessions=std::stoi(value());
        else if(a=="--concurrent-pkg-ops")concurrent_pkg_ops=true;
        else if(a=="--json-summary")json_summary=value();
        else if(a=="--activity")activity=value();else if(a=="--activity-library")activity_library=value();
        else if(a=="--full-warmup")full_warmup=true;
        else if(a=="--csv")csv=value();else if(a=="--write-ids")write_ids=value();else if(a=="--dump")dump=value();
        else if(a=="--sequential")sequential=true;else throw std::invalid_argument("unknown argument: "+a);
    }
    CHECK(prompt>0&&generated>0&&runs>0&&n_sessions>0&&context>=prompt+generated);
    models::ModelStore store;store.init();auto manifest=models::ModelRegistry::instance().resolve("ornith:35b",store);CHECK(manifest);
    runtime::ModelPaths paths;const auto* primary=manifest->find_file_by_role("primary");if(!primary)primary=manifest->find_file_by_role("shard");CHECK(primary);
    paths.primary_model_path=primary->local_path;paths.pack_dir=manifest->metadata.at("pack_dir");paths.tokenizer_dir=paths.pack_dir+"/tokenizer";
    memory::ExecutionPlan plan;plan.context_length=context;plan.prefill_batch=512;plan.mtp_spec_tokens=0;
    std::string err;const auto load_start=Clock::now();auto model=runtime::GuildModel::load(paths,manifest->to_descriptor(),plan,err);
    if(!model)throw std::runtime_error(err);
    const double load_ms=elapsed(load_start);auto* m=model->impl();CHECK(!m->mtp&&m->batched_prefill);
    std::vector<int32_t> base;CHECK(model->tokenizer().encode("The quick brown fox jumps over the lazy dog repeatedly to test transformer attention. ",base,false,err));
    std::vector<int32_t> ids(prompt);for(int i=0;i<prompt;++i)ids[i]=base[i%base.size()];
    if(!write_ids.empty()){std::ofstream f(write_ids);CHECK(f);for(auto i:ids)f<<i<<' ';CHECK(f);}
    std::ofstream file;if(!csv.empty()){file.open(csv);CHECK(file);}
    std::ostream& out=csv.empty()?std::cout:file;
    out<<"batch,run,prompt_tokens,prefill_tokens,generated_tokens,prefill_ms,decode_ms,wall_ms,ttft_ms,prefill_tps,decode_tps,load_ms,vram_sample_bytes,host_peak_bytes,host_kv_allocated_bytes,kv_staging_bytes,prefill_scratch_bytes,expert_groups,expert_weight_bytes,activation_transfer_bytes,cpu_gu_ms,cpu_quant_ms,cpu_down_ms,cpu_group_ms,host_wait_ms,gpu_dense_ms,gpu_gdn_ms,gpu_attention_ms,file_reads,gpu_groups,finish\n"<<std::setprecision(12);
    const std::vector<int> batches=sequential?std::vector<int>{0}:only_batch?std::vector<int>{only_batch}:std::vector<int>{8,16,32,64,128,256,512};
    std::vector<int32_t> expected;
    void (*activity_start)(const char*)=nullptr;
    void (*activity_mark)(const char*)=nullptr;
    void (*activity_stop)()=nullptr;
    if(!activity.empty()) {
#if defined(__linux__)
        auto* lib=dlopen(activity_library.c_str(),RTLD_NOW|RTLD_LOCAL);
        if(!lib)throw std::runtime_error(dlerror());
        activity_start=reinterpret_cast<decltype(activity_start)>(dlsym(lib,"guild_activity_start"));
        activity_mark=reinterpret_cast<decltype(activity_mark)>(dlsym(lib,"guild_activity_mark"));
        activity_stop=reinterpret_cast<decltype(activity_stop)>(dlsym(lib,"guild_activity_stop"));
        CHECK(activity_start&&activity_mark&&activity_stop);
        CHECK(runs==1&&batches.size()==1);
#else
        throw std::runtime_error("activity collection requires Linux and optional CUPTI library");
#endif
    }
    for(int batch:batches) {
        if(sequential)setenv("GUILD_PREFILL_SEQUENTIAL","1",1);else unsetenv("GUILD_PREFILL_SEQUENTIAL");
        if(batch)m->batched_prefill->set_batch_size(batch);
        runtime::GenerationRequest warm;warm.prompt_tokens={9419,11,220};warm.max_new_tokens=4;
        if(full_warmup){warm.prompt_tokens=ids;warm.max_new_tokens=generated;}
        runtime::RuntimeTelemetry warm_t;auto session=model->create_session();
        std::vector<int32_t> warm_out;
        CHECK(session->generate(warm,[&](const runtime::TokenEvent& ev){warm_out.push_back(ev.token_id);return true;},warm_t,err));
        if(full_warmup && expected.empty()) expected = warm_out;

        if(n_sessions == 1 && !concurrent_pkg_ops && json_summary.empty()) {
            for(int r=0;r<runs;++r) {
                CHECK(m->ver->wait_commit(err));CHECK(cudaDeviceSynchronize()==cudaSuccess);
                if(activity_start){activity_start(activity.c_str());activity_mark("prefill_and_first_token");}
                runtime::GenerationRequest req;req.prompt_tokens=ids;req.max_new_tokens=generated;
                runtime::RuntimeTelemetry t;std::vector<int32_t> output;
                const auto file0=m->src.file_reads();const auto start=Clock::now();double first_ms=0;
                CHECK(session->generate(req,[&](const runtime::TokenEvent& ev){
                    if(output.empty()) {
                        first_ms=elapsed(start);
                        if(activity_mark)activity_mark("decode");
                        if(!dump.empty()) {
                            std::vector<float> logits(model->descriptor().attn.vocab_size);CHECK(m->ver->copy_logits(0,logits.data()));
                            std::ofstream f(dump+".f32",std::ios::binary);CHECK(f.write(reinterpret_cast<const char*>(logits.data()),logits.size()*4));
                        }
                    }
                    output.push_back(ev.token_id);return true;
                },t,err));
                CHECK(m->ver->wait_commit(err));CHECK(cudaDeviceSynchronize()==cudaSuccess);const double wall=elapsed(start);
                if(activity_stop){activity_mark("end");activity_stop();}
                CHECK(t.completion_tokens==int(output.size())&&t.drafts_offered==0&&t.drafts_accepted==0);
                // Early EOS is recorded, never relabeled as the requested output length.
                if(expected.empty())expected=output;else CHECK(expected==output);
                if(!dump.empty()){std::ofstream f(dump+".ids");for(auto id:output)f<<id<<' ';CHECK(f);}
                size_t free=0,total=0;CHECK(cudaMemGetInfo(&free,&total)==cudaSuccess);
                rusage usage{};CHECK(getrusage(RUSAGE_SELF,&usage)==0);
                const auto& st=m->ss.qsa_states[m->ss.qsa_primary()];
                const uint64_t stage=uint64_t(st.n_slots)*4*m->g.n_head_kv*m->g.head_dim*4;
                const runtime::BatchedPrefillStats p=sequential?runtime::BatchedPrefillStats{}:m->batched_prefill->stats();
                out<<batch<<','<<r<<','<<t.prompt_tokens<<','<<prompt-1<<','<<t.completion_tokens<<','<<t.prompt_ms<<','<<t.decode_ms<<','<<wall<<','<<first_ms<<','
                   <<t.prompt_tok_s<<','<<t.decode_tok_s<<','<<load_ms<<','<<total-free<<','<<uint64_t(usage.ru_maxrss)*1024<<','
                   <<core::qsa_kv_host_bytes()<<','<<stage<<','<<p.device_bytes<<','<<p.expert_groups<<','<<p.expert_bytes<<','<<p.activation_bytes<<','
                   <<p.ms_cpu_gu<<','<<p.ms_cpu_quant<<','<<p.ms_cpu_down<<','<<p.cpu_group_ms<<','<<p.host_wait_ms<<','
                   <<p.gpu_dense_ms<<','<<p.gpu_gdn_ms<<','<<p.gpu_attention_ms<<','<<m->src.file_reads()-file0<<','<<p.gpu_groups<<','<<t.finish_reason<<'\n';
                out.flush();std::cerr<<"batch="<<batch<<" prompt="<<prompt<<" generated="<<t.completion_tokens<<" pp="<<t.prompt_tok_s<<" tg="<<t.decode_tok_s<<'\n';
            }
        } else {
            std::vector<std::unique_ptr<runtime::GuildSession>> sessions_vec;
            sessions_vec.push_back(std::move(session));
            for(int s=1;s<n_sessions;++s) {
                auto ns = model->create_session(context);
                CHECK(ns != nullptr);
                sessions_vec.push_back(std::move(ns));
            }
            if(expected.empty()) {
                runtime::GenerationRequest ref_req;ref_req.prompt_tokens=ids;ref_req.max_new_tokens=generated;
                runtime::RuntimeTelemetry ref_t;
                CHECK(sessions_vec[0]->generate(ref_req,[&](const runtime::TokenEvent& ev){expected.push_back(ev.token_id);return true;},ref_t,err));
            }
            std::atomic<bool> stop_pkg{false};
            std::atomic<int64_t> pkg_list_ops{0};
            std::atomic<int64_t> busy_rejections{0};
            std::atomic<int64_t> max_kernel_leases{0};
            std::atomic<int64_t> max_runner_leases{0};
            std::thread pkg_thread;
            if(concurrent_pkg_ops) {
                pkg_thread = std::thread([&]() {
                    auto& mgr = abi::GKernelPackageManager::instance();
                    auto& kreg = kernels::KernelRegistry::instance();
                    auto& rreg = model::ModelRunnerRegistry::instance();
                    while(!stop_pkg.load(std::memory_order_relaxed)) {
                        auto pkgs = mgr.list_packages();
                        pkg_list_ops.fetch_add(1, std::memory_order_relaxed);
                        int64_t k_sum = 0, r_sum = 0;
                        for(const auto& pkg : pkgs) {
                            k_sum += kreg.active_package_leases(pkg.package_id);
                            r_sum += rreg.active_package_leases(pkg.package_id);
                            if(pkg.enabled && pkg.loaded_in_process) {
                                std::string perr;
                                if(mgr.disable_package(pkg.package_id, perr) == GUILD_STATUS_ERR_ACTIVE_SESSION_BUSY) {
                                    busy_rejections.fetch_add(1, std::memory_order_relaxed);
                                }
                                if(mgr.remove_package(pkg.package_id, perr) == GUILD_STATUS_ERR_ACTIVE_SESSION_BUSY) {
                                    busy_rejections.fetch_add(1, std::memory_order_relaxed);
                                }
                            }
                        }
                        for(const char* builtin_pkg : {"builtin.guild-kernels-cpu", "builtin.guild-kernels-cuda", "builtin.guild-kernels-ornith"}) {
                            k_sum += kreg.active_package_leases(builtin_pkg);
                            r_sum += rreg.active_package_leases(builtin_pkg);
                        }
                        int64_t prev_k = max_kernel_leases.load(std::memory_order_relaxed);
                        while(k_sum > prev_k && !max_kernel_leases.compare_exchange_weak(prev_k, k_sum)) {}
                        int64_t prev_r = max_runner_leases.load(std::memory_order_relaxed);
                        while(r_sum > prev_r && !max_runner_leases.compare_exchange_weak(prev_r, r_sum)) {}
                    }
                });
            }
            std::vector<double> all_prefill_tps, all_decode_tps, all_ttft_ms, all_wall_ms, round_wall_ms;
            std::mutex rec_mu;
            for(int r=0;r<runs;++r) {
                const auto round_start = Clock::now();
                std::vector<std::thread> workers;
                for(int s=0;s<n_sessions;++s) {
                    workers.emplace_back([&, s, r]() {
                        runtime::GenerationRequest req;req.prompt_tokens=ids;req.max_new_tokens=generated;
                        runtime::RuntimeTelemetry t;std::vector<int32_t> output;
                        const auto start=Clock::now();double first_ms=0;
                        std::string lerr;
                        CHECK(sessions_vec[static_cast<size_t>(s)]->generate(req,[&](const runtime::TokenEvent& ev){
                            if(output.empty()) first_ms=elapsed(start);
                            output.push_back(ev.token_id);return true;
                        },t,lerr));
                        const double wall=elapsed(start);
                        CHECK(output == expected);
                        std::lock_guard<std::mutex> lk(rec_mu);
                        all_prefill_tps.push_back(t.prompt_tok_s);
                        all_decode_tps.push_back(t.decode_tok_s);
                        all_ttft_ms.push_back(first_ms);
                        all_wall_ms.push_back(wall);
                        size_t free=0,total=0;CHECK(cudaMemGetInfo(&free,&total)==cudaSuccess);
                        rusage usage{};CHECK(getrusage(RUSAGE_SELF,&usage)==0);
                        const auto& st=m->ss.qsa_states[m->ss.qsa_primary()];
                        const uint64_t stage=uint64_t(st.n_slots)*4*m->g.n_head_kv*m->g.head_dim*4;
                        const runtime::BatchedPrefillStats p=sequential?runtime::BatchedPrefillStats{}:m->batched_prefill->stats();
                        out<<batch<<','<<(r*n_sessions+s)<<','<<t.prompt_tokens<<','<<prompt-1<<','<<t.completion_tokens<<','<<t.prompt_ms<<','<<t.decode_ms<<','<<wall<<','<<first_ms<<','
                           <<t.prompt_tok_s<<','<<t.decode_tok_s<<','<<load_ms<<','<<total-free<<','<<uint64_t(usage.ru_maxrss)*1024<<','
                           <<core::qsa_kv_host_bytes()<<','<<stage<<','<<p.device_bytes<<','<<p.expert_groups<<','<<p.expert_bytes<<','<<p.activation_bytes<<','
                           <<p.ms_cpu_gu<<','<<p.ms_cpu_quant<<','<<p.ms_cpu_down<<','<<p.cpu_group_ms<<','<<p.host_wait_ms<<','
                           <<p.gpu_dense_ms<<','<<p.gpu_gdn_ms<<','<<p.gpu_attention_ms<<','<<0<<','<<p.gpu_groups<<','<<t.finish_reason<<'\n';
                    });
                }
                for(auto& th : workers) th.join();
                round_wall_ms.push_back(elapsed(round_start));
            }
            stop_pkg.store(true);
            if(pkg_thread.joinable()) pkg_thread.join();
            if(!json_summary.empty()) {
                size_t free=0,total=0;CHECK(cudaMemGetInfo(&free,&total)==cudaSuccess);
                rusage usage{};CHECK(getrusage(RUSAGE_SELF,&usage)==0);
                const auto& st=m->ss.qsa_states[m->ss.qsa_primary()];
                const uint64_t single_pool_stage=uint64_t(st.n_slots)*4*m->g.n_head_kv*m->g.head_dim*4;
                const bool shares = (m->ss.qsa_alloc > 1) ? m->ss.qsa_states[m->ss.qsa_ord0 + 1].shares_staging : false;
                const uint64_t total_stage = (shares ? 1ULL : uint64_t(m->ss.qsa_alloc)) * single_pool_stage;
                std::ofstream jf(json_summary);CHECK(jf);
                jf << std::setprecision(10)
                   << "{\n"
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
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
