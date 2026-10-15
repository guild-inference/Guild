#include "../check.hpp"
#include "../../src/runtime/model_impl.hpp"
#include "guild/models/store.hpp"
#include "guild/models/registry.hpp"
#include <sys/resource.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

using namespace guild;
using Clock=std::chrono::steady_clock;
double elapsed(Clock::time_point a){return std::chrono::duration<double,std::milli>(Clock::now()-a).count();}
int main(int argc,char**argv) try {
    int prompt=512,generated=128,runs=3,context=32768,only_batch=0;std::string csv,write_ids,dump;
    bool sequential=false;
    for(int i=1;i<argc;++i) {
        const std::string a=argv[i];auto value=[&](){if(i+1>=argc)throw std::invalid_argument("missing argument");return std::string(argv[++i]);};
        if(a=="--prompt")prompt=std::stoi(value());else if(a=="--generate")generated=std::stoi(value());
        else if(a=="--runs")runs=std::stoi(value());else if(a=="--context")context=std::stoi(value());
        else if(a=="--batch")only_batch=std::stoi(value());
        else if(a=="--csv")csv=value();else if(a=="--write-ids")write_ids=value();else if(a=="--dump")dump=value();
        else if(a=="--sequential")sequential=true;else throw std::invalid_argument("unknown argument: "+a);
    }
    CHECK(prompt>0&&generated>0&&runs>0&&context>=prompt+generated);
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
    for(int batch:batches) {
        if(sequential)setenv("GUILD_PREFILL_SEQUENTIAL","1",1);else unsetenv("GUILD_PREFILL_SEQUENTIAL");
        if(batch)m->batched_prefill->set_batch_size(batch);
        runtime::GenerationRequest warm;warm.prompt_tokens={9419,11,220};warm.max_new_tokens=4;
        runtime::RuntimeTelemetry warm_t;auto session=model->create_session();CHECK(session->generate(warm,nullptr,warm_t,err));
        for(int r=0;r<runs;++r) {
            CHECK(m->ver->wait_commit(err));CHECK(cudaDeviceSynchronize()==cudaSuccess);
            runtime::GenerationRequest req;req.prompt_tokens=ids;req.max_new_tokens=generated;
            runtime::RuntimeTelemetry t;std::vector<int32_t> output;
            const auto file0=m->src.file_reads();const auto start=Clock::now();double first_ms=0;
            CHECK(session->generate(req,[&](const runtime::TokenEvent& ev){
                if(output.empty()) {
                    first_ms=elapsed(start);
                    if(!dump.empty()) {
                        std::vector<float> logits(model->descriptor().attn.vocab_size);CHECK(m->ver->copy_logits(0,logits.data()));
                        std::ofstream f(dump+".f32",std::ios::binary);CHECK(f.write(reinterpret_cast<const char*>(logits.data()),logits.size()*4));
                    }
                }
                output.push_back(ev.token_id);return true;
            },t,err));
            CHECK(m->ver->wait_commit(err));CHECK(cudaDeviceSynchronize()==cudaSuccess);const double wall=elapsed(start);
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
    }
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
