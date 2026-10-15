#include "../check.hpp"
#include "guild/runtime/model.hpp"
#include "guild/runtime/session.hpp"
#include "../../src/runtime/model_impl.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>

using namespace guild;
struct Image { std::vector<float> logits,state; std::vector<uint16_t> kv; std::vector<int32_t> output; double pp=0, tg=0; };
struct Trace {std::vector<float> residual,weights;std::vector<int32_t> ids;int64_t rows=0;};
void compare(const std::vector<float>& a,const std::vector<float>& b,const char* what) {
    CHECK(a.size()==b.size()); double num=0,den=0,maxabs=0;
    for(size_t i=0;i<a.size();++i) { CHECK(std::isfinite(a[i])&&std::isfinite(b[i]));
        const double d=double(a[i])-b[i];num+=d*d;den+=double(a[i])*a[i];maxabs=std::max(maxabs,std::abs(d)); }
    const double rel=std::sqrt(num/std::max(den,1e-30));
    std::cout<<what<<" max_abs="<<maxabs<<" relative_l2="<<rel<<" bitwise="<<(std::memcmp(a.data(),b.data(),a.size()*4)==0)<<'\n';
    CHECK(maxabs<=1e-4 && rel<=1e-5);
}
Image generate(runtime::GuildModel& model,const std::vector<int32_t>& ids,bool sequential,int batch) {
    if(sequential) setenv("GUILD_PREFILL_SEQUENTIAL","1",1);else unsetenv("GUILD_PREFILL_SEQUENTIAL");
    model.impl()->batched_prefill->set_batch_size(batch);
    auto session=model.create_session(); runtime::GenerationRequest req;req.prompt_tokens=ids;req.max_new_tokens=8;
    runtime::RuntimeTelemetry t; std::string err;Image im;
    const bool ok=session->generate(req,[&](const runtime::TokenEvent& ev){im.output.push_back(ev.token_id);return true;},t,err);
    if(!ok)std::cerr<<err<<'\n';
    CHECK(ok);CHECK(t.drafts_offered==0&&t.drafts_accepted==0);
    auto* m=model.impl();CHECK(m->ver->wait_commit(err));CHECK(cudaDeviceSynchronize()==cudaSuccess);
    im.pp=t.prompt_tok_s;im.tg=t.decode_tok_s;im.logits.resize(model.descriptor().attn.vocab_size);
    CHECK(m->ver->copy_logits(0,im.logits.data()));
    const auto& g=m->g;const size_t per=size_t(g.ssm_state_size)*g.ssm_v_heads*g.ssm_state_size+
        size_t(g.ssm_conv_channels)*(g.ssm_d_conv-1);
    im.state.resize(per*g.n_gdn_layers());
    CHECK(cudaMemcpy(im.state.data(),m->ss.gdn_state,im.state.size()*4,cudaMemcpyDeviceToHost)==cudaSuccess);
    const int64_t cells=ids.size()-1+t.completion_tokens;
    for(int l=0;l<g.n_qsa_layers();++l) {
        const auto& st=m->ss.qsa_states[l];CHECK(st.kv_mode==3);
        const int64_t pages=(cells+3)/4; std::vector<uint16_t> raw(size_t(pages)*g.n_head_kv*4*g.head_dim);
        for(const auto* source:{st.host.k_pool,st.host.v_pool}) {
            CHECK(cudaMemcpy(raw.data(),source,raw.size()*2,cudaMemcpyDefault)==cudaSuccess);
            for(int64_t j=0;j<cells;++j)for(int h=0;h<g.n_head_kv;++h) {
                const size_t off=size_t(((j/4)*g.n_head_kv+h)*4+j%4)*g.head_dim;
                im.kv.insert(im.kv.end(),raw.begin()+off,raw.begin()+off+g.head_dim);
            }
        }
    }
    return im;
}
int main(int argc,char**argv) {
    CHECK(argc>=3);
    runtime::ModelPaths paths;paths.primary_model_path=argv[1];paths.pack_dir=argv[2];paths.tokenizer_dir=paths.pack_dir+"/tokenizer";
    memory::ExecutionPlan plan;plan.context_length=32768;plan.kv_mode=memory::KvMode::HostOnly;
    plan.prefill_batch=512;plan.mtp_spec_tokens=0;
    std::string err;auto model=runtime::GuildModel::load(paths,{},plan,err);if(!model)std::cerr<<err<<'\n';CHECK(model);
    CHECK(model->impl()->batched_prefill && !model->impl()->mtp);
    std::vector<int32_t> base;CHECK(model->tokenizer().encode("The quick brown fox jumps over the lazy dog. A JSON object: {\"a\": 17}.\n",base,false,err));
    std::vector<int> lengths={1,16,127,128,129,512};
    if(argc>3) lengths={std::stoi(argv[3])};
    for(int n:lengths) {
        std::vector<int32_t> ids(n);for(int i=0;i<n;++i)ids[i]=base[i%base.size()];
        std::cout<<"PROMPT "<<n<<std::endl;
        const auto ref=generate(*model,ids,true,1);
        Trace trace_reference;
        auto trace_run=[&](Trace& trace) {
            const int N=int(model->impl()->g.n_embd), L=int(model->impl()->g.n_layers), K=int(model->impl()->ss.k);
            trace.residual.resize(size_t(n)*L*N);trace.weights.resize(size_t(n)*L*K);trace.ids.resize(size_t(n)*L*K);
            model->impl()->batched_prefill->on_layer=[&,N,L,K](int64_t p,int64_t layer,int64_t T,const float* R,
                                                              const int32_t* ids,const float* weights) {
                std::vector<float> r(size_t(T)*N),w(size_t(T)*K);std::vector<int32_t> i(size_t(T)*K);
                CHECK(cudaMemcpy(r.data(),R,r.size()*4,cudaMemcpyDeviceToHost)==cudaSuccess);
                CHECK(cudaMemcpy(w.data(),weights,w.size()*4,cudaMemcpyDeviceToHost)==cudaSuccess);
                CHECK(cudaMemcpy(i.data(),ids,i.size()*4,cudaMemcpyDeviceToHost)==cudaSuccess);
                for(int t=0;t<T;++t) {
                    const size_t off=size_t((p+t)*L+layer);
                    std::copy_n(r.data()+size_t(t)*N,N,trace.residual.data()+off*N);
                    std::copy_n(w.data()+size_t(t)*K,K,trace.weights.data()+off*K);
                    std::copy_n(i.data()+size_t(t)*K,K,trace.ids.data()+off*K);
                }
                trace.rows+=T;
            };
        };
        const std::vector<int> batches=n>512?std::vector<int>{128}:std::vector<int>{1,2,8,16,32,64,128,256,512};
        for(int batch:batches) {
            Trace trace;
            if(n<=129) trace_run(trace);
            const auto actual=generate(*model,ids,false,batch);
            model->impl()->batched_prefill->on_layer={};
            std::cout<<"batch="<<batch<<" pp="<<actual.pp<<" sequential_pp="<<ref.pp<<std::endl;
            compare(ref.logits,actual.logits,"logits");compare(ref.state,actual.state,"GDN/conv");
            CHECK(ref.kv==actual.kv);CHECK(ref.output==actual.output);
            if(n<=129) {
                CHECK(trace.rows==int64_t(n-1)*model->impl()->g.n_layers);
                if(batch==1)trace_reference=std::move(trace);
                else {CHECK(trace_reference.ids==trace.ids);compare(trace_reference.weights,trace.weights,"routing weights");
                    compare(trace_reference.residual,trace.residual,"layer residuals");}
            }
        }
    }
    unsetenv("GUILD_PREFILL_SEQUENTIAL");
    auto& bp=*model->impl()->batched_prefill;
    std::vector<int64_t> ids={9419,11,220,17};
    bp.should_stop=[] {return true;};CHECK(!bp.run(ids.data(),4,0,err)&&err=="cancelled");
    bp.should_stop={};CHECK(!bp.run(ids.data(),4,32767,err));
    std::cout<<"ornith_prefill_test: actual model logits, state, KV and greedy parity PASS\n";
}
