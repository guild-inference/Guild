// Optional development oracle, linked to the user's installed llama.cpp.
// Identical numeric IDs, greedy sampling, no template/tokenizer or speculative path.
#include "llama.h"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <cstdio>
using Clock=std::chrono::steady_clock;
double ms(Clock::time_point a){return std::chrono::duration<double,std::milli>(Clock::now()-a).count();}
int main(int argc,char**argv) try {
    if(argc!=8)throw std::invalid_argument("usage: reference MODEL IDS CONTEXT GENERATED NGL THREADS OUT_PREFIX");
    std::ifstream f(argv[2]);std::vector<llama_token> ids;int id;while(f>>id)ids.push_back(id);
    if(ids.empty())throw std::invalid_argument("empty IDs file");
    llama_log_set([](ggml_log_level level,const char* text,void*) {
        if (level==GGML_LOG_LEVEL_ERROR) std::fputs(text,stderr);
    },nullptr);
    llama_backend_init();auto mp=llama_model_default_params();mp.n_gpu_layers=std::stoi(argv[5]);
    const auto load_start=Clock::now();auto* model=llama_model_load_from_file(argv[1],mp);if(!model)throw std::runtime_error("model load failed");
    auto cp=llama_context_default_params();cp.n_ctx=std::stoi(argv[3]);cp.n_batch=512;cp.n_ubatch=512;
    cp.n_threads=std::stoi(argv[6]);cp.n_threads_batch=cp.n_threads;cp.type_k=GGML_TYPE_F16;cp.type_v=GGML_TYPE_F16;
    auto* ctx=llama_init_from_model(model,cp);if(!ctx)throw std::runtime_error("context load failed");
    const double load_ms=ms(load_start);const int generated=std::stoi(argv[4]);auto* vocab=llama_model_get_vocab(model);
    const int n_vocab=llama_vocab_n_tokens(vocab);std::ofstream csv(std::string(argv[7])+".csv");
    csv<<"run,prompt,generated,prefill_ms,decode_ms,load_ms\n";
    for(int run=-1;run<3;++run) {
        llama_memory_clear(llama_get_memory(ctx),true);
        llama_batch b=llama_batch_init(512,0,1);const auto start=Clock::now();
        for(size_t off=0;off<ids.size();off+=512) {
            b.n_tokens=int(std::min<size_t>(512,ids.size()-off));
            for(int j=0;j<b.n_tokens;++j){b.token[j]=ids[off+j];b.pos[j]=int(off+j);b.n_seq_id[j]=1;b.seq_id[j][0]=0;b.logits[j]=(off+j+1==ids.size());}
            if(llama_decode(ctx,b))throw std::runtime_error("prefill failed");
        }
        llama_synchronize(ctx);const double pp=ms(start);std::vector<llama_token> output;
        if(run==0){const auto* logits=llama_get_logits_ith(ctx,-1);std::ofstream o(std::string(argv[7])+".f32",std::ios::binary);
            o.write(reinterpret_cast<const char*>(logits),size_t(n_vocab)*4);}
        const auto decode_start=Clock::now();
        for(int j=0;j<generated;++j) {
            const auto* logits=llama_get_logits_ith(ctx,-1);const int next=int(std::max_element(logits,logits+n_vocab)-logits);
            output.push_back(next);if(llama_vocab_is_eog(vocab,next)||j+1==generated)break;
            b.n_tokens=1;b.token[0]=next;b.pos[0]=int(ids.size())+j;b.n_seq_id[0]=1;b.seq_id[0][0]=0;b.logits[0]=1;
            if(llama_decode(ctx,b))throw std::runtime_error("decode failed");
        }
        llama_synchronize(ctx);const double tg=ms(decode_start);
        if(run>=0){csv<<run<<','<<ids.size()<<','<<output.size()<<','<<pp<<','<<tg<<','<<load_ms<<'\n';
            std::ofstream o(std::string(argv[7])+".ids");for(auto t:output)o<<t<<' ';}
        llama_batch_free(b);
    }
    llama_free(ctx);llama_model_free(model);llama_backend_free();return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
