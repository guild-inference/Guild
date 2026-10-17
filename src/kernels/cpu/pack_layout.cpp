// Guild/Strata MIT: expert pack interpretation and file offsets stay in Core.
// Arithmetic and feature-dispatch implementations belong to guild-kernels-cpu.
#include "guild/kernels/cpu/expert_layout.hpp"
#include <cstdlib>
#include <fstream>
#include <sstream>

namespace guild::kernels::cpu {
namespace { ExpertLayout layout; }
const ExpertLayout& expert_layout(){return layout;}

#if !defined(GUILD_NATIVE_EXPERTS)
bool native_experts_available() noexcept{return false;}
bool native_fmt(int,int,int64_t,int64_t,NativeFmt&,std::string& error){error="built without native experts";return false;}
void native_quant_act(const NativeFmt&,const float*,void*){std::abort();}
void native_quant_h(const NativeFmt&,const float*,void*){std::abort();}
void native_gu_rows(const NativeFmt&,const uint8_t*,const void* const*,int,float* const*,int,int){std::abort();}
void native_down_rows(const NativeFmt&,const uint8_t*,const void* const*,int,float* const*,int,int){std::abort();}
#endif

bool expert_layout_load(const std::string& pack_dir,int64_t layers,int64_t experts,std::string& error,
                         int64_t embedding,int64_t hidden,const guild_cpu_backend_ext_v2_t* backend){
    ExpertLayout next;
    next.n_layers=layers;next.n_expert=experts;
    std::ifstream input(pack_dir+"/native_experts.txt");
    if(!input){next.total=uint64_t(layers)*uint64_t(experts)*uint64_t(BLOB);layout=next;return true;}
#if !defined(GUILD_NATIVE_EXPERTS)
    error="this pack has native (IQ) experts but the engine was built without GUILD_NATIVE_EXPERTS";
    return false;
#else
    next.native=true;next.fmt.resize(size_t(layers));
    next.offset.assign(size_t(layers),~0ull);next.bytes.assign(size_t(layers),0);next.max_blob=0;
    std::string line;
    while(std::getline(input,line)){
        if(line.empty() || line[0]=='#'){
            if(!line.empty()){
                const std::string tag=line.rfind("# strata native experts v",0)==0?
                    "# strata native experts v":"# guild native experts v";
                if(line.compare(0,tag.size(),tag)==0){
                    next.version=std::atoi(line.c_str()+tag.size());
                    if(next.version>kExpertLayoutVersion){
                        error="native_experts.txt is v"+std::to_string(next.version)+"; this engine reads up to v"+
                            std::to_string(kExpertLayoutVersion)+" (the pack was written by a newer tools/iq_pack.py: update the engine, or repack with this one)";
                        return false;
                    }
                }
                const size_t at=line.find("(n_expert ");
                if(at!=std::string::npos)next.n_expert=std::atoll(line.c_str()+at+10);
            }
            continue;
        }
        std::istringstream row(line);
        long long layer=-1,gu=-1,down=-1;
        unsigned long long offset=0,blob=0,gate_offset=0,up_offset=0,down_offset=0;
        if(!(row>>layer>>gu>>down>>offset>>blob) || layer<0){error="native_experts.txt: a malformed line: "+line;return false;}
        if(layer>=layers)continue;
        NativeFmt format;
        if(backend && backend->format){
            guild_error_info_t info{};
            if(backend->format(int(gu),int(down),embedding,hidden,&format,&info)!=GUILD_STATUS_OK){error=info.message;return false;}
        }else if(!native_fmt(int(gu),int(down),embedding,hidden,format,error))return false;
        if(format.bytes!=blob){error="native_experts.txt: layer "+std::to_string(layer)+" blob is "+std::to_string(blob)+
            " B but its formats make "+std::to_string(format.bytes);return false;}
        if(row>>gate_offset>>up_offset>>down_offset){
            if(next.gguf_off.empty())next.gguf_off.assign(size_t(3*layers),0);
            next.gguf_off[size_t(3*layer)]=gate_offset;next.gguf_off[size_t(3*layer+1)]=up_offset;
            next.gguf_off[size_t(3*layer+2)]=down_offset;
            std::string file;
            if(row>>file){
                if(next.gguf_file.empty())next.gguf_file.assign(size_t(3*layers),std::string());
                std::vector<std::string> parts;
                for(size_t from=0;;){const size_t comma=file.find(',',from);
                    parts.push_back(file.substr(from,comma==std::string::npos?comma:comma-from));
                    if(comma==std::string::npos)break;from=comma+1;}
                if(parts.size()==1){const std::string one=parts[0];parts.assign(3,one);}
                std::string extra;
                if(parts.size()!=3 || (row>>extra)){error="native_experts.txt: layer "+std::to_string(layer)+
                    ": the shard column is one name or gate,up,down, not: "+line;return false;}
                for(size_t role=0;role<3;++role)next.gguf_file[size_t(3*layer)+role]=parts[role];
            }
        }
        next.fmt[size_t(layer)]=format;next.offset[size_t(layer)]=offset;next.bytes[size_t(layer)]=blob;
        if(blob>next.max_blob)next.max_blob=blob;
    }
    uint64_t at=0;
    for(int64_t layer=0;layer<layers;++layer){
        if(next.offset[size_t(layer)]!=at){error="native_experts.txt: layer "+std::to_string(layer)+" is missing or not contiguous";return false;}
        at+=next.bytes[size_t(layer)]*uint64_t(next.n_expert);
    }
    next.total=at;layout=next;return true;
#endif
}
}
