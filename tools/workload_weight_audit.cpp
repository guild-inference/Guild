// Compare the BF16 pack tensors that NativeDense intentionally does not replace
// with dequantized GGUF values. Diagnostic only; never changes model weights.
#include "guild/artifact/gguf_reader.hpp"
#include "ggml.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

int main(int argc,char** argv) try {
    if(argc!=3)throw std::invalid_argument("usage: workload_weight_audit MODEL PACK");
    ggml_cpu_init();
    guild::GgufModel model(guild::gguf_split_paths(argv[1]));
    std::ifstream index(std::string(argv[2])+"/index.txt"), dense(std::string(argv[2])+"/dense.bin",std::ios::binary);
    if(!index||!dense)throw std::runtime_error("cannot open pack");
    std::cout<<"tensor,gguf_type,elements,max_abs,relative_l2,changed_floats,bf16_rounding_mismatches\n"<<std::setprecision(12);
    std::string line;
    while(std::getline(index,line)) {
        if(line.empty()||line[0]=='#')continue;
        std::istringstream in(line);std::string name;int file,kind;
        uint64_t off,bytes,dst,dst_bytes,n0,n1;
        if(!(in>>name>>file>>kind>>off>>bytes>>dst>>dst_bytes>>n0>>n1))throw std::runtime_error("bad pack row");
        if(file!=0||kind!=4)continue;
        size_t shard=0;const auto* tensor=model.find(name,&shard);
        if(!tensor)throw std::runtime_error("tensor absent: "+name);
        const size_t rows=std::max<uint64_t>(1,n1), count=n0*rows;
        if(bytes!=count*2)throw std::runtime_error("not raw BF16: "+name);
        std::vector<uint16_t> packed(count);
        dense.seekg(off);dense.read(reinterpret_cast<char*>(packed.data()),bytes);
        if(!dense)throw std::runtime_error("short pack read");
        const auto* data=model.shard(shard).tensor_data(*tensor);
        const auto type=ggml_type(tensor->type);
        const auto* traits=ggml_get_type_traits(type);
        std::vector<float> reference(count);
        for(size_t row=0;row<rows;++row) {
            if(type==GGML_TYPE_F32)std::memcpy(reference.data()+row*n0,data+row*n0*4,n0*4);
            else if(traits->to_float)traits->to_float(data+row*ggml_row_size(type,n0),reference.data()+row*n0,n0);
            else throw std::runtime_error("no dequantizer: "+name);
        }
        double maximum=0,err2=0,ref2=0;size_t changed=0,round_diff=0;
        for(size_t i=0;i<count;++i) {
            uint32_t bits=uint32_t(packed[i])<<16;float value;std::memcpy(&value,&bits,4);
            if(!std::isfinite(value)||!std::isfinite(reference[i]))throw std::runtime_error("nonfinite weight");
            const double d=double(value)-reference[i];maximum=std::max(maximum,std::abs(d));err2+=d*d;ref2+=double(reference[i])*reference[i];
            changed+=std::memcmp(&value,&reference[i],4)!=0;
            uint32_t rbits;std::memcpy(&rbits,&reference[i],4);
            const uint16_t expected=uint16_t((rbits+0x7fffu+((rbits>>16)&1))>>16);
            round_diff+=expected!=packed[i];
        }
        std::cout<<name<<','<<int(type)<<','<<count<<','<<maximum<<','<<std::sqrt(err2/(ref2+1e-300))<<','<<changed<<','<<round_diff<<'\n';
    }
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
