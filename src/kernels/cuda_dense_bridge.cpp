// Orchestration-only adapters: no CUDA kernels or CPU numerical fallback.
#include "guild/kernels/cuda_dense.hpp"
#include "guild/kernels/bf16_gemv.hpp"
#include "guild/kernels/native_mmvq.hpp"
#include "guild/kernels/iq_kernels.hpp"
#include "guild/artifact/gguf_reader.hpp"
#include <limits>
#include <stdexcept>

#if GUILD_BUNDLED_CUDA_DENSE
extern "C" const guild_cuda_dense_ext_v1_t* guild_cuda_dense_table_v1();
#endif
namespace guild::kernels {
namespace {
thread_local const guild_cuda_dense_ext_v1_t* active=nullptr;
thread_local bool exact_multi=true;
const guild_cuda_dense_ext_v1_t& backend() {
    if(active)return *active;
#if GUILD_BUNDLED_CUDA_DENSE
    return *guild_cuda_dense_table_v1();
#else
    throw std::runtime_error("CUDA BF16/native-MMVQ provider missing; bundled dense arithmetic is disabled; install a CDD1 CUDA .gkernel");
#endif
}
guild_exec_context_v1_t context(void* stream) {
    guild_exec_context_v1_t c{};c.struct_size=sizeof(c);c.abi_version=GUILD_KERNEL_ABI_VERSION_V1;
    c.backend=GUILD_BACKEND_CUDA;c.device_id=0;c.device_stream=stream;c.sync_mode=GUILD_SYNC_STREAM_ORDERED;return c;
}
void checked(guild_status_t result,const guild_error_info_t& error) {
    if(result==GUILD_STATUS_OK)return;
    if(result==GUILD_STATUS_ERR_INVALID_ARGUMENT || result==GUILD_STATUS_ERR_ABI_MISMATCH)
        throw std::invalid_argument(error.message);
    throw std::runtime_error(error.message[0]?error.message:"CUDA dense backend execution failed");
}
void projection(uint32_t mode,const void* x,const uint16_t* w,float* y,int64_t k,int64_t n,
                int rows,int64_t ldx,int64_t ldy,int threads,void* stream) {
    guild_cuda_bf16_projection_v1_t p{};p.struct_size=sizeof(p);p.mode=mode;p.x=x;p.w=w;p.y=y;
    p.n_in=k;p.n_out=n;p.rows=rows;p.ldx=ldx;p.ldy=ldy;p.threads_per_row=threads;
    const auto c=context(stream);guild_error_info_t e{};
    checked(backend().bf16(&c,&p,&e),e);
}
}
ScopedCudaDense::ScopedCudaDense(const ResolvedCudaDense& binding):previous_(active){active=&binding.table;}
ScopedCudaDense::~ScopedCudaDense(){active=previous_;}
void bf16_gemv(const uint16_t* x,const uint16_t* w,float* y,int64_t k,int64_t n,void* s) {
    if(k<=0 || n<=0)return;
    projection(GUILD_CUDA_BF16_ACT_AUTO_V1,x,w,y,k,n,1,k,n,0,s);
}
void bf16_gemv_split(const uint16_t* x,const uint16_t* w,float* y,int64_t k,int64_t n,int threads,void* s) {
    if(k<=0 || n<=0)return;
    projection(GUILD_CUDA_BF16_ACT_SPLIT_V1,x,w,y,k,n,1,k,n,threads,s);
}
void bf16_gemv_fp32_mmvf(const float* x,const uint16_t* w,float* y,int64_t k,int64_t n,void* s) {
    projection(GUILD_CUDA_BF16_F32_SINGLE_V1,x,w,y,k,n,1,k,n,0,s);
}
void bf16_gemv_fp32_mmvf_multi(const float* x,int64_t ldx,const uint16_t* w,float* y,int64_t ldy,int64_t k,int64_t n,int rows,void* s) {
    projection(GUILD_CUDA_BF16_F32_MULTI_V1,x,w,y,k,n,rows,ldx,ldy,0,s);
}
void bf16_gemv_fp32_mmvf_cols(const float* x,const uint16_t* w,float* y,int64_t k,int64_t n,int rows,void* s) {
    if(rows<=0)return;
    projection(GUILD_CUDA_BF16_F32_COLS_V1,x,w,y,k,n,rows,k,n,0,s);
}
std::size_t native_q8_1_bytes(int k,int rows) {
    if(k<=0 || k%32 || rows<1 || rows>512)throw std::invalid_argument("native Q8_1 invalid shape");
    return std::size_t(k/32)*36*std::size_t(rows);
}
void native_mmvq_set_multi_exact(bool exact){exact_multi=exact;}
bool native_mmvq_multi_exact(){return exact_multi;}
void native_quantize_q8_1(const float* x,void* q,int k,int rows,void* stream) {
    guild_cuda_quantize_q8_1_v1_t p{};p.struct_size=sizeof(p);p.x=x;p.q=q;p.n_in=k;p.rows=rows;p.q_bytes=native_q8_1_bytes(k,rows);
    const auto c=context(stream);guild_error_info_t e{};checked(backend().quantize_q8_1(&c,&p,&e),e);
}
void native_swiglu_quantize_q8_1(const float* x,const float* up,void* q,int k,int rows,void* stream) {
    guild_cuda_quantize_q8_1_v1_t p{};p.struct_size=sizeof(p);p.x=x;p.up=up;p.q=q;p.n_in=k;p.rows=rows;p.q_bytes=native_q8_1_bytes(k,rows);
    const auto c=context(stream);guild_error_info_t e{};checked(backend().quantize_q8_1(&c,&p,&e),e);
}
void native_mmvq(int type,const void* w,const void* q,float* y,int k,int n,int rows,void* stream) {
    guild_cuda_mmvq_v1_t p{};p.struct_size=sizeof(p);p.exact_multi=exact_multi;p.weight_dtype=type;
    p.n_in=k;p.n_out=n;p.rows=rows;p.weights=w;p.q=q;p.y=y;
    const auto c=context(stream);guild_error_info_t e{};checked(backend().mmvq(&c,&p,&e),e);
}
// These are GGUF storage/shape checks, not GPU arithmetic or hidden fallbacks.
bool native_mmvq_supported(int type) noexcept {
    switch(type){case 2:case 6:case 7:case 8:case 11:case 12:case 13:case 14:case 20:case 23:case 42:
        case 16:case 17:case 18:case 21:case 22:case 29:return true;default:return false;}
}
std::size_t native_mmvq_weight_bytes(int type,int k,int n) {
    int block=0,bytes=0;
    switch(type){case 2:case 20:block=32;bytes=18;break;case 6:block=32;bytes=22;break;case 7:block=32;bytes=24;break;
    case 8:block=32;bytes=34;break;case 11:block=256;bytes=110;break;case 12:block=256;bytes=144;break;
    case 13:block=256;bytes=176;break;case 14:block=256;bytes=210;break;case 23:block=256;bytes=136;break;
    case 42:block=64;bytes=18;break;case 16:case 17:case 18:case 21:case 22:case 29:block=256;bytes=int(iq_row_bytes(type,256));break;
    default:throw std::invalid_argument("unsupported native MMVQ GGML type");}
    if(k<=0 || k%block || n<=0)throw std::invalid_argument("native MMVQ invalid weight shape");
    const auto row=std::size_t(k/block)*bytes;
    if(row>std::numeric_limits<std::size_t>::max()/std::size_t(n))throw std::length_error("native MMVQ byte overflow");
    return row*std::size_t(n);
}
#define GUILD_MMVQ_ADAPTER(name,type) \
void native_##name##_mmvq(const void* w,const void* q,float* y,int k,int n,int rows,void* s){native_mmvq(type,w,q,y,k,n,rows,s);} \
void native_##name##_f32(const void* w,const float* x,void* q,float* y,int k,int n,int rows,void* s){ \
    native_quantize_q8_1(x,q,k,rows,s);native_mmvq(type,w,q,y,k,n,rows,s);}
GUILD_MMVQ_ADAPTER(q4_0,2)
GUILD_MMVQ_ADAPTER(q5_0,6)
GUILD_MMVQ_ADAPTER(q8_0,8)
GUILD_MMVQ_ADAPTER(q3_k,11)
GUILD_MMVQ_ADAPTER(q4_k,12)
GUILD_MMVQ_ADAPTER(q5_k,13)
GUILD_MMVQ_ADAPTER(q6_k,14)
GUILD_MMVQ_ADAPTER(iq4_nl,20)
GUILD_MMVQ_ADAPTER(iq4_xs,23)
GUILD_MMVQ_ADAPTER(q2_0,42)
#undef GUILD_MMVQ_ADAPTER
}
