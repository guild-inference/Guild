// No arithmetic: fail-closed legacy entry points for an external-only Core.
#include "guild/kernels/cpu/expert_layout.hpp"
#include "guild/kernels/cpu/kq_avx2.hpp"
#include "guild/kernels/cpu/kq_avx1.hpp"
#include <stdexcept>
namespace guild::kernels::cpu {
namespace {
[[noreturn]] void missing() {
    throw std::runtime_error("bundled CPU arithmetic disabled: bind a complete CPU .gkernel");
}
} // namespace
void act_quant_q8_1(const float*, int, ActQ&) {
    missing();
}
void act_quant_q8_1_avx2(const float*, int, ActQ&) {
    missing();
}
void act_quant_any(const float*, int, ActQ&) {
    missing();
}
void q2_rows_any(const uint8_t*, size_t, int, const ActQ* const*, int, float* const*, int, int) {
    missing();
}
void q2_0_gguf_rows_multi(const uint8_t*, size_t, int, const ActQ* const*, int, float* const*, int, int) {
    missing();
}
void q2_0_gguf_rows_multi_avx2(const uint8_t*, size_t, int, const ActQ* const*, int, float* const*, int, int) {
    missing();
}
void s2_expert_vnni_q(const uint8_t*, const ActQ&, float*, ExpertScratch&) {
    missing();
}
void s2_expert_gu_rows(const uint8_t*, const ActQ&, float*, int, int) {
    missing();
}
void s2_expert_down_rows(const uint8_t*, const ActQ&, float*, int, int) {
    missing();
}
void s2_expert_gu_rows_multi(const uint8_t*, const ActQ* const*, int, float* const*, int, int) {
    missing();
}
void s2_expert_down_rows_multi(const uint8_t*, const ActQ* const*, int, float* const*, int, int) {
    missing();
}
void bf16_rows_dot(const uint16_t*, int, int, const float*, float*) {
    missing();
}
void bf16_rows_dot_multi(const uint16_t*, int, int, const float*, int, float*) {
    missing();
}
void bf16_rows_dot_multi_avx1(const uint16_t*, int, int, const float*, int, float*) {
    missing();
}
void swiglu(const float*, const float*, float*, int) {
    missing();
}
bool expert_oracle_q8_0_enabled() {
    return false;
}
void expert_set_oracle_q8_0(bool) {
    missing();
}
#if defined(GUILD_NATIVE_EXPERTS)
bool native_experts_available() noexcept {
    return true;
}
bool native_fmt(int, int, int64_t, int64_t, NativeFmt&, std::string& error) {
    error = "external CPU format callback required";
    return false;
}
void native_quant_act(const NativeFmt&, const float*, void*) {
    missing();
}
void native_quant_h(const NativeFmt&, const float*, void*) {
    missing();
}
void native_gu_rows(const NativeFmt&, const uint8_t*, const void* const*, int, float* const*, int, int) {
    missing();
}
void native_down_rows(const NativeFmt&, const uint8_t*, const void* const*, int, float* const*, int, int) {
    missing();
}
#endif
} // namespace guild::kernels::cpu
