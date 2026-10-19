// Copyright (c) Guild Inference Contributors. MIT License.
// Phase 5: GEXEC Hardware-Specialized CPU Expert Kernel Providers.
#pragma once

#include "guild/abi/cpu_backend_abi.h"

namespace guild::kernels::cpu {

inline constexpr const char* kGExecFusedAvx2ImplId = "cpu.expert_pool.gexec_fused_avx2";
inline constexpr const char* kGExecFusedAvx512ImplId = "cpu.expert_pool.gexec_fused_avx512";
inline constexpr const char* kGExecInterleavedAvx2ImplId = "cpu.expert_pool.gexec_interleaved_avx2";
inline constexpr const char* kGExecInterleavedAvx512ImplId = "cpu.expert_pool.gexec_interleaved_avx512";
inline constexpr uint32_t kGExecExpertImplVersion = GUILD_KERNEL_ABI_MAKE_VERSION(5, 0, 0);

// Returns the v2 CPU backend extension table for GEXEC fused 2-row/4-row expert execution
// over the standard PortableContiguous [gate(n_ff) | up(n_ff) | down(n_embd)] layout (AVX2).
extern "C" const guild_cpu_backend_ext_v2_t* guild_cpu_gexec_fused_avx2_table_v2() noexcept;

// Returns the v2 CPU backend extension table for GEXEC fused 2-row/4-row expert execution
// over the standard PortableContiguous [gate(n_ff) | up(n_ff) | down(n_embd)] layout (AVX-512BW).
extern "C" const guild_cpu_backend_ext_v2_t* guild_cpu_gexec_fused_avx512_table_v2() noexcept;

// Returns the v2 CPU backend extension table for GEXEC fused 2-row/4-row expert execution
// over the GateUpRowInterleaved [gate_0, up_0, ..., gate_{n_ff-1}, up_{n_ff-1} | down(n_embd)] layout (AVX2).
extern "C" const guild_cpu_backend_ext_v2_t* guild_cpu_gexec_interleaved_avx2_table_v2() noexcept;

// Returns the v2 CPU backend extension table for GEXEC fused 2-row/4-row expert execution
// over the GateUpRowInterleaved [gate_0, up_0, ..., gate_{n_ff-1}, up_{n_ff-1} | down(n_embd)] layout (AVX-512BW).
extern "C" const guild_cpu_backend_ext_v2_t* guild_cpu_gexec_interleaved_avx512_table_v2() noexcept;

// Returns true if the given impl_id expects GateUpRowInterleaved expert blobs.
bool is_gexec_interleaved_impl(const char* impl_id) noexcept;

// Returns true if the given impl_id is any GEXEC-specialized CPU expert implementation.
bool is_gexec_optimized_cpu_impl(const char* impl_id) noexcept;

} // namespace guild::kernels::cpu
