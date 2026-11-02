// include/guild/core/pool_rows.hpp - the CPU expert pool's rows of the two formats Qwen3.8-Flash-Next's experts mostly
// take (IQ3_S gate/up repacked to NL4, IQ4_NL down) on this library's AVX-512BW kernels, two rows to a register:
// glm5cpu::nl4_rows and cpu_moe::iq4nl_rows512.  Every (row, token) is bitwise what the table it wraps computes -
// both reproduce ggml-cpu's one-token dot, as the guild-kernels-cpu rows do in ggml's order (GUILD_IQ_GGML_ORDER,
// the default) - so only the speed changes.
#pragma once

#include "guild/abi/cpu_expert_abi.h"

namespace guild::core {

/// `base` with its gate/up of repacked IQ3_S (kIq3sNl4Type) rows and its down of IQ4_NL rows (an even number of
/// 32-value blocks) on the AVX-512BW rows, when the CPU has AVX-512 F, BW, DQ and VL; every other format, and every
/// row when GUILD_POOL_ROWS512=0 or GUILD_IQ_GGML_ORDER=0, through `base`.  Process-wide: the last `base` given is
/// the one wrapped.
guild_cpu_expert_ext_v1_t pool_rows512(const guild_cpu_expert_ext_v1_t& base);

} // namespace guild::core
