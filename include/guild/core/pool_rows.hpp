// include/guild/core/pool_rows.hpp - the CPU expert pool's rows of the formats Qwen3.8-Flash-Next's experts mostly
// take (IQ3_S gate/up repacked to NL4, IQ4_NL down, and layer 2's IQ4_XS gate/up) and Drama 3's and Ornith's (Q4_K
// gate/up, Q4_K and Q6_K down) on this library's AVX-512BW kernels, two rows to a register: glm5cpu::nl4_rows,
// cpu_moe::iq4nl_rows512, cpu_moe::iq4xs_rows512 and cpu_moe::kq_rows512.  Every (row, token) is bitwise what the
// library's table computes - all reproduce ggml-cpu's one-token dot, as the guild-kernels-cpu rows do in ggml's order
// (GUILD_IQ_GGML_ORDER, the default) - so only the speed changes (kq_rows512_test compares the two tables).  That
// table is the only one they are proven against, so it is the only one wrapped.
#pragma once

#include "guild/abi/cpu_expert_abi.h"

namespace guild::core {

/// `base` with its gate/up of repacked IQ3_S (kIq3sNl4Type), IQ4_XS (GUILD_POOL_IQ4XS) and Q4_K (GUILD_KQ_ROWS512)
/// rows and its down of IQ4_NL rows (an even number of 32-value blocks) and Q4_K / Q6_K rows (GUILD_KQ_ROWS512) on
/// the AVX-512BW rows, when the CPU has AVX-512 F, BW, DQ and VL; every other format through `base`.  Only when
/// `base` is the bundled guild-kernels-cpu table (all four of its functions guild_cpu_backend_table_v2()'s, the
/// table kq_rows512_test compares these rows with) and `gate_up_interleaved` is false (the layout's gate/up rows are
/// [gate | up], the only layout those rows read): any other table - a GEXEC table (gexec_expert.cpp, fused or
/// interleaved), an external CPU package's - comes back unchanged, as does every table when GUILD_POOL_ROWS512=0 or
/// GUILD_IQ_GGML_ORDER=0.  Each table returned leaves the other formats to its own `base`, so the tables of models
/// loaded side by side stay independent.  Thread-safe.
guild_cpu_expert_ext_v1_t pool_rows512(const guild_cpu_expert_ext_v1_t& base, bool gate_up_interleaved);

namespace detail {

/// How many distinct bases (by gate_up and down) the wrappers can serve at once.
inline constexpr int kPoolRowsSlots = 8;

/// pool_rows512 without its check of which table `base` is (for it and pool_rows_slots_test only: the rows are
/// proven against the library's table alone).  The first base given takes slot 0, each new one the next slot, and a
/// base with the gate_up and down of one already given reuses its slot.  With every slot taken `base` comes back
/// unchanged, with one line on stderr; a table this returned comes back as it is.
guild_cpu_expert_ext_v1_t pool_rows512_slot(const guild_cpu_expert_ext_v1_t& base);

} // namespace detail

} // namespace guild::core
