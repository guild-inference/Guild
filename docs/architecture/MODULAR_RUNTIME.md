# Modular Runtime Architecture

Guild separates inference execution into three strictly decoupled layers connected by a versioned C-compatible ABI (`include/guild/abi/kernel_abi.h`):

1. **Guild Core (`src/runtime/`, `src/memory/`, `src/plan/`, `src/spec/`, `src/server/`)**
   - GGUF/pack loading, memory hierarchy planning (VRAM, pinned host RAM, mmap/file expert tiers), KV/recurrent state lifecycle, speculative decoding orchestration (`SpeculativeController`, `SuffixDrafter`, `MtpDrafter`), tokenization, sampling, HTTP/OpenAI/Anthropic APIs, and CLI.
   - Completely independent of model-specific layer equations or hardware-specific kernel intrinsics.

2. **Model Architecture Runners (`include/guild/model/runner.hpp`, `src/model/runner.cpp`)**
   - Define how a model family (`qwen4exp`, `qwen35moe`, or future architectures) maps GGUF metadata and tensors into `ModelDescriptor`, validates architectural invariants (such as mandatory PLE tables or hybrid recurrent/full-attention intervals), declares required kernel operations (`required_ops`), and binds prefill/decode execution graphs.
   - Registered in `guild::model::ModelRunnerRegistry` (`runner.qwen4exp.v1`, `runner.qwen35moe.v1`, or external ABI v1 runners).

3. **Hardware Kernels (`include/guild/kernels/registry.hpp`, `src/kernels/registry.cpp`, `src/core/`, `src/engine/`)**
   - Architecture- and device-specific compute implementations for CPU (`AVX2`, `AVX-512F/BW/DQ/VL`, `AVX-512 VNNI`, `NEON`), NVIDIA CUDA (`SM 6.1` Pascal through `SM 9.0`), and AMD HIP (`gfx1100+`).
   - Registered in `guild::kernels::KernelRegistry` with explicit operation kinds (`guild_op_kind_t`), hardware/ISA/SM requirements, supported data types, batch ranges, scratch memory calculators, and numerical parity flags.

---

## Execution Lifecycle & Zero-Overhead Hot Path

To preserve single-token decode latency and batched prefill throughput, **neither virtual method dispatch nor hash-table/registry lookups occur inside layer or token loops**:

1. **Startup / Built-In Registration**
   - `KernelRegistry::instance().register_builtin_kernels()` registers built-in CPU (`builtin.guild-kernels-cpu`) and GPU (`builtin.guild-kernels-cuda` / `builtin.guild-kernels-amd`) kernel descriptors.
   - `ModelRunnerRegistry::instance().register_builtin_runners()` registers `Qwen4ExpRunner` (`builtin.guild-kernels-qwen`) and `Qwen35MoERunner` (`builtin.guild-kernels-ornith`).

2. **Model Load (`GuildModel::load`)**
   - Reads the GGUF header and resolves the architecture runner via `ModelRunnerRegistry::find_by_architecture(arch)`.
   - Invokes `runner->validate_model(descriptor)` to enforce architectural contracts before allocating device/host buffers.
   - Queries `KernelRegistry::discover_capabilities()` and verifies that every `guild_op_kind_t` required by `runner->required_ops(descriptor, has_gpu_backend)` has a compatible kernel implementation on the active hardware (`verify_runner_kernel_requirements`).

3. **Session Initialization (`GuildSession` / `BatchedPrefill::init` / `Verifier`)**
   - Acquires `SessionRunnerLease` and `SessionKernelLease`, preventing any runner or kernel package from being unregistered or unloaded while an inference session is active (`GUILD_STATUS_ERR_ACTIVE_SESSION_BUSY`).
   - `BatchedPrefill::init()` calls `KernelRegistry::resolve_prefill_kernels()` **once**, storing raw C/CUDA function pointers in `kernels::ResolvedPrefillKernels` (`rmsnorm_bf16`, `gemm_bf16_w_bf16`, `qsa_rope_2d`, `gdn_conv_silu`, `gdn_recurrence`, `silu_mul_bf16`, `shared_gate_scale_add`, `add_inplace_bf16`, `embed_bf16_tokens`, `scale_bf16`, `add_rms_norm_bf16`).
   - `Verifier` captures pre-bound CUDA graphs for single-token and speculative verify windows.

4. **Token / Layer Inner Loops**
   - `BatchedPrefill::process_chunk()` invokes the pre-bound function pointers in `ResolvedPrefillKernels` and the pre-bound `ExpertPool::run_split_multi_native(..., parallel_quant = true)` workers directly.
   - `Verifier` launches captured CUDA graphs directly.

---

## Measured Performance Comparison vs Baseline (`40a4612`)

Measured on dual Intel Xeon Gold 6130 (32C/64T, AVX-512F/BW/DQ/VL, no VNNI) + NVIDIA GeForce GTX 1070 (8 GiB VRAM, SM 6.1), `Ornith-1.5-35B-Q4_K_M.gguf`, `prompt=512`, `generate=128`, 3 runs:

| Configuration | Pre-Modularization (`40a4612`) Prefill | Post-Modularization Prefill | Pre-Modularization (`40a4612`) Decode | Post-Modularization Decode |
|---|---:|---:|---:|---:|
| Default NUMA (`context=32768`, `--sequential`) | `37.68 tok/s` | `38.76 tok/s` | `36.78 tok/s` | `37.90 tok/s` |
| Default NUMA (`context=32768`, `--batch 128`) | `133.19 tok/s` | `129.67 tok/s` | `38.77 tok/s` | `38.72 tok/s` |
| NUMA-Interleaved (`context=2048`, `--batch 128`, `numactl --interleave=0,1`) | `131.87 tok/s` | `132.36 tok/s` | `42.94 tok/s` | `43.51 tok/s` |

Qwen3.8-Flash-Next (`unsloth-UD-IQ4_XS`) parity via `qwen_runtime_parity_test`:
- Reference logits: `bitwise=1`, `max_abs=0`, `relative_l2=0`
- Greedy continuations match reference token-for-token.

---

## Follow-Up Research Plan: `llama.cpp` Prefill Optimizations

Reference checkout inspected: `/home/ubuntu/llama.cpp` at commit `11fe02151f79c41d0d4af7da708755d73b9c0da6` (`src/models/qwen35moe.cpp`, `src/models/delta-net-base.cpp`, `ggml/src/ggml-cuda/`).

### Key Findings from `llama.cpp` Inspection

1. **Chunked Gated DeltaNet (`llm_build_delta_net_base::build_delta_net_chunking` in `src/models/delta-net-base.cpp`)**
   - For multi-token prefill (`n_tokens > 1`), `llama.cpp` replaces sequential token-by-token recurrence with a chunked WY-style intra-chunk parallel algorithm (`CS = 64` for standard GDN, `CS = 16` for KDA):
     - Pads the sequence to a multiple of `CS`, computes cumulative decay `g_cs = cumsum(g)`, constructs the lower-triangular decay mask `exp(g_cs_j - g_cs_i)`, and solves `(I + tril(K * K_b^T * decay))^-1` via `ggml_solve_tri`.
     - Computes intra-chunk outputs via batched matrix multiplications (`ggml_mul_mat`) and updates the `[S_v, S_v, H_v]` recurrent state once per 64-token chunk instead of once per token.
   - **Applicability to Guild (`GUILD_OP_GDN_RECURRENCE`)**:
     - In Guild's `Ornith-1.5-35B` prefill (`B=128`), `gdn_recurrence_batched` currently accounts for only `5.21 ms` across all 30 GDN layers (`~1.3%` of total prefill time at `B=128`), because Guild's fused CUDA kernel (`launch_gdn_recurrence_batched`) keeps the `[128, 128]` state in registers/shared memory per head block `(h, d)` across all `B` tokens.
     - However, for larger GPUs or FP16/BF16 Tensor Core architectures (`SM 8.0+`) with `B >= 256`, registering a chunked GDN kernel (`cuda.gdn_recurrence.chunk64.sm80.v1`) behind `GUILD_OP_GDN_RECURRENCE` can convert the recurrence into Tensor Core GEMMs without changing `Qwen35MoERunner`.

2. **Fused Linear-Attention QKV + Gate Projection (`build_qkvz` in `src/models/qwen35moe.cpp`)**
   - `llama.cpp` projects `wqkv` (`[n_embd, key_dim * 2 + value_dim]`) and `wqkv_gate` (`[n_embd, value_dim]`) as two separate `mul_mat` nodes unless fused in the model file.
   - **Applicability to Guild (`GUILD_OP_GEMM_BF16`)**:
     - Guild's `BatchedPrefill::run_gdn_layer` already concatenates `attn_qkv` (`8192` rows) and `attn_gate` (`4096` rows) into a single contiguous `12288`-row BF16 device buffer (`gdn_qkvz_proj`) and executes a single `[12288, 2048] x [2048, B]` GEMM per GDN layer, saving 30 kernel launches and 30 input-activation reads per chunk.

3. **Full-Attention FlashAttention / Vectorized KV Tile Reads (`ggml-cuda/fattn-*.cu`)**
   - `llama.cpp` dispatches tile-based FlashAttention kernels (`fattn-tile`, `fattn-vec`, `fattn-mma-f16`) that tile over `Q` rows (`B` prompt tokens) in shared memory while streaming `K/V` blocks once per `Q` tile.
   - **Applicability to Guild (`GUILD_OP_ATTN_PREFILL`)**:
     - In Guild's `BatchedPrefill::run_full_attn_layer`, `launch_qsa_prompt_attention` currently processes each prompt token `b in [0..B)` with a separate grid over KV heads, reading the prior KV cache `B` times per chunk (`43.62 ms` at `prompt=512`, scaling quadratically at `8K–16K` context).
     - **Stage 1 Follow-Up**: Implement a Q-tiled prompt attention kernel (`cuda.attn_prefill.qtiled.sm61.v1`) registered under `GUILD_OP_ATTN_PREFILL` that tiles `T_q = 16` or `32` query tokens per threadblock per GQA group (`8 Q heads / KV head` on Ornith, `16 Q heads / KV head` on Qwen), reducing KV cache HBM bandwidth by `8x–16x` during long-context prefill (`4K–32K`).

4. **Quantized Weight MMQ Prefill (`ggml-cuda/mmq.cu`)**
   - For architectures or layers where trunk projections remain quantized (`Q4_K`, `Q5_K`, `Q8_0`) in VRAM rather than expanded to BF16, `llama.cpp` quantizes the `[K, B]` activation batch to `q8_1` once and runs tiled `mmq_x_y` kernels using `__dp4a` (SM 6.1+).
   - **Applicability to Guild (`GUILD_OP_GEMM_QUANT`)**:
     - On 8 GiB SM 6.1 cards, Ornith's trunk projections fit in BF16 (`1265 MiB`), and `__dp4a` MMQ (`guild_mmq`) is already used when quantized device weights are active. Exposing `GUILD_OP_GEMM_QUANT` in `KernelRegistry` allows plug-in `SM 6.1 __dp4a` and `SM 8.0+ mma.sync` MMQ kernels to be selected per layer weight dtype.
