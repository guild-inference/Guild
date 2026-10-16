# Modular Runtime Architecture

Guild separates inference execution into three strictly decoupled layers connected by a versioned C-compatible ABI (`include/guild/abi/kernel_abi.h`) and declarative `.gkernel` package manager (`include/guild/abi/gkernel_package.hpp`):

1. **Guild Core (`src/runtime/`, `src/memory/`, `src/plan/`, `src/spec/`, `src/server/`, `src/abi/`)**
   - GGUF/pack loading, memory hierarchy planning (VRAM, pinned host RAM, mmap/file expert tiers), KV/recurrent state lifecycle, speculative decoding orchestration (`SpeculativeController`, `SuffixDrafter`, `MtpDrafter`), tokenization, sampling, HTTP/OpenAI/Anthropic APIs, CLI, and `.gkernel` cryptographic package management (`GKernelPackageManager`).
   - Completely independent of model-specific layer equations or hardware-specific kernel intrinsics.

2. **Model Architecture Runners (`include/guild/model/runner.hpp`, `src/model/runner.cpp`)**
   - Define how a model family (`qwen4exp`, `qwen35moe`, or external ABI v1 runners) maps GGUF metadata and tensors into `ModelDescriptor`, validates architectural invariants (such as mandatory PLE tables or hybrid recurrent/full-attention intervals), declares required kernel operations (`required_ops`), and binds prefill/decode execution graphs.
   - Registered in `guild::model::ModelRunnerRegistry` (`runner.qwen4exp.v1`, `runner.qwen35moe.v1`, or external `.gkernel` runners).

3. **Hardware Kernels (`include/guild/kernels/registry.hpp`, `src/kernels/registry.cpp`, `src/core/`, `src/engine/`, `plugins/`)**
   - Architecture- and device-specific compute implementations for CPU (`AVX2`, `AVX-512F/BW/DQ/VL`, `AVX-512 VNNI`, `NEON`), NVIDIA CUDA (`SM 6.1` Pascal through `SM 9.0`), and AMD HIP (`gfx1100+`).
   - Registered in `guild::kernels::KernelRegistry` (both built-in and dynamically loaded `.gkernel` plugins) with explicit operation kinds (`guild_op_kind_t`), hardware/ISA/SM requirements, supported data types, batch ranges, scratch memory calculators, and numerical parity flags.

---

## Execution Lifecycle & Zero-Overhead Hot Path

To preserve single-token decode latency and batched prefill throughput, **neither virtual method dispatch nor hash-table/registry lookups occur inside layer or token loops**:

1. **Startup & Dynamic `.gkernel` Package Loading**
   - `KernelRegistry::instance()` registers built-in CPU (`builtin.guild-kernels-cpu`) and GPU (`builtin.guild-kernels-cuda` / `builtin.guild-kernels-amd`) kernel descriptors.
   - `ModelRunnerRegistry::instance()` registers `Qwen4ExpRunner` (`builtin.guild-kernels-qwen`) and `Qwen35MoERunner` (`builtin.guild-kernels-ornith`).
   - `GKernelPackageManager::instance().load_enabled_packages()` validates and dynamically loads (`dlopen` with `RTLD_NOW | RTLD_LOCAL`) all enabled `.gkernel` packages from `<GUILD_DATA_DIR>/kernels`, verifying Ed25519 signatures and per-artifact SHA-256 digests before executing `guild_plugin_entry_v1` via transactional registration.

2. **Model Load (`GuildModel::load`)**
   - Calls `GKernelPackageManager::instance().load_enabled_packages()` so any enabled external `.gkernel` runners and kernels are active.
   - Reads the GGUF header and resolves the architecture runner via `ModelRunnerRegistry::resolve_for_gguf()`.
   - Invokes `runner->validate_and_describe()` and `ModelRunnerRegistry::validate_runner_requirements()` to verify that every required operation has a hardware-compatible kernel on the host CPU/GPU.
   - Acquires per-package `SessionRunnerLease` and `SessionKernelLease` bound to the model's `cudaStream_t` (`main_stream`).

3. **Prefill & Verifier Binding (`BatchedPrefill::init` / `Verifier::init`)**
   - `BatchedPrefill::init()` calls `KernelRegistry::resolve_prefill_kernels()` **once**, resolving the highest-priority enabled kernel for each prefill operation (including any external `.gkernel` plugin providing `guild_prefill_kernel_ext_v1_t`, such as `ext.cuda.moe.router_topk_bf16`) into raw function pointers in `ResolvedPrefillKernels`, and attaches their `package_id`s to the model's `SessionKernelLease`.
   - `Verifier` captures pre-bound CUDA graphs for single-token and speculative verify windows.

4. **Token / Layer Inner Loops**
   - `BatchedPrefill::run()` invokes the pre-bound function pointers in `ResolvedPrefillKernels` and the pre-bound `ExpertPool::run_split_multi_native()` workers directly.
   - `Verifier::run()` launches captured CUDA graphs directly.

---

## Measured Phase 1 Performance Verification (`Ornith-1.5-35B-Q4_K_M`)

Measured on dual Intel Xeon Gold 6130 (32 physical cores / 64 threads, AVX-512F/BW/DQ/VL, no VNNI, 377 GiB DDR4 across 2 NUMA nodes) + NVIDIA GeForce GTX 1070 (8 GiB VRAM, SM 6.1, CUDA 12.0, driver `580.178.04`), `Ornith-1.5-35B-Q4_K_M.gguf`, `context=32768`, `batch=256`, `generate=128`, 1 full-shaped warmup + 10 measured repetitions (`ornith_prefill_bench`):

| Prompt Tokens | Configuration | `9d77159` Baseline Prefill Median | Phase 1 Prefill Median (10 runs) | `9d77159` Baseline Decode Median | Phase 1 Decode Median (10 runs) |
|---:|---|---:|---:|---:|---:|
| `512` | Built-in (`gpu.moe.router_topk_native`) | `234.156 tok/s` | **`240.176 tok/s`** (`+2.57%`) | `42.977 tok/s` | **`43.427 tok/s`** (`+1.05%`) |
| `512` | External `.gkernel` (`ext.cuda.moe.router_topk_bf16`) | `234.156 tok/s` | **`240.089 tok/s`** (`+2.53%`) | `42.977 tok/s` | **`43.327 tok/s`** (`+0.81%`) |
| `2,048` | Built-in (`gpu.moe.router_topk_native`) | `220.620 tok/s` | **`231.634 tok/s`** (`+4.99%`) | `37.909 tok/s` | **`37.819 tok/s`** (`-0.24%`) |
| `4,096` | Built-in (`gpu.moe.router_topk_native`) | `209.275 tok/s` | **`216.998 tok/s`** (`+3.69%`) | `33.041 tok/s` | **`32.909 tok/s`** (`-0.40%`) |

Real-model end-to-end plugin verification (`ornith_gkernel_plugin_test`):
- `ext.guild-kernels-ornith-prefill@1.0.0` (`.gkernel` bundle signed with Ed25519) dynamically loaded via `GKernelPackageManager::install_package`.
- Verified `ext.cuda.moe.router_topk_bf16` bound in `BatchedPrefill` and executed 40 times per prompt across all 40 layers (`760` token-rows for a 20-token prompt).
- Full-vocabulary logits (`248,320` floats) and 8-token greedy continuation match the built-in kernel **bit-for-bit** (`std::memcmp == 0`).
- Active lease protection rejects `disable_package` / `remove_package` (`GUILD_STATUS_ERR_ACTIVE_SESSION_BUSY`) while bound, and cleanly falls back to `gpu.moe.router_topk_native` after lease release and `disable_package`.

---

## Roadmap: Phase 6 — Long-Context Decode Optimization (`8K–32K+`)

On 8 GiB VRAM cards (`GTX 1070`), Ornith-1.5-35B places its 10 full-attention layers' KV cache (`1,280 MiB` at 32K context in FP16) in pinned host RAM (`HostOnly` mode) and streams KV blocks over PCIe 3.0 x16 during decode (`42.98 tok/s` at 512 tokens -> `26.07 tok/s` at 8,192 tokens -> `18.05 tok/s` at 16,384 tokens). Phase 6 targets this long-context decode bottleneck via modular `.gkernel` and runtime enhancements:

1. **Multi-CTA Split-K FlashDecoding (`GUILD_OP_ATTN_DECODE`)**
   - Partition long sequence history (`N_kv >= 4096`) across multiple threadblocks per KV head (`split_k = 4..16`), computing partial online-softmax `(m_i, l_i, O_i)` tiles in parallel across all SMs and reducing across splits in a lightweight combine kernel.
2. **Asynchronous Double-Buffered Host-to-Device KV Tile Prefetching**
   - Overlap PCIe host-to-device KV tile transfers for full-attention layer `l_next` on a dedicated DMA stream while the GPU executes the 3 preceding Gated DeltaNet layers and CPU expert pool dispatch.
3. **Quantized Host KV Cache Streaming (`Q8_0` / `K8V4` for `Qwen35MoE`)**
   - Extend `GUILD_OP_KV_WRITE` and `GUILD_OP_ATTN_DECODE` in `Qwen35MoERunner` to support `Q8_0` and `K8V4` host-tier KV storage, cutting PCIe 3.0 KV read traffic by `2x–2.67x` at `8K–32K` context while preserving `512 MiB` VRAM safety reserve.
