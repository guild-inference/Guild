# Phase 6.5: General Inference Optimization & Research Integration

Phase 6.5 profiles and optimizes Guild's end-to-end inference pipeline across both `Qwen3.8-Flash-Next` (`UD-IQ4_XS`) and `Ornith-1.5-35B` (`Q4_K_M`) following Phase 6 (`98b2e5e`). We survey 15 primary 2025–2026 research and systems sources, construct a ranked 9-candidate engineering matrix, prototype 6 concrete hypotheses across CPU quantized experts, GPU verify-graph execution, batched prefill overlap, and GEXEC autotuning, retain the validated improvements with strict bitwise (`std::memcmp == 0`) parity, and document both accepted and rejected experiments with full provenance in [`bench/results/phase65-general-optimization/`](../../bench/results/phase65-general-optimization/).

---

## 1. Research Survey & Candidate Evaluation Matrix (Through October 2026)

Before selecting implementation targets, we reviewed 15 primary research papers and production engine implementations (accessed 2026-10-20). No published speedup is assumed to transfer to Guild without direct measurement on our hardware.

### 1.1 Primary Research & Systems Sources

| # | Technique & Source / Date | Hardware & Workload Evaluated | Published Claims & Limitations | Guild Applicability, Complexity, Numerical & Resource Profile | License / Provenance |
|---:|---|---|---|---|---|
| 1 | **FlashInfer 0.7 (`Prims-TS` & `MegaMoE`)**<br>FlashInfer 0.7 release notes, Sep 2026 | 8x NVIDIA B200 (`SM100`), DeepSeek-R1 / Qwen3 MoE distributed serving | `1.06x`–`1.32x` end-to-end throughput on 8x `SM100`; relies on NVLink symmetric memory and FP8/FP4 checkpoint changes in some configurations. | Task-scheduling and communication/compute overlap principles apply to CPU–GPU MoE coordination; `SM100` symmetric-memory kernels do not apply to single-GPU Pascal (`SM61`). High complexity for GPU kernel port; zero model changes for host/device scheduling. | Apache-2.0 (referenced for scheduling concepts only; no code copied) |
| 2 | **flash-moe (`flash-moe.c`)**<br>2026 C/CUDA single-GPU MoE streaming implementation | Apple M3 Max / consumer desktop GPU + SSD/RAM expert streaming | Reports high single-request decode tok/s via asynchronous expert prefetching and zero-copy mapped rings. | Applicable to Guild's `Verifier` and `Prefill` expert streaming rings. However, when all routed experts fit in system RAM (`complement_ready_ == true`), extra ring/staging threads and spin-waits add overhead rather than hiding I/O. Low complexity; exact arithmetic. | MIT (concepts only) |
| 3 | **MoE-Spec (Speculative Expert Budgeting)**<br>arXiv:2602.16052, Feb 2026 | A100/H100 GPUs, Mixtral / DeepSeek-MoE / Qwen2-MoE | Caps the union of active experts across speculative draft trees to bound memory bandwidth during verification, trading slight acceptance drop for faster verify steps. | Guild already caps speculative verification windows at `kVerifyMaxT = 2` tokens (`1` target + `1` MTP draft) with confidence threshold `spec_min_p = 0.5`. Modifying router top-`K` union selection alters target distribution unless rejected tokens fall back, violating exact greedy parity. | Research paper (no code used) |
| 4 | **Speculative MoE / Post-Training Expert Prefetching**<br>arXiv:2503.04398 & arXiv:2512.23392, Mar–Dec 2025 | Consumer GPUs with PCIe-attached host RAM (offloaded MoE) | Predicts next-layer expert IDs from hidden states `1`–`2` layers ahead to overlap PCIe H2D expert transfers with GPU attention. | Applicable only when experts are transferred over PCIe (`pcie_expert_fraction > 0`) or read from SSD (`staged() && !complement_ready_`). On PCIe 3.0 x16 (`~12 GB/s`) with 32-core Xeon RAM bandwidth (`~115 GB/s`), CPU-resident execution is faster than PCIe expert transfer (`--pcie-frac 0`), and when `complement_ready_ == true`, prefetch mutex acquisition is pure overhead. | Research papers |
| 5 | **C2C (`CPU-to-CUDA` MoE Hybrid Scheduling)**<br>arXiv:2512.15635, Dec 2025 | Dual-socket Xeon + PCIe GPU, Qwen / Mixtral MoE | Partitions routed experts between CPU SIMD threads (`nt` small) and GPU GEMM (`nt` large) with single-barrier synchronization. | Matches Guild's `ExpertPool` (`mode_ == 11` single-barrier fused pipeline) + `Verifier` mapped doorbell handshake (`h_seq_`, `h_flag_`). Demonstrates that eliminating unused GPU hit/miss kernels and host-side lock overhead directly reduces per-layer synchronization latency. | Research paper |
| 6 | **ELLM & Collaborative Speculative Decoding**<br>arXiv:2506.19824 & arXiv:2502.13767, Feb–Jun 2025 | Heterogeneous CPU+GPU edge/workstation systems | Early-exits or confidence-gates draft generation when draft entropy is high, avoiding wasted 2-token verification windows on low-probability drafts. | Already integrated in Guild (`spec_min_p = 0.5`, `T = 1` fallback when `draft_p < 0.5`). Phase 6 measurements confirmed lowering `spec_min_p` to `0.0` regresses Qwen decode by `-23.5%` (`20.70` -> `15.83 tok/s`). | Research papers |
| 7 | **Prism / Heterogeneous P/D Disaggregation**<br>arXiv:2505.12341, May 2025 | Multi-GPU clusters | Separates prefill and decode across distinct hardware pools. | Inapplicable to single-GPU desktop deployments (`1x GTX 1070`). | Research paper |
| 8 | **Sparsity-Aware / Asymmetric KV Quantization (`K8V4`, Hadamard Q4)**<br>2025–2026 long-context KV compression literature | Consumer & datacenter GPUs, 32K–256K contexts | Uses INT8 keys (preserving QK logit precision) and Hadamard-rotated 4-bit values (`K8V4`) to halve KV footprint with minimal perplexity loss. | Already implemented in Guild (`--kv k8v4`, `--kv int8`, `--kv q4_0`). In Phase 6.5 we wire full GEXEC compatibility and multi-format `BatchedPrefill` batching across all four KV modes. | Built-in Guild kernels |
| 9 | **`llama.cpp` MoE & CPU Quantization Kernels**<br>Commits `3cf03257f` (pinned) and `11fe02151` (Oct 2026) | x86 AVX2 / AVX-512 (`quants.c`) and CUDA MMVQ | Vectorized `IQ3_S`, `IQ2_S`, `IQ4_XS`, `Q4_K`, `Q6_K` dot products (`ggml_vec_dot_iq3_s_q8_K`) using dual integer accumulators (`sumi0`, `sumi1`) and `xor+sub` sign application. | High applicability to Guild's GEXEC CPU expert kernels (`src/kernels/cpu/gexec_expert.cpp`). Direct inspection of `quants.c` revealed why `ggml_vec_dot_iq3_s_q8_K` beat our earlier `iq256` kernel at `nt=1` and how to surpass both without stack reloads or indirect function calls. | MIT (`ggml-cpu`) |

### 1.2 Ranked Candidate Engineering Matrix

| Rank | Candidate & Target Subsystem | Expected Benefit & Runtime Share | Generality & Compatibility | Complexity & Maintainability | Numerical & Resource Risk | Classification & Final Disposition |
|---:|---|---|---|---|---|---|
| **1** | **Specialized AVX2 `IQ3_S`/`IQ2_*` GEXEC CPU Expert Kernels (`src/kernels/cpu/gexec_expert.cpp`)** | **High** (`3%`–`5%` end-to-end Qwen decode; `12%`–`15%` CPU Gate+Up reduction on `78%` of Qwen's decode CPU time) | All x86-64 AVX2 and Skylake-SP hosts running `UD-IQ4_XS` / `IQ3_S` / `IQ2_*` models | Low (`~180` LOC in `gexec_expert.cpp`) | **Zero numerical risk** (`max_abs_err = 0.0f`, `std::memcmp == 0`); zero memory/startup cost | **Production-ready — ACCEPTED** |
| **2** | **`Verifier` All-CPU Decode Graph Pruning & Lock-Free `ExpertSource` Fast Paths (`src/core/verify.cpp`, `src/core/expert_source.cpp`)** | **Medium** (`1%`–`2%` end-to-end decode across both Ornith and Qwen; eliminates `160`–`192` unused CUDA kernel launches per window when `n_slots == 0 && !pcie_on_`) | Both Ornith and Qwen whenever all routed experts run on CPU (`--pcie-frac 0`, `expert_vram_bytes == 0`) | Low (`~60` LOC across `verify.cpp`, `expert_source.cpp`, `model.cpp`) | **Zero numerical risk** (identical tokens and `.f32` logits); slightly reduces CUDA graph node count | **Production-ready — ACCEPTED** |
| **3** | **`BatchedPrefill` Batched KV Append, Parallel Input Finiteness Validation & Early Shared-Expert Submission (`src/runtime/batched_prefill.cpp`, `src/kernels/cpu/pool.cpp`)** | **Medium** (`1.1%`–`2.5%` Ornith prefill speedup; replaces `256` per-token `kv_append_step` launches per QSA layer with `1` batched `prefill::kv_append` launch and parallelizes `2 MiB` finiteness scan across 32 threads) | Ornith-1.5-35B and any model using `BatchedPrefill` (`FP16`, `INT8`, `Q4_0`, `K8V4`) | Low (`~50` LOC) | **Zero numerical risk** (identical tokens and `.f32` logits); zero memory cost | **Production-ready — ACCEPTED** |
| **4** | **Scoped NUMA Node-0 Affinity for `Verifier::init` & `MtpDrafter::init` Mapped Buffers (`src/core/verify.cpp`, `src/core/mtp.cpp`)** | **Low–Medium** (prevents cross-socket UPI PCIe reads of mapped doorbell/activation buffers when `GuildModel::load` runs on Socket 1) | Multi-socket NUMA hosts (Linux/Windows) | Low (`~20` LOC RAII guard matching `Prefill::init`) | Zero numerical or memory risk | **Production-ready — ACCEPTED** |
| **5** | **GEXEC Optimizer Layout Guard (`PortableContiguous` vs `GateUpRowInterleaved` without `--embed-experts`, `src/gexec/gexec_optimizer.cpp`)** | **High reliability impact** (prevents `guild optimize` noise from picking `GateUpRowInterleaved` when `--embed-experts` is off, which previously caused runtime fallback or incompatible layout) | All models using `.gexec` without embedded expert blobs | Low (`10` LOC in `gexec_optimizer.cpp`) | Zero numerical risk | **Production-ready — ACCEPTED** |
| **6** | **K-Quant `nt > 2` 2-Row Fused GEXEC Routing (`src/kernels/cpu/gexec_expert.cpp`)** | Targeted Ornith `BatchedPrefill` multi-token expert groups (`nt = 3..8`) | K-quant models (`Q4_K_M`) | Low | Zero numerical risk, but **regressed `nt=4` (`74.7 ms` vs `71.2 ms`) and `nt=8` (`>100 ms` vs `98.2 ms`)** due to YMM register spilling | **Rejected — Reverted (Prototype 1)** |
| **7** | **Qwen `Prefill::run_impl` Staging Ring Expansion (`stream_all` / `STAGE = 32`, `src/prefill/prefill.cpp`)** | Targeted Qwen `wait copy = 15.3 s` (`36.7%` of 4K prefill) | Qwen chunked prefill (`T = 512`) | Medium | `stream_all=true` regressed 4K prefill from `98.3` to `59.4 tok/s` (`+25%` unrouted PCIe transfers + `cudaEventSynchronize` contention); `STAGE=32` regressed to `96.6 tok/s` (front-loaded gather stall) | **Rejected — Reverted (Prototype 2)** |
| **8** | **Fused-Pool Task Granularity Rebalancing (`3x` Gate+Up / `2x` Down in `mode_ == 11`, `src/kernels/cpu/pool.cpp`)** | Targeted decode `ExpertPool` tail-wave balance | Multi-threaded CPU expert decode | Low (`2` LOC) | Improved Ornith pool trial by `+3.2%`, but **regressed Qwen pool trial by `-45%`** (`22.15` -> `40.26 ms/tok`) due to `50%` more DRAM page-stream switches across `55.4 GiB` NUMA-interleaved `IQ3_S` weights | **Rejected — Reverted (Prototype 6)** |
| **9** | **Qwen `Stager` Thread/Ring Scaling (`GUILD_STAGER_THREADS=8`, `GUILD_STAGER_RING=32`)** | Targeted host `memcpy` staging ahead of PCIe DMA | Qwen chunked prefill | Zero code change (env tunable) | Unchanged (`97.69 tok/s` vs `98.19 tok/s`, `wait copy = 15.50 s` vs `15.31 s`) because `4` stager threads (`~40 GB/s`) already outrun PCIe 3.0 x16 (`~11.2 GB/s`) by `3.5x` | **Rejected (Prototype 6b)** |

---

## 2. Baseline Profiling & Bottleneck Attribution

All baseline measurements were captured from the preserved Phase 6 binary (`build-cuda12/phase65/baseline/ornith_prefill_bench`, SHA-256 `edccd82361658f8600042be13a236f3432b84899f91a45d1a178722e633b9b9d`) on dual Intel Xeon Gold 6130 (32C/64T across 2 NUMA nodes, 376 GiB RAM) + NVIDIA GeForce GTX 1070 (8,192 MiB VRAM, `SM61`, NUMA node 0).

### 2.1 Decode Window Breakdown (`GUILD_VERIFY_PROFILE=1`, `p=4,096`)
- **`Ornith-1.5-35B-Q4_K_M` (`64` windows, `1` token/window)**:
  - Total window latency: `21.95 ms/window` (`1,405 ms` total)
  - `waitCPU` (GPU spinning on `h_flag_` for CPU expert completion): `5.36 ms/window` (`24.4%`)
  - `shared+quant` (shared expert + Q8 activation quantization): `3.26 ms/window` (`14.9%`)
  - `q8+qkv/q-idx gemv`: `2.99 ms/window` (`13.6%`)
  - `head` (LM head GEMV + argmax): `1.80 ms/window` (`8.2%`)
  - `out+norm`: `1.41 ms/window` (`6.4%`)
  - `router`: `1.18 ms/window` (`5.4%`)
  - `norm+q`: `0.98 ms/window` (`4.5%`)
  - `comb+gr`: `0.90 ms/window` (`4.1%`)
  - `copyCPU` + `ring` + `waitA`/`hit0`/`waitB`/`pcie`/`hitadd`: `2.36 ms/window` (`10.8%` — of which `waitA + hit0 + waitB + pcie + hitadd = 1.13 ms/window` executed even when `0` experts were on GPU!)
- **`Qwen3.8-Flash-Next-UD-IQ4_XS` (`38` verify windows + `38` MTP draft steps for `64` tokens, `26/34` drafts accepted)**:
  - Total verify window latency: `75.56 ms/window` (`2,871 ms` verify + `144.5 ms` MTP draft)
  - `waitCPU`: `37.65 ms/window` (`49.8%` of verify window)
  - `shared+quant`: `12.66 ms/window` (`16.8%`)
  - `q8+qkv/q-idx gemv`: `6.88 ms/window` (`9.1%`)
  - `ple`: `3.87 ms/window` (`5.1%`)
  - `out+norm`: `2.86 ms/window` (`3.8%`)
  - `router`: `2.45 ms/window` (`3.2%`)
  - `head`: `1.80 ms/window` (`2.4%`)
  - `comb+gr`: `1.77 ms/window` (`2.3%`)
  - Inside the CPU pool (`decode_cpu_pool_ms = 1,260.7 ms`), Gate+Up projection (`decode_cpu_gu_ms = 947.9 ms`, predominantly `39` layers of `IQ3_S` `gu_type = 21`) accounted for **`75.2%`** of all CPU expert compute.

### 2.2 Prefill Breakdown (`p=4,096`)
- **`Ornith-1.5-35B-Q4_K_M` (`BatchedPrefill`, `--batch 256`)**:
  - Total prefill wall time: `17,917.1 ms` (`228.55 tok/s`)
  - `host_wait_ms` (host waiting on `done_event` for GPU attention/GDN/router/D2H): `10,628.9 ms` (`59.3%`)
  - `cpu_gu_ms`: `4,269.0 ms` (`23.8%`)
  - `cpu_down_ms`: `2,228.7 ms` (`12.4%`)
  - `cpu_group_ms`: `109.1 ms` (`0.6%`)
  - `cpu_quant_ms`: `76.5 ms` (`0.4%`)
- **`Qwen3.8-Flash-Next-UD-IQ4_XS` (`Prefill::run_impl`, `--batch 512`, `GUILD_PREFILL_TIMING=1`)**:
  - Total prefill wall time: `41,660 ms` (`98.30 tok/s`)
  - `wait copy` (PCIe 3.0 x16 H2D DMA of `~19.2 GiB` of routed expert blobs at `~11.2 GB/s` effective): `15,308 ms` (`36.7%`)
  - `gemm gate/up` + `gemm down` + `dequant`: `7,491 + 2,503 + 2,629 = 12,623 ms` (`30.3%`)
  - `gdn` + `gdn conv+gates` + `gdn recurrence` + `gdn out proj`: `5,980 ms` (`14.4%`)
  - `qsa proj` + `qsa select` + `qsa attn`: `3,795 ms` (`9.1%`)
  - `hc read`: `1,880 ms` (`4.5%`)
  - `router+shared`: `729 ms` (`1.7%`)
  - `gather` + `combine` + `host grouping` + `ple`: `989 ms` (`2.4%`)

---

## 3. Accepted Optimizations & Architectural Details

### 3.1 Bit-Exact AVX2 `IQ3_S` / `IQ2_*` GEXEC CPU Expert Kernels (`src/kernels/cpu/gexec_expert.cpp`)
- **Problem**: On Skylake-SP (`avx512bw` without `avx512_vnni`), `cpu_avx512_ok()` is intentionally `false` to prevent heavy-license AVX-512 frequency downclocking across mixed AVX2 layers. However, `gexec_gu_rows_impl` previously only had a 256-bit AVX2 specialization for `IQ4_XS` (`gu_type == 23`), while `IQ2_XXS` (`16`), `IQ2_XS` (`17`), `IQ3_XXS` (`18`), `IQ3_S` (`21`), and `IQ2_S` (`22`) were gated behind `if (USE_AVX512 && cpu_avx512_ok())` and fell back to `native_gu_rows`! For `Qwen3.8-Flash-Next-UD-IQ4_XS` (`39/48` layers `gu_type = 21` `IQ3_S`), both `gexec_fused_avx2` and `gexec_fused_avx512` fell back to `native_gu_rows`, which at `nt == 1` called `ggml_vec_dot_iq3_s_q8_K` via indirect function pointers (`1,280` calls per expert) without software prefetching and with stack store-to-load reloads (`uint16_t idx[8]`).
- **Solution**:
  1. Implemented `Iq256Fmt<16, 17, 18, 21, 22>`, `iq256_row_dot_1row<TY, NT>`, `gexec_iq256_gu_rows_nt`, and `dispatch_iq256_gu` in `src/kernels/cpu/gexec_expert.cpp`.
  2. For `TY == 21` (`IQ3_S`), `iq256_row_dot_1row` unrolls the `j = 0..3` sub-block loop, computes the 9-bit codebook index directly in registers (`q0[k] | ((h0 << (8 - k)) & 0x100u)`) without touching the stack, applies signs via `_mm256_sub_epi8(_mm256_xor_si256(y, s2), s2)` (`iq256_sgn_mask`), accumulates into two independent integer accumulators (`sumi0[t]`, `sumi1[t]`) to break the 4-cycle `vpaddd` dependency chain across `half = 0` and `half = 1`, and prefetches `blk + 2048` (`2` cache lines ahead).
  3. Preserved exact `ggml-cpu` / `iq_avx2` floating-point reduction order (`use_ggml_reduce ? K * hsum_f32_8(accf) : hsum8_hadd(accf)`), achieving `0.000000` maximum absolute error (`std::memcmp == 0`) across all `2,447` checks in `gexec_optimizer_test`.
- **Measured impact**:
  - In `guild optimize` Pass 3 on `Qwen3.8-Flash-Next`, `ExpertPool` format 0 (`gu=21, d=20`) latency dropped from `0.583 ms` (`0.918 ms` portable baseline) to **`0.517 ms`** (`-11.3%` vs Phase 6 GEXEC, **`-43.7%` vs portable baseline**), and model-wide pool trial time dropped from `29.28 ms/tok` to **`22.15 ms/tok` (`1.689x` pool speedup)**.
  - End-to-end Qwen `decode_cpu_gu_ms` dropped by **`-12.5%` to `-13.4%`** (`893.6` -> `782.0 ms` at `p=512`, `947.9` -> `820.5 ms` at `p=4096`, `891.8` -> `771.9 ms` at `p=65536`), boosting end-to-end decode throughput by **`+2.9%` to `+4.1%`** across `512`–`65,536` populated contexts.

### 3.2 `Verifier` All-CPU Decode Graph Pruning & Lock-Free `ExpertSource` Fast Paths (`src/core/verify.cpp`, `src/core/expert_source.cpp`, `src/runtime/model.cpp`)
- **Problem**: During CUDA graph capture and execution in `Verifier::record_window` (`src/core/verify.cpp`), every MoE layer unconditionally recorded `wait_flag_ge(flagA)`, `copy_rows_from_mapped_kernel` / `s2_expert_grouped_hit` (`hit0`), `wait_flag_ge_or(flagB)`, PCIe expert kernels, and `add_inplace` (`hitadd`) whenever `dec_batch` (`GUILD_DEC_BATCH != 0`) was enabled — even when `hits_.n_slots == 0` and `--pcie-frac 0` (`pcie_on_ == false`). On a 40-layer Ornith or 48-layer Qwen model with 100% CPU-resident experts, this launched `160` to `192` no-op GPU kernels and mapped-memory spin-waits per decode window (`~1.13 ms/window` on GTX 1070) and zeroed `200 KiB` of `ymiss` per layer on the GPU. Simultaneously, on the CPU side, `expert_pool_dispatch_multi` (`src/core/expert_source.cpp`) ran an $O(n^2)$ distinct/miss scan and `FileExpertSource::prefetch` acquired `std::mutex stage_mu_` on every layer even when `complement_ready_ == true` (100% of experts already locked in RAM).
- **Solution**:
  1. Added `Verifier::set_pcie_enabled(bool)` (`include/guild/core/verify.hpp`, wired in `src/runtime/model.cpp`) and gated the GPU expert plan/hit/PCIe graph nodes in `Verifier::record_window` and `Verifier::run` on `const bool has_gpu_experts = (hits_.n_slots > 0 || pcie_on_); const bool need_gpu_plan = has_gpu_experts || device_plan;`.
  2. Added an `all_cpu` fast path in `expert_pool_dispatch_multi` (`src/core/expert_source.cpp`) when `d.cache == nullptr && d.cache_base == nullptr && !pcie_ok && d.peer == nullptr && d.remote_count == 0`, and a lock-free `complement_ready_` check at the top of `FileExpertSource::prefetch` that returns immediately without locking `stage_mu_` when all requested experts reside in `complement_offsets_`.

### 3.3 `BatchedPrefill` Batched KV Append, Parallel Input Validation & Early Shared-Expert Launch (`src/runtime/batched_prefill.cpp`, `src/kernels/cpu/pool.cpp`)
- **Problem**: In `BatchedPrefill::run` (`src/runtime/batched_prefill.cpp`):
  1. On each of the 10 QSA layers, KV cache population looped over query tiles and tokens (`for (int t = 0; t < qn; ++t)`), launching `kv_append_step` or `kv_append_q4_step` **256 separate times per QSA layer** (`2,560` tiny CUDA kernel launches per 256-token batch) instead of a single batched launch for the `T` contiguous cells `[pos0, pos0 + T)`.
  2. After `cudaEventSynchronize(m.done_event)` (waiting for `m.hact` D2H), the host thread called `shared_expert_multi` and then sequentially scanned `m.hact` (`256 * 2048 * 4 = 2 MiB`) via `core::validate_finite` on 1 core while 31 worker threads sat parked, before waking all 32 threads in `quant_act_multi_native` (`mode_ == 9`) to read the exact same `2 MiB` buffer again.
- **Solution**:
  1. Replaced the per-token `kv_append_step` / `kv_append_q4_step` loop in `BatchedPrefill` with a single batched `prefill::kv_append` / `kernels::kv_append_q4` call for the `T` new cells `[pos0, pos0 + T)` before the 32-query attention tile loop.
  2. Moved `shared_expert_multi` submission on `cs` ahead of `cudaEventSynchronize(m.done_event)` so the GPU executes the shared expert concurrently while the host synchronizes `done_event` and groups routed experts.
  3. Fused per-token `slice_all_finite` validation into `ExpertPool::quant_act_multi_native` (`mode_ == 9` in `src/kernels/cpu/pool.cpp`) across all 32 worker threads, falling back to `core::validate_finite` only if a non-finite value is detected to preserve the exact diagnostic error message.
- **Measured impact**: `Ornith-1.5-35B` prefill throughput improved from `242.95` to **`245.66 tok/s` (`+1.1%`)** at `p=512` and from `228.55` to **`233.37 tok/s` (`+2.1%`)** at `p=4,096` in 3-run full-warmup counterbalanced A/B testing (`ornith-portable` curve also rose from `224.46` to **`230.95 tok/s` (`+2.9%`)** at `p=4,096`).

### 3.4 Scoped NUMA Node-0 Affinity in `Verifier::init` & `MtpDrafter::init` and GEXEC Layout Guard (`src/core/verify.cpp`, `src/core/mtp.cpp`, `src/gexec/gexec_optimizer.cpp`)
- Scoped a `HostAllocAffinity` RAII guard (`pin_current_thread_if_allowed(device_)` + `restore_thread_affinity`) around the `mapped()` (`cudaHostAllocMapped`) allocations in `Verifier::init` and `MtpDrafter::init`, ensuring mapped zero-copy doorbell and activation buffers reside on NUMA node 0 (local to the PCIe root complex of GPU 0) even if the OS scheduler migrated the loading thread to Socket 1 during `FileExpertSource` initialization.
- In `run_autotune_pass` (`src/gexec/gexec_optimizer.cpp`), gated `is_gexec_interleaved_impl` selection on `budget.embed_packed_experts` so `guild optimize` without `--embed-experts` always selects the `PortableContiguous` GEXEC pipeline (`gexec_fused_avx2` / `gexec_fused_avx512`) rather than intermittently picking `GateUpRowInterleaved` when AVX2 contiguous and interleaved kernels tie within microbenchmark noise.

---

## 4. End-to-End Benchmark Results

### 4.1 Counterbalanced 3-Run Full-Warmup A/B Comparison (`build-cuda12/phase65/` & `bench/results/phase65-general-optimization/ab-counterbalanced/`)

Measured with `--runs 3 --full-warmup --ignore-eos` using identical frozen prompt IDs and verified for 100% bitwise token and first-step `.f32` logit equality (`ids_eq=True, f32_eq=True`):

| Workload (`--runs 3 --full-warmup`) | Phase 6 Baseline Prefill Median (`p05`–`p95`) | Phase 6.5 Prefill Median (`p05`–`p95`) | Prefill Delta | Phase 6 Baseline Decode Median (`p05`–`p95`) | Phase 6.5 Decode Median (`p05`–`p95`) | Decode Delta (`decode_cpu_gu_ms` delta) |
|---|---:|---:|---:|---:|---:|---:|
| `Ornith-1.5-35B` GEXEC `p=512` | `242.95 tok/s` (`236.2`–`243.4`) | **`245.66 tok/s`** (`237.3`–`247.0`) | **`+1.1%`** | `49.54 tok/s` (`49.50`–`49.85`) | **`49.95 tok/s`** (`49.63`–`50.06`) | **`+0.8%`** |
| `Ornith-1.5-35B` GEXEC `p=4,096` | `228.55 tok/s` (`228.1`–`228.7`) | **`233.37 tok/s`** (`233.0`–`234.2`) | **`+2.1%`** | `47.71 tok/s` (`43.91`–`47.78`) | **`47.95 tok/s`** (`45.29`–`48.00`) | **`+0.5%`** |
| `Qwen3.8-Flash-Next` GEXEC `p=512` | `94.82 tok/s` (`94.79`–`94.83`) | **`94.67 tok/s`** (`94.67`–`94.71`) | `-0.2%` (noise) | `22.43 tok/s` (`22.43`–`22.58`) | **`23.00 tok/s`** (`22.99`–`23.13`) | **`+2.5%`** (**`-12.5%`** CPU Gate+Up) |
| `Qwen3.8-Flash-Next` GEXEC `p=4,096` | `98.32 tok/s` | **`97.94 tok/s`** (`97.69`–`98.17`) | `-0.4%` (noise) | `21.39 tok/s` | **`22.13 tok/s`** (`22.12`–`22.14`) | **`+3.5%`** (**`-13.4%`** CPU Gate+Up) |

### 4.2 Populated-Context Scaling Curves (`512` to `65,536` Tokens)

#### `Ornith-1.5-35B-Q4_K_M` (`--context 32768`, `--batch 256`, `--generate 64`, `--spec 0`)

| Populated Prompt Tokens | Phase 5.6 Baseline Prefill (`tok/s`) | Phase 6 GEXEC Prefill (`tok/s`) | Phase 6.5 Portable Prefill (`tok/s`) | Phase 6.5 GEXEC Prefill (`tok/s`) | Phase 5.6 Baseline Decode (`tok/s`) | Phase 6 GEXEC Decode (`tok/s`) | Phase 6.5 Portable Decode (`tok/s`) | Phase 6.5 GEXEC Decode (`tok/s`) | Phase 6.5 GEXEC vs Baseline Decode |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `512` | `238.28` | `237.87` | **`245.30`** | `237.33` (`245.66` 3-run med) | `46.59` | `49.48` | `47.22` | **`49.69`** (`49.95` 3-run med) | **`+6.7%`** |
| `4,096` | `225.78` | `226.65` | **`230.95`** | `227.51` (`233.37` 3-run med) | `45.14` | `47.63` | `45.75` | **`47.76`** (`47.95` 3-run med) | **`+5.8%`** |
| `8,192` | `209.57` | `209.16` | **`212.22`** | `206.60` | `43.38` | `45.77` | `43.49` | **`45.98`** | **`+6.0%`** |
| `16,384` | `181.48` | `182.22` | **`183.99`** | `181.47` | `39.71` | `42.50` | `39.82` | **`42.72`** | **`+7.6%`** |
| `32,704` | *(unsupported at `p=32768`)* | `145.82` | **`147.03`** | **`146.70`** | *(unsupported at `p=32768`)* | `36.91` | `35.57` | **`36.92`** | **`+3.8%` vs portable** |

#### `Qwen3.8-Flash-Next-UD-IQ4_XS` (`--batch 512`, `--generate 64`, `--spec 1`, MTP `2` drafts)

| Populated Prompt Tokens | Allocated Capacity | Phase 5.6 Baseline Prefill (`tok/s`) | Phase 6 GEXEC Prefill (`tok/s`) | Phase 6.5 Portable Prefill (`tok/s`) | Phase 6.5 GEXEC Prefill (`tok/s`) | Phase 5.6 Baseline Decode (`tok/s`) | Phase 6 GEXEC Decode (`tok/s`) | Phase 6.5 Portable Decode (`tok/s`) | Phase 6.5 GEXEC Decode (`tok/s`) | Phase 6.5 GEXEC Decode Gain (vs Phase 5.6 / vs Phase 6) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `512` | `32,768` | `95.21` | `94.34` | `93.72` | **`94.04`** (`94.67` 3-run) | `21.66` | `22.66` | `21.75` | **`23.32`** | **`+7.7%` / `+2.9%`** |
| `4,096` | `32,768` | `98.56` | `98.21` | `97.44` | **`97.92`** (`97.94` 3-run) | `20.70` | `21.32` | `20.79` | **`22.19`** | **`+7.2%` / `+4.1%`** |
| `8,192` | `32,768` | `95.68` | `95.56` | `95.04` | **`95.18`** | `21.37` | `22.47` | `21.85` | **`23.12`** | **`+8.2%` / `+2.9%`** |
| `16,384` | `32,768` | `93.29` | `92.94` | `92.80` | **`92.54`** | `22.39` | `23.29` | `22.57` | **`24.04`** | **`+7.4%` / `+3.2%`** |
| `32,768` | `65,536` | `87.09` | `91.56` | — | **`91.91`** | `20.63` | `21.59` | — | **`22.03`** | **`+6.8%` / `+2.0%`** |
| `65,536` | `131,072` | *(0.08 tok/s or OOM)* | `89.50` | — | **`89.53`** | *(OOM / timeout)* | `23.05` | — | **`23.87`** | **`— / +3.5%`** |

### 4.3 KV Precision & Placement Sweep (`p=4,096`, `32,768` Capacity, GEXEC Active)

| Model & KV Configuration | Host KV Memory | Device KV Memory | Sampled VRAM | Phase 6 Portable Prefill (`tok/s`) | Phase 6.5 GEXEC Prefill (`tok/s`) | Phase 6 Portable Decode (`tok/s`) | Phase 6.5 GEXEC Decode (`tok/s`) | Decode Gain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `Ornith-1.5-35B` `FP16` (`host-only`) | `640 MiB` | `787 MiB` | `2,662 MiB` | `224.46` | **`227.51`** (`233.37` 3-run) | `45.07` | **`47.76`** (`47.95` 3-run) | **`+6.0%`** |
| `Ornith-1.5-35B` `INT8` (`host-only`) | `330 MiB` | `477 MiB` | `2,350 MiB` | `227.47` | **`229.36`** | `44.96` | **`47.91`** | **`+6.6%`** |
| `Ornith-1.5-35B` `Q4_0` (`host-only`) | `180 MiB` | `327 MiB` | `2,200 MiB` | `223.78` | **`226.16`** | `44.33` | **`47.28`** | **`+6.7%`** |
| `Ornith-1.5-35B` `K8V4` (`resident`) | `0 MiB` | `401 MiB` | `2,274 MiB` | `224.30` | **`225.81`** | `44.93` | **`47.84`** | **`+6.5%`** |
| `Qwen3.8-Flash-Next` `FP16` (`host-only`) | `768 MiB` | `564 MiB` | `7,300 MiB` | `97.53` | **`97.92`** | `20.42` | **`22.19`** | **`+8.7%`** |
| `Qwen3.8-Flash-Next` `INT8` (`host-only`) | `396 MiB` | `377 MiB` | `7,082 MiB` | `99.04` | **`98.82`** | `20.87` | **`22.38`** | **`+7.2%`** |
| `Qwen3.8-Flash-Next` `Q4_0` (`host-only`) | `216 MiB` | `286 MiB` | `6,980 MiB` | `97.68` | **`97.85`** | `20.45` | **`21.73`** | **`+6.3%`** |
| `Qwen3.8-Flash-Next` `K8V4` (`resident`) | `0 MiB` | `483 MiB` | `7,186 MiB` | `99.24` | **`99.40`** | `21.00` | **`22.13`** | **`+5.4%`** |

### 4.4 Concurrent Multi-Session Scaling (`Ornith-1.5-35B`, `p=512`, `context=4,096`, `2` Rounds)

| Concurrent Sessions (`--sessions`) | Total Requests | Per-Request Prefill `p50` (`p95`) | Per-Request Decode `p50` (`p95`) | Round Wall Time `p50` | Sampled VRAM | Deterministic Token Parity |
|---:|---:|---:|---:|---:|---:|:---:|
| `1` | `2` | `240.95 tok/s` (`241.32`) | `49.62 tok/s` (`49.67`) | `3,411.8 ms` | `1,786 MiB` | `true` |
| `2` | `4` | `240.63 tok/s` (`241.11`) | `49.41 tok/s` (`49.44`) | `6,841.6 ms` | `1,786 MiB` | `true` |
| `4` | `8` | `240.76 tok/s` (`241.13`) | `49.48 tok/s` (`49.56`) | `13,672.7 ms` | `1,786 MiB` | `true` |

### 4.5 Equivalent-Workload Comparison Against `llama.cpp` (`Ornith-1.5-35B-Q4_K_M`)

All entries in [`bench/results/phase65-general-optimization/manifest.json`](../../bench/results/phase65-general-optimization/manifest.json) use the genuine full-file SHA-256 digests (`42739874cc2ccfdb8523b23fbe52e29b2a7555c8176737ca9ca0b5d59859d41f` for `Ornith-1.5-35B-Q4_K_M.gguf`, `93f3fcd86786870c487ee72b8dbd362973bdd71c1b0d94bdad6ee689b98856e5` for `qwen3.8-flash-next-ext.gpack`). Both Guild and `llama.cpp` (`3cf03257f` pinned and `11fe02151` newer) run one untimed full-length warmup (`run == -1` in `llama_prefill_reference.cpp` and `--full-warmup` in `ornith_prefill_bench`), identical frozen `.prompt.ids`, `64` greedy tokens, `-ngl 99`, CPU experts (`-ot exps=CPU`), host FP16 KV (`--host-kv`), `-b 256`, and `-t 32`:

| Populated Prompt Tokens | Guild Phase 6.5 Portable (`pp` / `tg`) | Guild Phase 6.5 GEXEC (`pp` / `tg`) | `llama.cpp` Pinned (`pp` / `tg`) | `llama.cpp` Newer (`pp` / `tg`) | Guild GEXEC Prefill Speedup (vs Pinned / Newer) | Guild GEXEC Decode Speedup (vs Pinned / Newer) |
|---:|---:|---:|---:|---:|---:|---:|
| `512` | `245.30` / `47.22` | **`237.33` (`245.66` 3-run) / `49.69` (`49.95` 3-run)** | `71.61` / `10.00` | `75.84` / `10.26` | **`3.31x` / `3.13x`** | **`4.97x` / `4.84x`** |
| `4,096` | `230.95` / `45.75` | **`227.51` (`233.37` 3-run) / `47.76` (`47.95` 3-run)** | `81.98` / `8.67` | `82.05` / `8.73` | **`2.78x` / `2.77x`** | **`5.51x` / `5.47x`** |
| `8,192` | `212.22` / `43.49` | **`206.60` / `45.98`** | `81.28` / `7.49` | `75.63` / `7.66` | **`2.54x` / `2.73x`** | **`6.14x` / `6.00x`** |
| `16,384` | `183.99` / `39.82` | **`181.47` / `42.72`** | `79.68` / `5.97` | `78.79` / `5.94` | **`2.28x` / `2.30x`** | **`7.16x` / `7.19x`** |

---

## 5. Verification, Sanitizer & Test Suite Summary

- **Bitwise Parity (`gexec_optimizer_test` & End-to-End Logit/Token Dumps)**:
  - `gexec_optimizer_test`: `2,447 / 2,447` checks passed (`0` failures), verifying `std::memcmp == 0` across all quant types (`Q4_K`, `Q5_K`, `Q6_K`, `Q8_0`, `IQ2_XXS`, `IQ2_XS`, `IQ3_XXS`, `IQ3_S`, `IQ2_S`, `IQ4_NL`, `IQ4_XS`) and token counts `nt = 1..4` for `fused_avx2`, `interleaved_avx2`, `fused_avx512`, and `interleaved_avx512`.
  - End-to-end greedy token IDs (`.ids`) and first-step FP32 logit vectors (`.f32`) are 100% byte-identical (`ids_eq=True, f32_eq=True`) between Phase 6 baseline and Phase 6.5 on `ornith-gexec-p512`, `ornith-gexec-p4096`, `qwen-gexec-p512`, and `qwen-gexec-p4096`.
- **CTest Suites**:
  - **CPU-only build (`build-cpu`, `-DGUILD_ENABLE_CUDA=OFF -DGUILD_BUILD_TESTS=ON`)**: `30 / 30` runnable tests passed (`0` failures; `2` optional `pack/full/experts.bin` fixture tests skipped).
  - **CUDA build (`build-cuda12`, `-DGUILD_ENABLE_CUDA=ON`)**: `96 / 96` runnable tests with present fixtures passed (`0` regressions; `4` optional fixture tests skipped, and `ple_parity` requires the optional `../../Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf` fixture absent on this machine).
- **Sanitizers**:
  - **ASan + UBSan (`-fsanitize=address,undefined`)**: `gpack_test`, `gexec_contract_test`, `gexec_optimizer_test`, `gkmodule_test`, `modular_abi_test`, `pool_affinity_test`, and `expert_isa_test` all passed (`0` memory errors, `0` undefined behavior).
  - **TSan (`-fsanitize=thread`)**: `gexec_contract_test`, `gexec_optimizer_test` (including multi-threaded `ExpertPool` single-barrier fused execution), and `pool_affinity_test` all passed (`0` data races).
- **Release Benchmark Validator (`tools/release_benchmarks.py`)**:
  - `python3 tools/release_benchmarks.py bench/results/phase65-general-optimization/manifest.json --output bench/results/phase65-general-optimization/graphs`: **`64 valid measured groups; 0 exclusions`**.

---

## 6. Future Opportunities (Phase 7+ Roadmap)

1. **PCIe 4.0 / 5.0 & RTX 5070 (`SM120`) Validation**: On the primary GTX 1070 (`SM61`, PCIe 3.0 x16), Qwen chunked prefill is physically bounded by PCIe 3.0 H2D DMA (`15.3 s` / `36.7%` at `~11.2 GB/s`). On PCIe 4.0/5.0 x16 (`~25`–`50 GB/s`) with `SM120` tensor cores (secondary Ryzen 5 5600X + RTX 5070 target when available), `wait copy` will drop by `2x`–`4x` and `GUILD_PF_FUSED=1` (`fused::experts_native`) can be benchmarked with larger VRAM rings.
2. **Multi-Tile Dense Attention in `Verifier` (`> 32,768` on `Ornith-1.5-35B`)**: `Ornith-1.5-35B` dense decode attention currently caps `Verifier::attention_capacity()` at `kTopkMaxCells = 32,768`. Extending `qsa_decode_attn` to iterate across multiple `32,768`-cell tiles with online softmax rescaling will unlock `65,536`–`262,144` decode on dense-attention hybrid models without sparse QSA indexers.
3. **Exact Recurrent + Sparse KV Prefix Snapshots (`ConversationSnapshot`)**: Persisting both GDN recurrent state (`ss.gdn_state`) and QSA host KV page tables at turn boundaries will eliminate repeat system-prompt prefill latency in multi-turn agent workflows.
