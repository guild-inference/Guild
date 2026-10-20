# Phase 6: Long-Context Optimization, GEXEC Selection & Reference Benchmarks

Phase 6 profiles and optimizes `Qwen3.8-Flash-Next` (`UD-IQ4_XS`) and `Ornith-1.5-35B` (`Q4_K_M`) across populated context lengths (`512` through `65,536` tokens on an 8 GiB GPU), resolves long-context KV staging and NUMA DMA bottlenecks, extends GEXEC CPU expert kernels (`src/kernels/cpu/gexec_expert.cpp`) and `BatchedPrefill` (`src/runtime/batched_prefill.cpp`), and adds reproducible populated-context and `llama.cpp` reference benchmarking and SVG release-graph tooling (`tools/bench_context_curve.py`, `tools/release_benchmarks.py`, `tools/llama_reference/`).

---

## 1. Hardware & Measurement Methodology

### 1.1 Primary Measured Hardware
- **Host CPU**: 2x Intel Xeon Gold 6130 (32 physical cores / 64 threads, Skylake-SP, `avx2`, `fma`, `f16c`, `avx512f`, `avx512dq`, `avx512cd`, `avx512bw`, `avx512vl`; no `avx512_vnni`), 376 GiB DDR4 across 2 NUMA nodes (`node 0`: CPUs `0-15,32-47`; `node 1`: CPUs `16-31,48-63`).
- **GPU**: NVIDIA GeForce GTX 1070, 8,192 MiB VRAM, Compute Capability `6.1` (Pascal), attached to PCIe root complex on **NUMA node 0** (`0000:93:00.0`), driver `580.178.04`, CUDA 12.0 (`nvcc 12.0.140`).
- **Secondary Target (`Ryzen 5 5600X + RTX 5070`)**: Not attached to this host; reported as pending hardware access rather than extrapolated.

### 1.2 Populated-Context Rules
- Every benchmarked context length `P` (`512`, `4,096`, `8,192`, `16,384`, `32,704` / `32,768`, `65,536`) ingests `P - 1` real prompt tokens through prefill and decodes `64` greedy tokens starting from position `P - 1` (`--ignore-eos`).
- Allocated KV capacity (`--context`) is never reported as populated history. Cases where `prompt + generated > capacity` or `prompt + generated > attention_capacity()` are recorded as `unsupported` in `*-curve.json`, never as zero-throughput samples.
- Raw CSVs, `.startup.json` phase breakdowns, `.command.json` provenance records, frozen `.prompt.ids`, and generated SVG curves are stored in [`bench/results/phase6-long-context/`](../../bench/results/phase6-long-context/).

---

## 2. Root Causes Identified & Optimizations Implemented

### 2.1 Shared-Staging KV Invalidation in `Prefill::run_impl` (`src/prefill/prefill.cpp`, `src/core/layer.cpp`)
- **Root cause**: At context capacities `>= 65,536` on an 8 GiB GPU, Qwen's 12 QSA layers share a single VRAM staging pool (`st.shares_staging == true`, `33.125 MiB` shared across 12 layers instead of `397.5 MiB` dedicated). In `Prefill::run_impl`, `kv_append` ran at the start of each QSA layer while `st.map.page_table` still held slot mappings from the previous QSA layer (`slot >= 0`). Because `kv_append` only writes to the host mirror (`st.host`) when `slot < 0` in `kv_mode == 3`, new KV cells overwrote stale slots without updating `st.host`. Then, inside the `kVerifyMaxT = 2` query tile loop, `core::qsa_kv_resolve` called `kv_stream_invalidate(st.map, stream)` on **every 2-query tile** (`256` times per 512-token chunk), wiping the page table and re-DMAing the entire selected KV history across PCIe on every tile (`0.08 tok/s` at `65,536`–`262,144` capacity).
- **Fix**:
  1. Invalidate `st.map` once per QSA layer immediately before `kv_append` when `st.kv_mode == 3 && st.shares_staging` (`src/prefill/prefill.cpp`), ensuring `kv_append` sees `slot == -1` and writes all new cells to `st.host`.
  2. Add `bool invalidate_shared = true` to `core::qsa_kv_resolve` (`include/guild/core/layer.hpp`, `src/core/layer.cpp`) and pass `invalidate_shared = false` inside the per-tile loop of `Prefill::run_impl` so tiles reuse resolved staging slots across the 512-token chunk.
  3. Account for `mtp_margin` and `plan.prompt_scratch_bytes` (`>= 1024 MiB`) when deciding whether per-layer QSA staging fits in 8 GiB VRAM (`src/runtime/model.cpp`), preventing `Verifier::init` / `Prefill::init` VRAM exhaustion at `--context 65536` and `131072` when MTP is enabled.

### 2.2 NUMA-Local Pinned DMA Buffer Allocation (`src/prefill/prefill.cpp`, `src/runtime/batched_prefill.cpp`)
- **Root cause**: `GuildSession::generate` pins the request thread to `pool->host_core()` (CPU 0 on NUMA node 0, local to GPU 0), but `GuildModel::load` called `Prefill::init` (`Stager::init` and `m.grp_host` mapped memory) and `BatchedPrefill::init` (`hact`, `hparts`) before any thread affinity was applied. Whenever the OS scheduler placed the loading thread on Socket 1 (NUMA node 1), `cudaHostAlloc` allocated the pinned staging buffers in Socket 1's physical RAM, forcing ~20 GiB of PCIe DMAs per 4K prefill across the cross-socket UPI link (`~94.7 tok/s` instead of `~98.5 tok/s`).
- **Fix**: Scope a `HostAllocAffinity` RAII guard (`pin_current_thread_if_allowed` + `restore_thread_affinity`) strictly around the `cudaHostAlloc` calls in `Stager::init`, `Prefill::init`, and `BatchedPrefill::init`, restoring the full multi-core affinity mask before spawning `Stager::threads`.

### 2.3 `BatchedPrefill` Multi-Format KV Support & 32-Query Attention Tiling (`src/runtime/batched_prefill.cpp`, `src/runtime/model.cpp`)
- **Root cause**: `BatchedPrefill` (`Ornith-1.5-35B`) previously rejected non-`FP16` KV formats (`INT8`, `Q4_0`, `K8V4`), used `kAttentionQueries = 16`, and called `kv_append_step` before `qsa_kv_resolve` on shared-staging layers.
- **Fix**: Support all four KV formats (`FP16`, `INT8`, `Q4_0` via `kv_append_q4_step`, and `K8V4` via `fwht256_inplace_cuda` + `kv_append_step` + `kv_append_q4_step`), resolve KV staging before `kv_append_step` on the first query tile (`q0 == 0`) so shared staging invalidates once per layer before appending, and increase `kAttentionQueries` from `16` to `32` to halve the number of KV resolve/attention launches per 256-token prefill batch.

### 2.4 Bit-Exact GEXEC CPU Expert Kernels (`src/kernels/cpu/gexec_expert.cpp`, `src/gexec/gexec_optimizer.cpp`)
- **Root cause**: On dual Xeon Gold 6130 (`avx512bw` without `avx512_vnni`), `guild_cpu_backend`'s `split_multi_native` runs `kq_avx2.cpp` (512-bit AVX-512BW for `Q4_K`/`Q6_K`) and `iq_avx2.cpp` (256-bit AVX2 for `IQ2_XXS`, `IQ2_XS`, `IQ3_XXS`, `IQ3_S`, `IQ2_S`, `IQ4_XS`). Previously, `gexec_expert.cpp` used `_mm512_reduce_add_ps` for 512-bit I-quants and `hsum_f32_8` at `nt >= 2` for `IQ4_XS` (`type 23`), causing slight floating-point association differences vs `split_multi_native` at `nt >= 2`, and executed `std::getenv` inside the inner row loop of `gexec_iq4xs_gu_rows_nt`. Furthermore, using 512-bit ZMM registers on `Q6_K` down-projections in mixed `UD-IQ4_XS` models triggered Skylake-SP heavy-license AVX-512 frequency downclocking across adjacent AVX2 `IQ4_XS`/`IQ4_NL` layers.
- **Fix**:
  1. Hoist `GUILD_NO_IQ256` and `GUILD_IQ_MT_MIN` environment lookups out of the row loop into `static const` initializers and match `split_multi_native`'s `hsum8_hadd` reduction at `nt >= 2` and `2048`-byte dual cache-line software prefetching in `gexec_iq4xs_gu_rows_nt`.
  2. Fold 512-bit integer accumulators to 256-bit (`_mm512_castsi512_si256 + _mm512_extracti32x8_epi32`) in `iq512_row_dot_2rows` when `!native_iq512_policy(NT)`, achieving bit-exact (`max_abs_err = 0.000000`) parity across all quant types and token counts `nt >= 1` (`2,447 / 2,447` checks in `gexec_optimizer_test`).
  3. Gate 512-bit `Q6_K` down-projection on `cpu_avx512_ok() || (cpu_avx512bw_ok() && (f.gu_type == 12 || f.gu_type == 14))` so pure K-quant models (`Ornith-1.5-35B-Q4_K_M`) run full AVX-512BW while mixed I-quant models (`Qwen3.8-Flash-Next-UD-IQ4_XS`) stay at full AVX2 turbo frequency without Skylake-SP ZMM license downclocking.

---

## 3. Measured Populated-Context Curves

### 3.1 `Ornith-1.5-35B-Q4_K_M` (`--context 32768`, `--batch 256`, `--generate 64`, `--spec 0`)

Note: On `Ornith-1.5-35B` (`is_dense_attention() == true`), `Verifier::attention_capacity()` is bounded by `kTopkMaxCells = 32,768` (`src/core/verify.cpp:418`), so with `64` generated tokens the maximum populated prompt length is `32,704` (`32,704 + 64 = 32,768`). Prompts `>= 32,768` with `64` generated tokens exceed single-tile dense attention capacity and are recorded as `unsupported`.

| Populated Prompt Tokens | Phase 5.6 Baseline Prefill (`tok/s`) | Phase 6 Portable Prefill (`tok/s`) | Phase 6 GEXEC Prefill (`tok/s`) | Phase 5.6 Baseline Decode (`tok/s`) | Phase 6 Portable Decode (`tok/s`) | Phase 6 GEXEC Decode (`tok/s`) | GEXEC Decode Speedup vs Baseline |
|---:|---:|---:|---:|---:|---:|---:|---:|
| `512` | `238.28` | `235.68` | **`237.87`** | `46.59` | `46.14` | **`49.48`** | **`+6.2%`** |
| `4,096` | `225.78` | `224.46` | **`226.65`** | `45.14` | `45.07` | **`47.63`** | **`+5.5%`** |
| `8,192` | `209.57` | `208.69` | **`209.16`** | `43.38` | `43.27` | **`45.77`** | **`+5.5%`** |
| `16,384` | `181.48` | `180.73` | **`182.22`** | `39.71` | `39.55` | **`42.50`** | **`+7.0%`** |
| `32,704` | *(unsupported at `p=32768`)* | `145.84` | **`145.82`** | *(unsupported at `p=32768`)* | `35.31` | **`36.91`** | — |

**Phase breakdown at `p=16,384` (`ornith-gexec-p16384.startup.json` vs `baseline/ornith-spec0-p16384.startup.json`):**
- **Cold model load**: `5,498.6 ms` (`dense_vram_bytes = 1,701.7 MiB`, `expert_ram_bytes = 16,698.0 MiB` NUMA-interleaved, `kv_host_bytes = 640.0 MiB`, `kv_device_bytes = 787.9 MiB`, `vram_sample_bytes = 2,662.2 MiB`).
- **Warmup (`first_inference`, 3 prompt + 4 decode tokens)**: `prompt_ms = 37.9 ms`, `decode_ms = 104.2 ms`.
- **Steady-state prefill (`16,383` tokens)**: `89,905.3 ms` (`182.22 tok/s`), `prefill_gpu_dense_ms = 23,604.2 ms`, `prefill_gpu_attn_ms = 21,622.6 ms`, `prefill_gpu_gdn_ms = 5,551.8 ms`, `prefill_host_wait_ms = 39,077.5 ms` (`prefill_cpu_gu_ms = 23,465.3 ms`, `prefill_cpu_down_ms = 12,175.3 ms`).
- **Steady-state decode (`64` tokens, `64` windows)**: `1,505.8 ms` (`42.50 tok/s`, `23.53 ms/tok`) vs baseline `1,611.8 ms` (`39.71 tok/s`, `25.18 ms/tok`). `decode_cpu_pool_ms` dropped from `572.6 ms` to **`493.8 ms` (`-13.8%`)**, `decode_cpu_gu_ms` dropped from `379.0 ms` to **`277.6 ms` (`-26.8%`)**, and `decode_cpu_validation_ms` dropped from `31.1 ms` to **`4.5 ms`** (single-barrier fused GEXEC pipeline validates output finiteness in-place during the down-projection reduction).

### 3.2 `Qwen3.8-Flash-Next-UD-IQ4_XS` (`--batch 512`, `--generate 64`, `--spec 1`, MTP 2 drafts, `spec_min_p=0.5`)

| Populated Prompt Tokens | Allocated Capacity | Phase 5.6 Baseline Prefill (`tok/s`) | Phase 6 Portable Prefill (`tok/s`) | Phase 6 GEXEC Prefill (`tok/s`) | Phase 5.6 Baseline Decode (`tok/s`) | Phase 6 Portable Decode (`tok/s`) | Phase 6 GEXEC Decode (`tok/s`) | GEXEC Decode Speedup (`cpu_pool+val` delta) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `512` | `32,768` | `95.21` | `94.48` | **`94.34`** | `21.66` | `21.73` | **`22.66`** | **`+4.6%`** (`-13.4%` CPU pool) |
| `4,096` | `32,768` | `98.56` | `97.53` | **`98.21`** | `20.70` | `20.42` | **`21.32`** | **`+3.0%`** (`-11.8%` CPU pool) |
| `8,192` | `32,768` | `95.68` | `95.23` | **`95.56`** | `21.37` | `21.62` | **`22.47`** | **`+5.2%`** (`-14.5%` CPU pool) |
| `16,384` | `32,768` | `93.29` | `92.46` | **`92.94`** | `22.39` | `22.44` | **`23.29`** | **`+4.0%`** (`-12.2%` CPU pool) |
| `32,768` | `65,536` | `87.09` | `91.41` | **`91.56`** (`+5.1%`) | `20.63` | `20.95` | **`21.59`** | **`+4.7%`** (`-12.5%` CPU pool) |
| `65,536` | `131,072` | *(0.08 tok/s or VRAM OOM)* | — | **`89.50`** | *(OOM / timeout)* | — | **`23.05`** | **`1,118x` prefill fix** |

**Phase breakdown at `p=65,536` (`qwen-gexec-c128k-p65536.startup.json`, `--context 131072`):**
- **Cold model load**: `11,297.2 ms` (`dense_vram_bytes = 4,889.1 MiB`, `expert_ram_bytes = 55,432.1 MiB` NUMA-interleaved, `kv_host_bytes = 3,072.0 MiB`, `kv_device_bytes = 733.5 MiB` with automatic shared QSA staging, `vram_sample_bytes = 7,482.2 MiB` on an 8,192 MiB GTX 1070).
- **Steady-state prefill (`65,535` tokens)**: `732,207.2 ms` (`89.50 tok/s`).
- **Steady-state decode (`64` tokens, `33` verify windows, `31/31` MTP drafts accepted = `100%` acceptance)**: `2,776.0 ms` (`23.05 tok/s`, `43.38 ms/tok`), `decode_verify_ms = 2,628.3 ms`, `decode_mtp_draft_ms = 147.7 ms`, `decode_host_wait_ms = 1,271.9 ms`, `decode_cpu_pool_ms = 1,187.3 ms` (`decode_cpu_gu_ms = 891.8 ms`, `decode_cpu_down_ms = 267.2 ms`, `decode_cpu_validation_ms = 4.0 ms`).

**Speculative decoding (`--spec 1` vs `--spec 0`) on `Qwen3.8-Flash-Next` (`baseline/qwen-spec1` vs `baseline/qwen-spec0`):**
- With MTP enabled (`--spec 1`, `spec_min_p = 0.5`), Qwen accepts `26/36` (`72.2%`) to `31/31` (`100%`) offered drafts across `512`–`65,536` contexts, achieving **`21.32`–`23.29 tok/s`** vs **`13.09`–`13.52 tok/s`** without speculation (`--spec 0`), a **`1.63x`–`1.72x` end-to-end decode speedup**.
- Lowering `spec_min_p` from `0.5` to `0.0` (`qwen-spec1-minp0-p4096.startup.json`) increased offered drafts from `36` to `86` while accepted drafts remained `26`, dropping decode throughput from `20.70 tok/s` to `15.83 tok/s` (`-23.5%`) because rejected drafts force 2-token CPU expert windows (`nt=2`) without advancing the sequence. Thus `spec_min_p = 0.5` remains the optimal default.

---

## 4. KV Cache Precision & Placement Comparison (`p=4,096`, `32,768` Capacity)

| Model & KV Configuration | Host KV Memory | Device KV Memory | Total Sampled VRAM | Prefill Throughput (`tok/s`) | Decode Throughput (`tok/s`) |
|---|---:|---:|---:|---:|---:|
| `Ornith-1.5-35B` `FP16` (`host-only`) | `640 MiB` | `787 MiB` | `2,662 MiB` | `224.46` | `45.07` |
| `Ornith-1.5-35B` `INT8` (`host-only`) | `330 MiB` | `477 MiB` | `2,350 MiB` | **`227.47`** | `44.96` |
| `Ornith-1.5-35B` `Q4_0` (`host-only`) | `180 MiB` | `327 MiB` | `2,200 MiB` | `223.78` | `44.33` |
| `Ornith-1.5-35B` `K8V4` (`resident`) | `0 MiB` | `401 MiB` | `2,274 MiB` | `224.30` | `44.93` |
| `Qwen3.8-Flash-Next` `FP16` (`host-only`) | `768 MiB` | `564 MiB` | `7,310 MiB` | `97.53` | `20.42` |
| `Qwen3.8-Flash-Next` `INT8` (`host-only`) | `396 MiB` | `377 MiB` | `7,090 MiB` | **`99.04`** | **`20.87`** |
| `Qwen3.8-Flash-Next` `Q4_0` (`host-only`) | `216 MiB` | `286 MiB` | `6,988 MiB` | `97.68` | `20.45` |
| `Qwen3.8-Flash-Next` `K8V4` (`resident`) | `0 MiB` | `483 MiB` | `7,194 MiB` | **`99.24`** | **`21.00`** |

---

## 5. Equivalent-Workload Comparison Against `llama.cpp`

Measured on the exact same machine, same `Ornith-1.5-35B-Q4_K_M.gguf` weights, identical frozen prompt token ID files (`ornith-gexec-p*.prompt.ids`), greedy decoding (`64` generated tokens), `context = 32768`, `batch = 256`, `threads = 32`, dense layers on GPU (`-ngl 99`), MoE routed experts on CPU (`--cpu-experts` / `-ot exps=CPU`), and host FP16 KV (`--host-kv`), against both:
1. **Pinned `llama.cpp` oracle** (`3cf03257f219afbe7334045ff7c6a06ac68c627d`, `build-cuda12/llama_prefill_reference`), and
2. **Newer `llama.cpp` checkout** (`11fe02151f79c41d0d4af7da708755d73b9c0da6`, built via `tools/llama_reference/CMakeLists.txt` at `build-cuda12/phase6/llama-ref-newer/llama_prefill_reference`).

| Populated Prompt Tokens | Guild GEXEC Prefill (`tok/s`) | `llama.cpp` Pinned Prefill (`tok/s`) | `llama.cpp` Newer Prefill (`tok/s`) | Guild Prefill Speedup (vs Pinned / Newer) | Guild GEXEC Decode (`tok/s`) | `llama.cpp` Pinned Decode (`tok/s`) | `llama.cpp` Newer Decode (`tok/s`) | Guild Decode Speedup (vs Pinned / Newer) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `512` | **`237.87`** | `71.61` | `75.84` | **`3.32x` / `3.14x`** | **`49.48`** | `10.00` | `10.26` | **`4.95x` / `4.82x`** |
| `4,096` | **`226.65`** | `81.98` | `82.05` | **`2.76x` / `2.76x`** | **`47.63`** | `8.67` | `8.73` | **`5.50x` / `5.46x`** |
| `8,192` | **`209.16`** | `81.28` | `75.63` | **`2.57x` / `2.77x`** | **`45.77`** | `7.49` | `7.66` | **`6.11x` / `5.97x`** |
| `16,384` | **`182.22`** | `79.68` | `78.79` | **`2.29x` / `2.31x`** | **`42.50`** | `5.97` | `5.94` | **`7.12x` / `7.16x`** |

---

## 6. Reproducing Curves & Release SVG Graphs

```bash
# 1. Run a populated context curve with frozen prompt IDs and full recorder metadata:
python3 tools/bench_context_curve.py \
  --exe build-cuda12/ornith_prefill_bench \
  --model ornith:35b \
  --gpack /mnt/models-ssd/Strata-data/gpacks/ornith-1.5-35b-ext.gpack \
  --output bench/results/phase6-long-context/optimized \
  --label ornith-gexec \
  --prompts 512 4096 8192 16384 32704 32768 65536 131072 262144 \
  --capacity 32768 --max-context 32768 --generate 64 --runs 1 --batch 256 --spec 0 \
  --extra --gexec build-cuda12/phase6/ornith-c32k-s0.gexec

# 2. Validate manifest, check equivalence/regressions, and generate SVG graphs:
python3 tools/release_benchmarks.py \
  bench/results/phase6-long-context/manifest.json \
  --output bench/results/phase6-long-context/graphs
```
