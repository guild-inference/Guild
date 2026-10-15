# Native Layer-Wise Ornith Prefill and MoE Batch Optimization

Measured on Ubuntu (`Linux 6.8.0-137-generic`), 2× Intel Xeon Gold 6130 (32 physical cores / 64 threads, AVX-512F/BW/DQ/VL, no VNNI), 377 GiB system RAM across 2 NUMA nodes, and 1× NVIDIA GeForce GTX 1070 8 GiB (SM 6.1), driver `580.173.02`, CUDA `12.0` (`V12.0.140`) Release build (`build-cuda12`).

- **Model artifact:** `/home/ubuntu/models/ornith-1.5-35b/Ornith-1.5-35B-Q4_K_M.gguf` (SHA-256 `42739874cc2ccfdb8523b23fbe52e29b2a7555c8176737ca9ca0b5d59859d41f`)
- **Native pack:** `/mnt/models-ssd/Strata-data/packs/ornith-1.5-35b`
- **Reference llama.cpp checkout:** `/home/ubuntu/llama.cpp` at commit `11fe02151f79c41d0d4af7da708755d73b9c0da6`
- **Corrections to earlier baseline notes:** The earlier `bench/results/ornith-baseline/` report's 8K/16K/32K context rows processed 4,096 prompt tokens and its bottleneck percentages / PCIe bandwidth figures were analytical estimates rather than hardware profiler counters. All stage timings and byte counts below come from direct runtime instrumentation (`BatchedPrefillStats` and `GUILD_PREFILL_PROFILE=1` CUDA events).

---

## Architecture and Implementation Summary

1. **Layer-wise chunked prompt ingestion (`BatchedPrefill`):**
   - Known prompt tokens `[0, n_prompt - 1)` are processed in chunks up to `batch_size_` (default auto-selected up to `128`, configurable via `--prefill-batch` / `GUILD_PREFILL_BATCH` in `1..512`, or disabled via `GUILD_PREFILL_SEQUENTIAL=1`) without running the LM head, sampler, or speculative verifier until the final prompt token (`n_prompt - 1`), which enters the unmodified `Verifier` decode path.
   - All GPU scratch buffers and pinned host staging buffers are allocated once during `BatchedPrefill::init()` after checking `cudaMemGetInfo` and preserving the 512 MiB VRAM safety reserve. Zero per-token or per-layer device allocations occur during `run()`.
   - Because `native_mmvq` and `native_quantize_q8_1` enforce `ncols <= 8` (`kVerifyMaxT`), `BatchedPrefill` tiles dense projections and Q8_1 activation quantizations in 8-column slices while keeping the full chunk resident on device across each layer.

2. **Ordered GDN recurrence and causal dense attention:**
   - GDN layers run `gdn_prefill_conv` (`gdn_conv_l2_multi_kernel`) and `gdn_prefill_recurrence` (`gdn_step_norm_multi_kernel`) in strict temporal order over the chunk, preserving exact sequential recurrence arithmetic and updating `ss.gdn_state` in place.
   - Dense causal attention layers apply per-row RoPE (`pos_stride = n_head + n_head_kv`), append each token's K/V into the host-only FP16 KV pool via `kv_append_step`, and evaluate `qsa_decode_attn_batch` in bounded 8-query tiles (`kAttentionQueries = 8`) so split-K scratch remains independent of prompt length.

3. **Grouped MoE execution and parallel intermediate quantization:**
   - Across a `T`-token chunk, routed `(token, rank)` pairs are inverted by expert ID (`0..255`). Active experts in the pinned RAM tier (or copied via `FileExpertSource::copy_blob` when sourced from transient file staging leases) are packed into `ExpertJobMulti` microbatches of up to `MAXT = 8` tokens per job, reading each expert's weights once per microbatch instead of once per token.
   - In `ExpertPool::run_split_multi_native(..., parallel_quant = true)`, when `nb >= 32` expert microbatches are active, intermediate activation quantization (`mode_ == 7`) runs in parallel across worker threads (one independent expert microbatch per task) between the gate/up and down row-split phases, reducing 512-token CPU intermediate quantization time from `333.92 ms` to `21.13 ms`.
   - Expert outputs are written into pinned host memory in `(token, rank)` order, copied back to device, and combined with the GPU shared expert via `native_moe_combine_multi`.

---

## Numerical and State Parity (`ornith_prefill_test`)

`ornith_prefill_test` compares sequential verifier ingestion (`GUILD_PREFILL_SEQUENTIAL=1`) against `BatchedPrefill` on the real Ornith-1.5-35B model at `context_length = 32768`, checking:
- all `248,320` first-decoded-token FP32 logits,
- all `30` layers' GDN recurrent + conv state (`82,808,832` floats),
- all `10` dense attention layers' host-only FP16 K/V cache cells,
- `8` greedy decoded token IDs (`drafts_offered == 0`, `drafts_accepted == 0`), and
- for `n <= 129`, per-layer routed expert IDs, routing weights, and layer residuals across batch sizes `{1, 2, 8, 16, 32, 64, 128, 256, 512}`.

| Prompt Tokens (`n`) | Prefill Tokens (`n - 1`) | Tested Batch Sizes | Logits `max_abs` / `rel_l2` | Logits Bitwise | GDN/Conv State Bitwise | Host KV Identical | Greedy Output Identical | Batched Prefill (`B=128`, tok/s) | Sequential Prefill (tok/s) | Speedup |
| ---: | ---: | :--- | :--- | :---: | :---: | :---: | :---: | ---: | ---: | ---: |
| 1 | 0 | 1, 2, 8, 16, 32, 64, 128, 256, 512 | `0` / `0` | Yes | Yes | Yes | Yes | — | — | — |
| 16 | 15 | 1, 2, 8, 16, 32, 64, 128, 256, 512 | `0` / `0` | Yes | Yes | Yes | Yes | 91.50 | 38.68 | 2.37× |
| 127 | 126 | 1, 2, 8, 16, 32, 64, 128, 256, 512 | `0` / `0` | Yes | Yes | Yes | Yes | 126.26 | 37.80 | 3.34× |
| 128 | 127 | 1, 2, 8, 16, 32, 64, 128, 256, 512 | `0` / `0` | Yes | Yes | Yes | Yes | 126.92 | 37.86 | 3.35× |
| 129 | 128 | 1, 2, 8, 16, 32, 64, 128, 256, 512 | `0` / `0` | Yes | Yes | Yes | Yes | 127.03 | 37.85 | 3.36× |
| 512 | 511 | 1, 2, 8, 16, 32, 64, 128, 256, 512 | `0` / `0` | Yes | Yes | Yes | Yes | 127.20 | 37.52 | 3.39× |
| 2050 | 2049 | 128 | `0` / `0` | Yes | Yes | Yes | Yes | 120.21 | 35.28 | 3.41× |
| 2051 | 2050 | 128 | `0` / `0` | Yes | Yes | Yes | Yes | 119.85 | 35.23 | 3.40× |
| 2052 | 2051 | 128 | `0` / `0` | Yes | Yes | Yes | Yes | 121.01 | 35.42 | 3.42× |
| 4096 | 4095 | 128 | `0` / `0` | Yes | Yes | Yes | Yes | 112.98 | 33.05 | 3.42× |
| 8192 | 8191 | 128 | `0` / `0` | Yes | Yes | Yes | Yes | 101.22 | 29.13 | 3.47× |
| 16384 | 16383 | 128 | `0` / `0` | Yes | Yes | Yes | Yes | 87.50 | 23.72 | 3.69× |

Raw logs and JSON records: `parity-*.log`, `short-parity.json`, and `long-parity.json`.

---

## Batch-Size Scaling and Decode Preservation (`default-batched.csv`, `default-sequential.csv`)

Measured with `ornith_prefill_bench` at `context = 32768`, `prompt = 512` (`511` prefill tokens + `1` verifier head token), `generate = 128`, 3 runs per batch size after warmup, default NUMA policy. All 21 batched runs and 3 sequential runs produced identical 128-token greedy sequences.

| Batch Size | Median Prefill (ms) | Median Prefill (tok/s) | Prefill Speedup vs Sequential | Median TTFT (ms) | Median Decode (ms) | Median Decode (tok/s) | Expert Microbatches (`expert_groups`) | Expert Weight Bytes Read | CPU Gate/Up (ms) | CPU Quant (ms) | CPU Down (ms) | CPU Group Invert (ms) | Host Stream Wait (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Sequential (`0`) | 13560.17 | 37.68 | 1.00× | 13590.74 | 3479.77 | 36.78 | — | — | — | — | — | — | — |
| 8 | 5920.03 | 86.32 | 2.29× | 5950.26 | 3453.08 | 37.07 | 104,869 | 200.26 GB | 2077.17 | 62.50 | 1258.46 | 119.19 | 1615.04 |
| 16 | 5267.80 | 97.00 | 2.57× | 5297.15 | 3420.06 | 37.43 | 83,304 | 159.19 GB | 1818.46 | 35.55 | 1076.16 | 113.86 | 1558.54 |
| 32 | 4434.90 | 115.22 | 3.06× | 4464.10 | 3369.10 | 37.99 | 53,199 | 101.43 GB | 1349.73 | 24.51 | 800.86 | 106.22 | 1546.09 |
| 64 | 4041.96 | 126.42 | 3.36× | 4070.92 | 3334.94 | 38.38 | 37,410 | 71.21 GB | 1131.34 | 38.35 | 663.15 | 101.97 | 1532.39 |
| **128** | **3836.49** | **133.19** | **3.53×** | **3865.08** | **3301.84** | **38.77** | **29,580** | **56.26 GB** | **1032.69** | **20.65** | **599.53** | **99.87** | **1530.27** |
| 256 | 3926.81 | 130.13 | 3.45× | 3955.10 | 3282.25 | 39.00 | 25,822 | 49.15 GB | 1123.02 | 17.84 | 659.19 | 78.34 | 1408.61 |
| 512 | 3955.07 | 129.20 | 3.43× | 3983.27 | 3283.45 | 38.98 | 23,818 | 45.35 GB | 1148.57 | 18.11 | 657.15 | 77.38 | 1408.59 |

- **Expert weight bandwidth reduction:** Grouping active experts across 128 tokens reduces total CPU expert weight reads for 511 prompt tokens from `511 × 40 × 8 = 163,520` single-token expert evaluations (~`312.3 GB`) to `29,580` multi-token expert microbatches (`56.26 GB`, a **5.55× reduction** in expert bytes streamed from RAM).
- **Why throughput peaks at `batch = 128`:** Above `batch = 128`, expert microbatches are already mostly full (`MAXT = 8` tokens per job) and the larger per-layer working set increases LLC misses during CPU gate/up and down projections (`1032.69 + 599.53 = 1632.22 ms` at `B=128` vs `1148.57 + 657.15 = 1805.72 ms` at `B=512`).

---

## Stage Breakdown (`profile.csv`, `GUILD_PREFILL_PROFILE=1`)

Measured at `context = 2048`, `batch = 128`, `prompt = 512` (`511` prefill tokens), `generate = 128`:
- **Total prefill wall time:** `3751.02 ms` (`136.23 tok/s`)
- **GPU dense projections + shared expert (`gpu_dense_ms`):** `1315.96 ms` (`35.1%`)
- **GPU GDN conv + recurrence (`gpu_gdn_ms`):** `142.84 ms` (`3.8%`)
- **GPU dense causal attention + RoPE + KV append (`gpu_attention_ms`):** `144.49 ms` (`3.9%`)
- **CPU routed expert gate/up (`cpu_gu_ms`):** `1027.33 ms` (`27.4%`)
- **CPU routed expert intermediate quantization (`cpu_quant_ms`):** `21.13 ms` (`0.6%`)
- **CPU routed expert down projection (`cpu_down_ms`):** `592.29 ms` (`15.8%`)
- **Host expert grouping & activation quantization (`cpu_group_ms`):** `105.71 ms` (`2.8%`)
- **Activation transfer bytes (D2H + H2D across 40 layers):** `1,508,308,480 bytes` (`1.51 GB`)

---

## Memory Footprint

| Metric | `context = 32768`, `batch_cap = 512` | `context = 2048`, `batch_cap = 128` |
| :--- | ---: | ---: |
| Sampled total VRAM used (`vram_sample_bytes`) | `2,179,137,536 B` (`2.03 GiB`) | `1,956,839,424 B` (`1.82 GiB`) |
| Host-only pinned KV allocation (`host_kv_allocated_bytes`) | `671,098,880 B` (`640.01 MiB`) | `41,953,280 B` (`40.01 MiB`) |
| GPU KV staging pool (`kv_staging_bytes`) | `67,125,248 B` (`64.02 MiB`) | `8,388,608 B` (`8.00 MiB`) |
| Batched prefill device scratch (`prefill_scratch_bytes`) | `212,471,812 B` (`202.63 MiB`) | `148,082,692 B` (`141.22 MiB`) |
| Pinned RAM expert complement (`resident_bytes`) | `19,503,513,600 B` (`18.16 GiB`) | `19,503,513,600 B` (`18.16 GiB`) |
| Process peak RSS (`host_peak_bytes`) | `40.46 GB` | `39.84 GB` |

---

## NUMA-Interleaved and `llama.cpp` Comparison (`context = 2048`, `prompt = 512`, `generate = 128`)

Both engines were given the exact same 512 prompt token IDs (`bench/results/ornith-prefill/prompt-512.ids`).

| Configuration | Median Prefill (ms) | Median Prefill (tok/s) | Median Decode (ms) | Median Decode (tok/s) |
| :--- | ---: | ---: | ---: | ---: |
| Guild sequential (`numactl --interleave=0,1`, `--sequential`) | 11645.11 | 43.88 | 2994.82 | 42.74 |
| **Guild batched (`numactl --interleave=0,1`, `--batch 128`)** | **3874.97** | **131.87** | **2980.82** | **42.94** |
| Guild batched (`GUILD_PREFILL_PROFILE=1`, `--batch 128`) | 3751.02 | 136.23 | 3255.98 | 39.31 |
| `llama.cpp` CPU-only (`-ngl 0 -t 16`, baseline) | 4167.28 | 122.86 | 12355.21 | 10.36 |
| `llama.cpp` partial GPU offload (`-ngl 16 -t 16`, `llama-ngl16.csv`) | 2887.56 | 177.31 (`512/2.888s`) | 11786.50 | 10.86 |

### Independent `llama.cpp` Numerical Comparison

Comparing Guild's post-prefill first-token logits (`248,320` floats) and 128 greedy generated token IDs (`guild.ids`) against `llama_prefill_reference` (`llama-ngl16.ids`) on the identical 512-token prompt (`prompt-512.ids`):
- **Top-1 next token ID:** `760` in both Guild and `llama.cpp`.
- **Greedy continuation agreement:** The first **64 consecutive generated tokens** match identically (`first_mismatch = 64`), and `106 / 128` positions match overall.
- **First-token logit difference:** `max_abs = 4.654204`, `rel_l2 = 0.120015`. Because Guild's batched prefill is **bitwise identical** (`max_abs = 0`, `rel_l2 = 0`) to Guild's sequential verifier on the same prompt, this logit difference is between Guild's native kernel/pack arithmetic and `llama.cpp`'s GGUF graph execution rather than introduced by batched prefill. Full numerical equivalence between Guild and `llama.cpp` on Ornith is therefore not claimed.

---

## Remaining Scope and Limitations

- **Single-tile dense attention staging cap:** Dense attention requests with `n_prompt + max_new_tokens > Verifier::attention_capacity()` (`32,768` cells) are explicitly refused before state mutation (`dense attention request exceeds the allocated single-tile history capacity; multi-tile execution is unavailable`).
- **Ordered GDN recurrence:** `gdn_prefill_recurrence` executes the sequential recurrence across the chunk in one kernel launch per layer (`3.8%` of prefill time) rather than an associative parallel scan, preserving bitwise state parity with sequential decode.

## Reproducibility Commands

```bash
# Build test, benchmark, and optional llama.cpp oracle binaries
cmake -B build-cuda12 -DGUILD_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61 \
  -DGUILD_EXPERIMENTAL_SM60=ON -DGUILD_BUILD_TESTS=ON \
  -DFETCHCONTENT_FULLY_DISCONNECTED=ON \
  -DGUILD_ORNITH_MODEL=/home/ubuntu/models/ornith-1.5-35b/Ornith-1.5-35B-Q4_K_M.gguf \
  -DGUILD_ORNITH_PACK=/mnt/models-ssd/Strata-data/packs/ornith-1.5-35b \
  -DGUILD_LLAMA_REFERENCE_DIR=/home/ubuntu/llama.cpp
cmake --build build-cuda12 --target ornith_prefill_test ornith_prefill_bench llama_prefill_reference prefill_failure_test -j 12

# Run short and long real-model bitwise parity suites (1..16384 tokens)
python3 tools/run_ornith_prefill_validation.py
python3 tools/run_ornith_prefill_validation.py --long

# Run batch-size sweep, sequential control, NUMA-interleaved, and profiled benchmarks
./build-cuda12/ornith_prefill_bench --context 32768 --prompt 512 --generate 128 --runs 3 \
  --write-ids bench/results/ornith-prefill/prompt-512.ids --csv bench/results/ornith-prefill/default-batched.csv
./build-cuda12/ornith_prefill_bench --context 32768 --prompt 512 --generate 128 --runs 3 \
  --sequential --csv bench/results/ornith-prefill/default-sequential.csv
numactl --interleave=0,1 ./build-cuda12/ornith_prefill_bench --context 2048 --prompt 512 --generate 128 --runs 3 \
  --batch 128 --csv bench/results/ornith-prefill/interleaved-batched.csv
numactl --interleave=0,1 ./build-cuda12/ornith_prefill_bench --context 2048 --prompt 512 --generate 128 --runs 3 \
  --sequential --csv bench/results/ornith-prefill/interleaved-sequential.csv
env GUILD_PREFILL_PROFILE=1 ./build-cuda12/ornith_prefill_bench --context 2048 --prompt 512 --generate 128 --runs 1 \
  --batch 128 --csv bench/results/ornith-prefill/profile.csv
```

