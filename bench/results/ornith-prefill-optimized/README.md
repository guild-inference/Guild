# Ornith-1.5-35B Native Prefill Optimization — Historical Artifact Index

This directory preserves the initial 3-run prefill optimization CSVs and parity logs produced during the development of commit `2909102` (`runtime: preserve validated native Ornith prefill optimizations`) relative to baseline commit `499b61b` (`499b61be26a8ba7e619e452003ec89f4910b0fee`).

For the canonical, 10-repetition full-warmup benchmark matrix, CUPTI/CPU overlap trace analysis, hardware IMC DRAM counter measurements, and controlled `llama.cpp` comparisons, see **[docs/performance/CPU_GPU_WORKLOAD_ANALYSIS.md](../../../docs/performance/CPU_GPU_WORKLOAD_ANALYSIS.md)** and **[bench/results/hybrid-workload-optimization/](../hybrid-workload-optimization/)**.

---

## Hardware and Model Provenance

- **Host:** 2× Intel Xeon Gold 6130 (32 physical cores / 64 threads, AVX-512F/BW/DQ/VL, no AVX-512 VNNI), 377 GiB DDR4 across 2 NUMA nodes (`Linux 6.8.0-142-generic`)
- **GPU:** 1× NVIDIA GeForce GTX 1070 8 GiB (Pascal SM 6.1), driver `580.178.04`, CUDA `12.0` (`V12.0.140`), Release build (`build-cuda12`)
- **Model GGUF:** `/home/ubuntu/models/ornith-1.5-35b/Ornith-1.5-35B-Q4_K_M.gguf` (SHA-256 `42739874cc2ccfdb8523b23fbe52e29b2a7555c8176737ca9ca0b5d59859d41f`)
- **Native pack:** `/mnt/models-ssd/Strata-data/packs/ornith-1.5-35b`

---

## Exact Medians of Retained 512-Token CSVs in This Directory

All rows below are exact medians of the 3-run CSV files in this directory (`prompt_tokens = 512`, `prefill_tokens = 511`, `generated_tokens = 128`, short 3-token warmup). Note that `baseline-default-512.csv` and `baseline-interleaved-512.csv` were recorded at context capacity `2,048` (`host_kv_allocated_bytes = 41,953,280`, `vram_sample_bytes = 1,956,839,424` = `1.822 GiB`), whereas `optimized-*.csv` were recorded at context capacity `32,768` (`host_kv_allocated_bytes = 671,098,880`, `vram_sample_bytes = 2,145,583,104` = `1.998 GiB`). Because context allocations differ between those early runs, use `bench/results/hybrid-workload-optimization/` for controlled same-context comparisons.

| CSV File | Batch | Samples | Context Cap | Median Prefill (ms) | Median Prefill (tok/s) | Median Decode (ms) | Median Decode (tok/s) | Load Time (ms) | Sampled VRAM |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline-default-512.csv` | 128 | 3 | 2,048 | 3,696.185 | 138.251 | 3,194.569 | 40.068 | 13,089.403 | 1.822 GiB |
| `baseline-interleaved-512.csv` | 128 | 3 | 2,048 | 3,859.527 | 132.400 | 2,950.068 | 43.389 | 15,733.655 | 1.822 GiB |
| `optimized-default-512.csv` | 128 | 3 | 32,768 | 2,340.857 | 218.296 | 3,021.552 | 42.362 | 7,446.806 | 1.998 GiB |
| `optimized-interleaved-512.csv` | 128 | 3 | 32,768 | 2,323.736 | 219.905 | 3,013.252 | 42.479 | 7,554.629 | 1.998 GiB |
| `optimized-sequential-512.csv` | 0 | 1 | 32,768 | 11,790.891 | 43.339 | 3,023.422 | 42.336 | 7,477.421 | 1.998 GiB |

### Batch-Size Sweep (`optimized-sweep-512.csv`, 3 samples/batch, context `32,768`)

| Batch | Median Prefill (ms) | Median Prefill (tok/s) | Median TTFT (ms) | Median Decode (ms) | Median Decode (tok/s) | Expert Groups | Scheduled Expert Weight Bytes |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 5,496.374 | 92.970 | 5,522.861 | 3,047.237 | 42.005 | 104,869 | 200,255,791,104 |
| 16 | 4,293.547 | 119.016 | 4,319.943 | 3,036.005 | 42.161 | 83,304 | 159,186,419,712 |
| 32 | 3,063.034 | 166.828 | 3,089.349 | 3,033.717 | 42.192 | 53,199 | 101,427,806,208 |
| 64 | 2,540.793 | 201.118 | 2,567.251 | 3,036.109 | 42.159 | 37,410 | 71,209,869,312 |
| 128 | 2,354.409 | 217.040 | 2,380.761 | 3,027.994 | 42.272 | 29,580 | 56,264,638,464 |
| 256 | 2,247.550 | 227.359 | 2,274.272 | 3,042.530 | 42.070 | 25,822 | 49,148,092,416 |
| 512 | 2,310.821 | 221.134 | 2,337.404 | 3,049.226 | 41.978 | 23,818 | 45,353,091,072 |

---

## Methodological Corrections Recorded in Subsequent Commits (`c4b0746`, `87fc64f`)

1. **`activation_transfer_bytes` accounting (`c4b0746`):**
   - In the CSVs in this directory, `activation_transfer_bytes` was logged as `1,508,308,480` bytes because `src/runtime/session.cpp` multiplied the H2D return payload by `K = 8` even when CPU-side MoE reduction (`combine_multi_native`) returned a single reduced hidden vector per token.
   - Commit `c4b0746` fixed `src/runtime/session.cpp` to count `K` uncombined rows only when GPU-cached expert groups exist (`m.ggrp > 0`), yielding the true prefill activation payload of **`336,197,120` bytes** (`168,752,640` B D2H + `167,444,480` B H2D across 511 prefill tokens and 40 layers).
2. **Legacy `cpu_gu_ms`, `cpu_quant_ms`, `cpu_down_ms`, and `gpu_dense_ms` columns:**
   - In whole-expert mode (`mode_ == 8`), `cpu_gu_ms`, `cpu_quant_ms`, and `cpu_down_ms` in the request CSV apportion total CPU expert wall time using the host thread's sampled phase proportions rather than summing per-worker CPU time. Commit `c4b0746` added `GUILD_CPU_PROFILE` (`CLOCK_THREAD_CPUTIME_ID` / `GetThreadTimes`) to record true summed thread CPU milliseconds alongside wall intervals.
   - `gpu_dense_ms` in `GUILD_PREFILL_PROFILE=1` spans the CUDA stream interval up to the routing D2H synchronization (including normalization, dense projections, router, and intervening stream work) and excludes the shared expert, which is launched after the routing D2H sync to overlap with CPU routed experts.
3. **Historical `llama.cpp` reference configuration:**
   - An earlier draft table cited `177.31 tok/s` prefill and `10.86 tok/s` decode as `-ngl 99 -t 32`. That historical run from `bench/results/ornith-prefill/README.md` was actually executed with **`-ngl 16 -t 16 -c 2048`** and divided by `512` rather than `511` prefill tokens. Controlled 10-run `llama.cpp` measurements at `context = 32,768`, `threads = 32`, `batch = 256`, and aligned `511`-token prefill accounting are documented in `docs/performance/CPU_GPU_WORKLOAD_ANALYSIS.md` Section 11.
4. **CPU K-quant reduction arithmetic:**
   - `q4k_dot_*` and `q6k_dot_*` in `src/kernels/cpu/kq_avx2.cpp` use integer pairwise multiply-accumulate (`vpmaddubsw` → `vpmaddwd`) followed by **FP32 (`float`)** scale FMA and exact 8-lane horizontal FP32 reduction (`hsum8_exact`), matching `ggml-cpu`'s single-token `vec_dot` bitwise without FP64 intermediates.
