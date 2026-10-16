# CPU/GPU workload analysis: Ornith on dual Xeon 6130 and GTX 1070

**Measured report, 2026-10-16.** All 28 full-shaped-warmup benchmark matrix
configurations (`280` measured rows, `10` repetitions per case, `128` generated
tokens per run, context capacity `32,768`, no excluded samples) and all
supporting CUPTI/CPU activity traces, hardware counter captures, assembly
inspections, and parity checks are complete and archived under
`bench/results/hybrid-workload-optimization/`.

## 1. Main findings

- Guild uses the GPU for the dense part of every layer, GDN state updates,
  causal attention, routing, the shared expert, and the output head. On this
  model/configuration, **all routed experts run on the CPU**. Zero GPU expert
  groups does not mean zero useful GPU work.
- Launch batching and caller-safe host affinity deliver consistent gains across
  the full-shaped-warmup matrix (`10` runs/case, `context = 32,768`, `128`
  generated tokens):
  - Across 512-token batch sizes, `final` improves median prefill throughput
    over `baseline` at every batch size: **`B=64`: `197.431` → `207.801 tok/s`
    (`+5.25%`)**, **`B=128`: `212.101` → `227.068 tok/s` (`+7.06%`)**,
    **`B=256`: `228.664` → `234.156 tok/s` (`+2.40%`)**, and **`B=512`:
    `217.035` → `225.979 tok/s` (`+4.12%`)**, while also improving median
    decode throughput across all four batch sizes (`42.540–42.722` →
    `42.977–43.785 tok/s`).
  - Across prompt lengths `512` through `16,384` at `B=256`, `final` improves
    median prefill and decode at every length, reaching **`234.156 tok/s` at
    `512`**, **`220.620 tok/s` at `2,048`**, **`209.275 tok/s` at `4,096`
    (`+2.39%`)**, **`194.436 tok/s` at `8,192` (`+2.28%`, `-960.59 ms`)**, and
    **`169.729 tok/s` at `16,384` (`+2.51%`, `-2,423.50 ms`)**. Across all five
    prompt lengths (`512..16,384`), **`B=256` is the fastest prefill batch size**
    among `{64, 128, 256, 512}`.
  - In the initial ten-sample, short-warmup A/B isolation at `512` tokens,
    batched BF16/router/shared-gate and GDN A/B projections reached **237.935
    prefill tok/s**, versus **228.003** with serial projection launches
    (**+4.36%**); with request-thread affinity enabled on both sides, the
    comparison was **237.994 versus 231.377 tok/s (+2.86%)**.
- CPU expert work and useful GPU work both occupy substantial time. In the
  retained profiled request, the prefill-through-first-token window contained
  **845.091 ms of CPU expert-call wall intervals**, **1,385.245 ms of useful
  GPU activity**, and **126.192 ms of nominal overlap**. During the following decode
  window, useful GPU activity was **1,904.450 ms**, CPU expert-call wall time
  was **1,012.096 ms**, and nominal overlap was **6.554 ms**. Clock-alignment
  sensitivity is reported in section 6. These are instrumented observations;
  profiling substantially perturbs decode.
- Device wait kernels must be separated from compute. The retained decode
  trace has **1,133.626 ms of `wait_*` kernel time**. Counting those waits as
  useful computation would inflate useful GPU activity from 1,904.450 ms to
  3,038.076 ms.
- AVX-512 wins for the tested multi-token CPU expert workload despite its
  lower measured clock. At 256 experts, eight tokens/expert and 32 cores,
  the median iteration was **10.020 ms**, versus **14.358 ms** under Guild's
  AVX2 dispatch cap. The corresponding whole-command retired instruction
  counts were **556.175 billion versus 1,025.210 billion**. Single-token cases
  use a different dispatch path and do not show the same advantage.
- The controlled llama.cpp result does **not** show a GPU-offload win:
  CPU-only reached **122.773 prefill / 13.465 decode tok/s**, partial offload
  **116.175 / 8.248**, and dense-only GPU offload **71.635 / 10.429**. These
  compare llama.cpp configurations. They are not Guild CPU-only measurements.
- Independent existing-dump comparison finds **bitwise-identical first-token
  logits and 128/128 generated IDs** for all six examined Guild variants
  against Guild baseline. All three llama.cpp dumps share top-1 ID 760 with
  Guild, but their first-token logits and later IDs differ; section 12 gives
  numerical errors and zero-based mismatch positions.

Sources: [initial CSVs][raw], [full-warmup matrix report][matrix-report],
[reviewed final activity summary][review-final], [final CPU profile][cpu-final],
[reviewed serial activity summary][review-serial],
[numerical comparison][numerical-result], and the measurement definitions below.

<a id="final-comparison-matrix"></a>

## 2. Final full-shaped-warmup comparison matrix

<!-- FINAL_MATRIX_BEGIN -->

All 28 matrix cases below execute with **one full-shaped warmup run** (`--full-warmup`, warming the exact prompt length and 128-token generation shape prior to run 0) followed by **10 measured repetitions (`n = 10`, `280` total measured rows, zero exclusions)** at `context = 32,768`, `generated_tokens = 128` (`finish = length`), `prefill_tokens = prompt_tokens - 1`, `vram_sample_bytes = 2,145,583,104` (`1.998 GiB`), `host_kv_allocated_bytes = 671,098,880` (`640.0 MiB`), `kv_staging_bytes = 67,125,248` (`64.0 MiB`), `prefill_scratch_bytes = 212,471,812` (`202.6 MiB`), `gpu_groups = 0`, and `file_reads = 0`.

- **Provenance and binary identity:**
  - The 5 saved `B=256` baseline cases (`baseline-p{512,2048,4096,8192,16384}-b256`) were recorded with `GUILD_PREFILL_BATCH_PROJECTIONS=0 GUILD_HOST_AFFINITY=0` on binary SHA-256 `d64290965bfe580156d090a24cdc2e95c1a643136261a8f24ef9dc7160c0c1c4` (commit `29091029e9284651214240706ddaf27948b137b7` plus harness diff `032d5c3c...`, label `old` in [report.md][matrix-report]).
  - All 20 `final` cases (`final-p{512,2048,4096,8192,16384}-b{64,128,256,512}`, default runtime policy) and the 3 additional 512-token baseline batch cases (`baseline-p512-b{64,128,512}`, `GUILD_PREFILL_BATCH_PROJECTIONS=0 GUILD_HOST_AFFINITY=0`) were recorded on the reviewed final binary SHA-256 `a13c606cc6b350a18c02902cbf009d1498e9bfdef7567886d0055d48d5370733` at HEAD `87fc64fc596bb859f6b5f4eca4db992475d115ca` (label `new` in [report.md][matrix-report]).
  - Per-case CSVs, `.command.json` records, and aggregated statistics are stored in [`bench/results/hybrid-workload-optimization/matrix-owner-20261016T033335Z-2eb5322b/`][matrix-dir] and indexed in [`summary.json`][raw-summary].

### 2.1 Prompt 512 batch sweep (`B = 64, 128, 256, 512`, `n = 10` per case)

| Batch | Baseline Prefill tok/s median `[p05, p95]` (MAD) | Final Prefill tok/s median `[p05, p95]` (MAD) | Prefill Delta | Baseline Decode tok/s median `[p05, p95]` | Final Decode tok/s median `[p05, p95]` | Decode Delta | Final Prefill ms | Final Decode ms | Final Wall ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **64** | 197.431 `[193.432, 199.682]` (1.841) | **207.801** `[207.380, 208.563]` (0.287) | **+5.25%** | 42.722 `[42.559, 42.797]` | **43.785** `[43.413, 43.872]` | **+2.49%** | 2,459.087 | 2,923.381 | 5,387.894 |
| **128** | 212.101 `[208.919, 216.221]` (2.477) | **227.068** `[225.966, 228.161]` (0.623) | **+7.06%** | 42.647 `[42.510, 42.686]` | **43.126** `[42.993, 43.273]` | **+1.12%** | 2,250.435 | 2,968.067 | 5,224.880 |
| **256** | 228.664 `[227.678, 229.418]` (0.421) | **234.156** `[232.718, 235.069]` (0.594) | **+2.40%** | 42.540 `[42.329, 42.592]` | **42.977** `[42.811, 43.087]` | **+1.03%** | 2,182.303 | 2,978.331 | 5,164.579 |
| **512** | 217.035 `[210.585, 218.431]` (0.957) | **225.979** `[221.869, 227.886]` (1.900) | **+4.12%** | 42.611 `[42.364, 42.652]` | **43.090** `[42.983, 43.205]` | **+1.13%** | 2,261.272 | 2,970.500 | 5,234.954 |

### 2.2 Context-length scaling at `B = 256` (`512` to `16,384` prompt tokens, `n = 10` per case)

| Prompt Tokens (`Prefill Tokens`) | Baseline (`B=256`) Prefill tok/s `[p05, p95]` | Final (`B=256`) Prefill tok/s `[p05, p95]` | Prefill Delta (`ms` saved) | Baseline (`B=256`) Decode tok/s `[p05, p95]` | Final (`B=256`) Decode tok/s `[p05, p95]` | Decode Delta | Final TTFT ms | Final Wall ms | Timed Activation Payload (B) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **512** (`511`) | 228.664 `[227.678, 229.418]` | **234.156** `[232.718, 235.069]` | **+2.40%** (`-52.42 ms`) | 42.540 `[42.329, 42.592]` | **42.977** `[42.811, 43.087]` | **+1.03%** | 2,208.672 | 5,164.579 | 336,197,120 |
| **2,048** (`2,047`) | 220.245 `[215.768, 222.631]` | **220.620** `[219.363, 224.197]` | **+0.17%** (`-16.05 ms`) | 37.412 `[37.325, 37.515]` | **37.909** `[37.791, 37.978]` | **+1.33%** | 9,307.977 | 12,652.461 | 1,346,762,240 |
| **4,096** (`4,095`) | 204.384 `[203.477, 206.259]` | **209.275** `[208.241, 210.124]` | **+2.39%** (`-468.30 ms`) | 32.425 `[32.375, 32.443]` | **33.041** `[32.886, 33.085]` | **+1.90%** | 19,601.100 | 23,444.178 | 2,694,182,400 |
| **8,192** (`8,191`) | 190.101 `[189.469, 190.460]` | **194.436** `[193.959, 194.699]` | **+2.28%** (`-960.59 ms`) | 25.880 `[25.848, 25.911]` | **26.070** `[26.024, 26.112]` | **+0.73%** | 42,168.909 | 47,040.519 | 5,389,022,720 |
| **16,384** (`16,383`) | 165.572 `[165.369, 165.814]` | **169.729** `[169.298, 170.073]` | **+2.51%** (`-2,423.50 ms`) | 17.750 `[17.731, 17.762]` | **18.047** `[18.017, 18.069]` | **+1.68%** | 96,583.122 | 103,619.831 | 10,778,703,360 |

### 2.3 Complete 28-case matrix dispersion and provenance summary (`280` rows)

| Case ID | Binary | Prompt | Batch | `n` | Prefill tok/s median `[p05, p95]` | Prefill MAD / SD | Decode tok/s median `[p05, p95]` | Decode MAD / SD | Prefill ms median | Decode ms median | TTFT ms median | Wall ms median |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline-p512-b64` | `new` | 512 | 64 | 10 | 197.431 `[193.432, 199.682]` | 1.841 / 2.344 | 42.722 `[42.559, 42.797]` | 0.042 / 0.088 | 2,588.252 | 2,996.096 | 2,614.608 | 5,587.817 |
| `baseline-p512-b128` | `new` | 512 | 128 | 10 | 212.101 `[208.919, 216.221]` | 2.477 / 2.770 | 42.647 `[42.510, 42.686]` | 0.013 / 0.066 | 2,409.232 | 3,001.395 | 2,435.827 | 5,414.081 |
| `baseline-p512-b256` | `old` | 512 | 256 | 10 | 228.664 `[227.678, 229.418]` | 0.421 / 0.634 | 42.540 `[42.329, 42.592]` | 0.051 / 0.106 | 2,234.720 | 3,008.955 | 2,261.718 | 5,248.035 |
| `baseline-p512-b512` | `new` | 512 | 512 | 10 | 217.035 `[210.585, 218.431]` | 0.957 / 3.249 | 42.611 `[42.364, 42.652]` | 0.041 / 0.114 | 2,354.465 | 3,003.920 | 2,381.739 | 5,361.419 |
| `baseline-p2048-b256` | `old` | 2,048 | 256 | 10 | 220.245 `[215.768, 222.631]` | 2.362 / 2.846 | 37.412 `[37.325, 37.515]` | 0.018 / 0.068 | 9,294.460 | 3,421.359 | 9,324.543 | 12,719.686 |
| `baseline-p4096-b256` | `old` | 4,096 | 256 | 10 | 204.384 `[203.477, 206.259]` | 0.679 / 1.066 | 32.425 `[32.375, 32.443]` | 0.015 / 0.025 | 20,035.833 | 3,947.575 | 20,069.845 | 23,988.942 |
| `baseline-p8192-b256` | `old` | 8,192 | 256 | 10 | 190.101 `[189.469, 190.460]` | 0.312 / 0.389 | 25.880 `[25.848, 25.911]` | 0.014 / 0.023 | 43,087.585 | 4,945.844 | 43,130.103 | 48,035.919 |
| `baseline-p16384-b256` | `old` | 16,384 | 256 | 10 | 165.572 `[165.369, 165.814]` | 0.145 / 0.176 | 17.750 `[17.731, 17.762]` | 0.006 / 0.011 | 98,947.940 | 7,211.457 | 99,008.038 | 106,161.839 |
| `final-p512-b64` | `new` | 512 | 64 | 10 | 207.801 `[207.380, 208.563]` | 0.287 / 0.453 | 43.785 `[43.413, 43.872]` | 0.073 / 0.195 | 2,459.087 | 2,923.381 | 2,484.770 | 5,387.894 |
| `final-p512-b128` | `new` | 512 | 128 | 10 | 227.068 `[225.966, 228.161]` | 0.623 / 0.840 | 43.126 `[42.993, 43.273]` | 0.082 / 0.102 | 2,250.435 | 2,968.067 | 2,276.473 | 5,224.880 |
| `final-p512-b256` | `new` | 512 | 256 | 10 | **234.156** `[232.718, 235.069]` | 0.594 / 0.873 | 42.977 `[42.811, 43.087]` | 0.077 / 0.103 | 2,182.303 | 2,978.331 | 2,208.672 | 5,164.579 |
| `final-p512-b512` | `new` | 512 | 512 | 10 | 225.979 `[221.869, 227.886]` | 1.900 / 2.292 | 43.090 `[42.983, 43.205]` | 0.019 / 0.079 | 2,261.272 | 2,970.500 | 2,287.309 | 5,234.954 |
| `final-p2048-b64` | `new` | 2,048 | 64 | 10 | 200.341 `[198.104, 202.253]` | 0.749 / 1.479 | 37.963 `[37.771, 38.018]` | 0.035 / 0.091 | 10,217.597 | 3,371.720 | 10,246.613 | 13,595.811 |
| `final-p2048-b128` | `new` | 2,048 | 128 | 10 | 212.967 `[211.861, 216.269]` | 1.046 / 1.762 | 37.852 `[37.624, 37.973]` | 0.046 / 0.120 | 9,611.815 | 3,381.587 | 9,641.545 | 13,005.621 |
| `final-p2048-b256` | `new` | 2,048 | 256 | 10 | **220.620** `[219.363, 224.197]` | 0.782 / 1.775 | 37.909 `[37.791, 37.978]` | 0.035 / 0.068 | 9,278.411 | 3,376.512 | 9,307.977 | 12,652.461 |
| `final-p2048-b512` | `new` | 2,048 | 512 | 10 | 212.054 `[208.947, 219.905]` | 2.528 / 4.052 | 37.781 `[37.742, 37.905]` | 0.038 / 0.064 | 9,653.230 | 3,387.903 | 9,682.601 | 13,042.582 |
| `final-p4096-b64` | `new` | 4,096 | 64 | 10 | 189.030 `[188.087, 190.036]` | 0.697 / 0.764 | 33.037 `[32.979, 33.158]` | 0.044 / 0.068 | 21,663.268 | 3,874.408 | 21,696.701 | 25,537.808 |
| `final-p4096-b128` | `new` | 4,096 | 128 | 10 | 203.325 `[202.837, 204.964]` | 0.420 / 0.874 | 32.859 `[32.758, 33.004]` | 0.078 / 0.096 | 20,140.153 | 3,895.383 | 20,174.267 | 24,033.517 |
| `final-p4096-b256` | `new` | 4,096 | 256 | 10 | **209.275** `[208.241, 210.124]` | 0.443 / 0.679 | 33.041 `[32.886, 33.085]` | 0.033 / 0.080 | 19,567.534 | 3,873.927 | 19,601.100 | 23,444.178 |
| `final-p4096-b512` | `new` | 4,096 | 512 | 10 | 205.818 `[204.776, 207.555]` | 0.832 / 1.031 | 32.960 `[32.899, 32.980]` | 0.018 / 0.034 | 19,896.268 | 3,883.441 | 19,929.843 | 23,782.139 |
| `final-p8192-b64` | `new` | 8,192 | 64 | 10 | 175.456 `[175.100, 175.575]` | 0.093 / 0.180 | 26.147 `[26.088, 26.184]` | 0.015 / 0.035 | 46,684.102 | 4,895.338 | 46,725.974 | 51,593.502 |
| `final-p8192-b128` | `new` | 8,192 | 128 | 10 | 188.804 `[188.301, 189.108]` | 0.138 / 0.299 | 26.184 `[26.173, 26.199]` | 0.007 / 0.010 | 43,383.568 | 4,888.550 | 43,425.209 | 48,274.696 |
| `final-p8192-b256` | `new` | 8,192 | 256 | 10 | **194.436** `[193.959, 194.699]` | 0.218 / 0.283 | 26.070 `[26.024, 26.112]` | 0.017 / 0.031 | 42,126.996 | 4,909.938 | 42,168.909 | 47,040.519 |
| `final-p8192-b512` | `new` | 8,192 | 512 | 10 | 191.253 `[190.166, 192.296]` | 0.576 / 0.776 | 26.095 `[25.883, 26.126]` | 0.028 / 0.104 | 42,828.042 | 4,905.215 | 42,870.070 | 47,735.034 |
| `final-p16384-b64` | `new` | 16,384 | 64 | 10 | 153.205 `[152.993, 153.341]` | 0.074 / 0.131 | 17.898 `[17.854, 17.911]` | 0.009 / 0.024 | 106,935.252 | 7,151.834 | 106,995.154 | 114,089.259 |
| `final-p16384-b128` | `new` | 16,384 | 128 | 10 | 165.434 `[164.534, 165.650]` | 0.138 / 0.416 | 17.829 `[17.780, 17.848]` | 0.014 / 0.026 | 99,030.174 | 7,179.426 | 99,090.719 | 106,213.980 |
| `final-p16384-b256` | `new` | 16,384 | 256 | 10 | **169.729** `[169.298, 170.073]` | 0.223 / 0.313 | 18.047 `[18.017, 18.069]` | 0.009 / 0.018 | 96,524.444 | 7,092.439 | 96,583.122 | 103,619.831 |
| `final-p16384-b512` | `new` | 16,384 | 512 | 10 | 168.354 `[168.081, 168.855]` | 0.125 / 0.288 | 18.024 `[17.995, 18.060]` | 0.016 / 0.024 | 97,312.808 | 7,101.822 | 97,372.099 | 104,412.506 |

<!-- FINAL_MATRIX_END -->

## 3. Machine, artifacts, and measurement boundaries

### Hardware and software

| Item | Recorded configuration |
| --- | --- |
| CPU | 2 × Intel Xeon Gold 6130, 16 physical cores/socket; 32 physical cores and 64 logical CPUs total |
| CPU ISA | AVX2, FMA, F16C, AVX-512F/BW/DQ/VL; no AVX-512 VNNI |
| RAM / NUMA | Approximately 377 GiB, two NUMA nodes; reported node-distance matrix has local 10 and remote 21 |
| CPU placement | Node 0: logical CPUs 0–15, 32–47; node 1: 16–31, 48–63. Guild reports 31 workers + one participating host thread |
| GPU | NVIDIA GeForce GTX 1070, 8 GiB, Pascal SM 6.1; no Tensor Cores |
| Driver / toolkit | NVIDIA 580.178.04; CUDA compiler 12.0.140. The driver's `nvidia-smi` CUDA compatibility version is 13.0, not the compiler version |
| OS / compiler | Linux 6.8.0-142-generic x86-64; Ubuntu GCC 13.3.0 |
| Build records | Release `build-cuda12`; environment capture includes CMake cache and a ggml CPU compile command with `-O3 -march=native` |
| Independent llama.cpp | `/home/ubuntu/llama.cpp`, revision `11fe02151f79c41d0d4af7da708755d73b9c0da6` |
| Guild's ggml dependency | Captured checkout revision `3cf03257f219afbe7334045ff7c6a06ac68c627d`; distinct from the independent reference checkout |

The source of this table is [environment.json][environment], with runtime pool
and expert-placement confirmation in [host-pinned-b256.stderr.txt][placement].
Nominal CPU frequency, instantaneous `/proc` frequency, GPU utilization sampling,
and the NUMA distance matrix are not throughput or interconnect-traffic counters.
Actual busy CPU clocks are discussed in section 9.

The [assembly snapshot][assembly-result] and refreshed [environment.json][environment]
archive six generated `flags.make` files for the CPU kernels, GPU kernels,
runtime, CPU microbenchmark, request benchmark and bundled ggml CPU target,
along with 9 compiler commands and 11 binary/model asset SHA-256 hashes. In this
build, `kq_avx2.cpp` has `-O3 -DNDEBUG -std=c++20` plus per-source
`-mavx2 -mfma -mf16c`; its AVX-512BW functions add explicit function-target
attributes. Bundled ggml CPU uses `-O3 -march=native`. CUDA targets SM 6.1;
individual native BF16/MMVQ sources have `--use_fast_math`, while that option is
not listed as a per-source override for `verify_kernels.cu`.

### Source and model identity

The starting public HEAD was
`499b61be26a8ba7e619e452003ec89f4910b0fee` with 16 already-modified optimization
source files. That starting working tree was preserved as commit `2909102`
(`29091029e9284651214240706ddaf27948b137b7`), the ABI-safe batched projection
and placement runtime changes were committed as `c4b0746`, and the diagnostic
and benchmark harness suite was committed as `87fc64f`
(`87fc64fc596bb859f6b5f4eca4db992475d115ca`). Each `.command.json` record also
captures working-tree status, diff hash, and binary SHA-256. See the
[measurement-directory provenance][raw-readme] and
[example command record][batch-command].

All engine configurations use the GGUF:

```text
/home/ubuntu/models/ornith-1.5-35b/Ornith-1.5-35B-Q4_K_M.gguf
size:   21,713,463,040 bytes
SHA256: 42739874cc2ccfdb8523b23fbe52e29b2a7555c8176737ca9ca0b5d59859d41f
```

Guild additionally uses the native pack at
`/mnt/models-ssd/Strata-data/packs/ornith-1.5-35b`. Its compatibility manifest,
conversion metadata, dense pack, expert index, and tokenizer hashes are recorded
in [environment.json][environment]. The same GGUF hash does not imply identical
in-memory weight representations or identical arithmetic; see section 12.

### Timing and sample definitions

- The main request contains **512 fixed prompt token IDs** and requests **128
  greedy output tokens**, with MTP/speculation disabled. Existing Guild CSV rows
  report 128 completed tokens, `finish=length`, and zero file reads/GPU expert
  groups for the rows discussed here.
- Guild prefills tokens `[0, 511)` and processes the last prompt token through
  the verifier/head path. Thus prefill throughput is **511 / prefill seconds**,
  not 512 / seconds. Time to first token includes the final prompt-token step.
- The main early comparisons use **context capacity 32,768** and active batch
  256 unless the filename says B128. Capacity is not the number of history cells
  read at every attention step: the actual prompt has 512 tokens.
- The benchmark allocates for a **batch capacity of 512** before selecting the
  active batch. Its 212,471,812-byte scratch allocation is consequently not a
  measurement of a minimal B256-only allocation.
- The initial Guild cases use one warmup with three prompt tokens and four
  output tokens. Ten measured rows exist for the explicitly labeled ten-sample
  cases below. This is a short warmup, not a full-shaped warmup. The later
  `--full-warmup` option warms the actual request shape.
- The current llama.cpp reference uses a full-shaped `run=-1` warmup before its
  measured rows. `--aligned` similarly sends the last prompt token to the decode
  timer, with 511 prefill tokens and 128 decode evaluations for these 128-token
  completions. The warmup mismatch limits early cross-engine comparisons.
- Each displayed median is derived directly from the individual CSV values.
  Median throughput and throughput computed from median time need not be exactly
  equal for an even sample count. Rounded tables do not establish statistical
  significance. The cases were not randomized paired experiments.

Sources: [Guild benchmark][bench-source], [session timing][session-source],
[reference benchmark][llama-source], and each case's `.command.json`/`.csv`.
`summary.json` is not the authority for this draft. The older
`bench/results/ornith-prefill-optimized/README.md` was not used for medians,
context settings, or operation attribution.

## 4. What executes on each processor

### Layer geometry and ordering

The active main network has **40 layers**, hidden width **2,048**, **256 routed
experts/layer**, **eight active experts/token**, and expert/shared intermediate
width **512**. The dense-attention geometry is 16 query heads, two KV heads and
head dimension 256. The architecture reader and actual runtime schedule are in
[archetype.cpp][archetype-source] and [batched_prefill.cpp][prefill-source].

The repeating zero-based pattern is:

```text
layers 0,1,2: GDN       layer 3: dense causal attention
layers 4,5,6: GDN       layer 7: dense causal attention
...                    ...
layers 36,37,38: GDN    layer 39: dense causal attention
```

There are **30 GDN layers and ten dense causal-attention layers**. The code's
`qsa` names refer to the inherited attention implementation; Ornith's active
mechanism here is dense causal attention, without the sparse-attention indexer.
Every layer then has a GPU router, one GPU shared expert, eight routed CPU
experts per token, and a residual update.

### Operation, weight, activation, and transfer map

This table describes the tested no-GPU-expert-cache path. Byte counts are logical
payloads unless identified as CUPTI copy records. Load-time weight uploads are
outside the timed request.

| Operation | Execution and weight location | Activation/state location and synchronization | Timed host/device payload |
| --- | --- | --- | --- |
| Token/position preparation | CPU fills token IDs, positions, and step metadata | Pinned host buffers uploaded once per chunk | Small H2D metadata; separate from the expert-activation counter |
| Embedding | GPU gather from native embedding weights in VRAM | FP32 residual rows remain in VRAM | Token IDs H2D; no per-token embedding vector round trip on this path |
| Attention/GDN input RMS norm and activation quantization | GPU; norm weights in VRAM, Q8_1 activation quantization | FP32 rows and quantized scratch in VRAM | No host transfer |
| GDN QKV/gate and output projections | GPU native quantized matrix/vector or multi-column kernels; native GGUF-format weights in VRAM | Q8_1 inputs, FP32 projected rows in VRAM | No host transfer |
| GDN alpha/beta projections | GPU; pack BF16 weights, FP32 arithmetic | FP32 gates/beta in VRAM; independent groups of eight rows in the batched launch | No host transfer |
| GDN convolution and recurrence | GPU; convolution/norm parameters and FP32 recurrent state in VRAM | Updates preserve token order within the chunk; conv-state commit follows recurrence | No host transfer |
| Dense Q/K/V/output projections, Q/K norm and RoPE | GPU; native projection and norm weights in VRAM | FP32 projected rows in VRAM | Q extraction includes device-to-device copying, not PCIe copying |
| Dense KV append | GPU kernel writes FP16 K/V into mapped pinned host KV memory | Authoritative history in RAM | 2,048 logical bytes/new token/dense layer; mapped writes are not ordinary CUDA memcpy activities |
| Dense KV resolve and attention | GPU resolver stages host history into the shared GPU pool; GPU attention in bounded eight-query tiles | FP16 K/V in host history and VRAM staging; FP32 query/output in VRAM; causal query boundaries | Host-history reads occur through mapped-memory kernels; actual PCIe bytes were not measured |
| Post-attention norm, router projection and top-eight selection | GPU; router BF16 weights in VRAM | FP32 normalized rows, IDs and routing weights copied into pinned CPU staging; host waits for completion | Per token/layer: `4 × (2048 + 2 × 8) = 8,256 B` D2H |
| Expert input quantization and expert grouping | CPU pool/host; resident expert index in RAM | CPU quantizes FP32 input to the expert format's activation type and inverts token/rank entries by expert | No additional device transfer; work follows the D2H completion |
| Routed expert gate/up, SwiGLU, intermediate quantization, down | CPU, native quantized expert weights in resident RAM | CPU activation and per-expert intermediate scratch; FP32 output in pinned `(token, rank)` host rows | No expert-weight upload in this configuration; CPU output transferred after reduction |
| Shared expert gate/up, SwiGLU/quantization, down and scalar gate | GPU; native quantized projections and BF16 scalar-gate weights in VRAM | FP32 shared output in VRAM; submitted before CPU grouping/expert execution | No host transfer; can overlap CPU expert work |
| Routed-output reduction | CPU pool, routing-rank-ordered FP32 FMA reduction of eight expert results | One FP32 hidden row/token in pinned host staging | Per token/layer: `4 × 2048 = 8,192 B` H2D |
| Shared/routed addition and residual update | GPU | FP32 residual stays in VRAM for the next layer | No extra round trip |
| Residual finite validation | Host validation after a D2H copy at each chunk end | Pageable CPU validation vector | Additional `511 × 2048 × 4 = 4,186,112 B` D2H across the prompt |
| Final norm, LM head and greedy token selection | GPU native head/verifier path; head weights in VRAM | Logits in VRAM; selected IDs visible to host | Small output/validation metadata; a requested logit dump adds a separate diagnostic copy |

Implementation sources: [prefill scheduler][prefill-source], [CPU pool][pool-source],
[native CPU experts][native-cpu-source], [shared expert][shared-source],
[KV planning/resolution][kv-source], and [verifier][verify-source].

### Exact prefill activation accounting

For the no-cache CPU-reduction path, the layer exchange is:

```text
D2H = 511 tokens × 40 layers × (2048 FP32 inputs + 8 IDs + 8 FP32 weights) × 4
    = 168,752,640 bytes
H2D = 511 tokens × 40 layers × 2048 FP32 reduced outputs × 4
    = 167,444,480 bytes
sum = 336,197,120 bytes
```

The [per-layer final CSV][layers-final] independently sums to these two values.
The retained trace records 167,491,500 H2D bytes, 168,752,648 pinned D2H bytes,
4,186,112 pageable D2H bytes, and 83,722,240 device-to-device bytes in the
prefill-through-first-token window. Metadata and the final prompt-token step
explain why a trace window is not identical to the narrowly defined activation
counter. Device-to-device bytes are not host-link traffic.

**The older 1,508,308,480-byte activation counter is an accounting overestimate
for this path.** It used `K × N` return rows even though the pre-existing
CPU reduction already returned only `N`. The current conditional counter fixes
that distinction; the 336 MB figure is not evidence of a new 4.49× transfer
reduction caused by the projection-launch experiment. A configuration with GPU
cached experts has a different combine/return path.

Mapped KV reads/writes and mapped decode handshakes are not included in CUPTI
explicit-copy bytes. Neither 336 MB nor the trace's explicit memcpy sum is a
measurement of total PCIe traffic, and dividing either by total request time
does not measure PCIe link bandwidth.

### Resident memory and expert placement

The [placement log][placement] reports all **10,240 routed experts** in an
**18.164 GiB / 19,503,513,600-byte pageable, resident, mlocked RAM complement**,
with **zero pinned expert bytes**, zero GPU-cache slots, and no timed file reads.
The small pinned activation buffers and host KV allocation are separate from
this pageable expert arena. Calling the entire expert complement "pinned RAM"
would be inaccurate for these runs.

The [profiled benchmark row][profile-final] records:

| Quantity | Bytes | Approximate size |
| --- | ---: | ---: |
| Sampled total device memory in use | 2,158,166,016 | 2.010 GiB |
| Process peak RSS | 40,494,292,992 | 37.713 GiB |
| Host KV allocation | 671,098,880 | 640.010 MiB |
| Shared GPU KV staging pool | 67,125,248 | 64.016 MiB |
| Prefill device scratch, capacity 512 | 212,471,812 | 202.629 MiB |

RSS includes mappings and other runtime memory, not only the resident expert
arena. Sampled total device use is not an allocation-by-allocation live peak.

## 5. Critical path and decode handshakes

### Prefill

For each chunk and layer, the main dependency chain is:

```text
GPU mixer/projections → GPU router → D2H activations/IDs/weights → host sync
                                                               |
                          +------------------------------------+----------------+
                          |                                                     |
                 GPU shared expert                              CPU grouping → routed experts
                          |                                                     |
                          +---------------- H2D reduced CPU output -------------+
                                               |
                                     GPU add/residual → next layer
```

The shared expert is enqueued after the routing D2H synchronization, then runs
while the host groups and evaluates routed experts. Layers are dependent on
the preceding layer's combined residual. GDN recurrence is temporally ordered;
the new independent projection groups do not turn it into a parallel scan.

The CPU's `host_wait_ms` is time inside `cudaStreamSynchronize` before expert
work. It overlaps GPU work and copies. Adding it to GPU event times would
double-count that part of the critical path. CPU phase wall times, GPU activity
times and request wall time are likewise not an additive utilization pie chart.

### Decode

The verifier replays a captured graph. A GPU doorbell kernel publishes a layer's
normalized activations and selected expert IDs to mapped host buffers. The host
observes the sequence, evaluates CPU experts, publishes the plan/completion
flags with the required ordering, and the GPU reads the returned expert rows.
GPU shared work can run before the combine waits. Sources:
[verify.cpp routing/shared/return path][verify-source] and its host pool loop.

For a single token in this all-CPU-expert configuration, the logical data payload
includes **8,224 B/layer** for the FP32 hidden row plus eight IDs published to
the CPU, and **65,536 B/layer** for eight FP32 expert-result rows read back by
the GPU, in addition to plan/flag metadata. Decode keeps the routed parts for
the GPU combine; the prefill CPU-reduced 8,192 B return should not be applied to
decode. Routing weights remain on device in this local non-remote path.

The `wait_flag_ge`/`wait_flag_ge_or` kernels spin on host-visible flags. Their
duration measures a dependency wait, not multiply/add throughput. It can include
CPU computation, host scheduling, mapped-memory visibility and synchronization.
The host also spins while waiting for the GPU doorbell. Neither spin interval
alone diagnoses a memory-bandwidth bottleneck.

In the measured retained trace the post-first-token phase has 127 one-token
network evaluations. The benchmark's 128-token decode metric includes the
final prompt-token evaluation/first output in its own timer. The trace phase
markers deliberately split at the first emitted token, so these two decode
durations have different boundaries.

## 6. What the profiling numbers mean

| Instrument | Meaning | Important boundary |
| --- | --- | --- |
| Benchmark request CSV | End-to-end prefill, decode, TTFT and wall observations | Unprofiled rows are the performance comparison; model load is reported separately |
| `GUILD_CPU_PROFILE` | Per-call start/end times and sums of participating threads' actual phase CPU time | Linux uses `CLOCK_THREAD_CPUTIME_ID`; CPU milliseconds sum across threads and can exceed elapsed milliseconds |
| Existing `cpu_gu_ms`, `cpu_quant_ms`, `cpu_down_ms` | Legacy phase-wall accounting | Whole-expert mode apportions total wall time using the host's sampled phase proportions, or a fallback; these are not summed worker CPU times |
| `GUILD_PREFILL_PROFILE_CSV` | Per-chunk/layer GPU event intervals, CPU expert/group/combine wall intervals, host waits and activation payloads | GPU event intervals cover all intervening stream work; the dense bucket is not a pure dense-projection kernel sum |
| Optional CUPTI collector | Concurrent kernel intervals, memcpy activity and CUDA runtime API intervals | Useful GPU union excludes spin/wait kernels; explicit memcpy records exclude mapped-memory traffic |
| `perf stat` core counters | Cycles, retired instructions, cache misses and scheduling counters for the launched command | Includes setup and warmup, not only the timed expert loop |
| Per-socket uncore IMC counters | DRAM read/write traffic of the whole machine during sampling intervals | Not process-exclusive, not UPI traffic, not logical expert bytes |
| `turbostat` | Busy-time effective CPU MHz | Startup/idle samples must be separated from sustained work |

The [CPU pool implementation][pool-source] writes true per-thread phase sums;
the legacy benchmark columns should not be relabeled as that new measurement.
The Linux result is the relevant clock implementation here; implementation
changes on other platforms have not been revalidated by these recorded runs.

The [CUPTI collector][collector-source] is an optional development library loaded
through benchmark `--activity`; it is not a mandatory inference dependency.
The [summarizer][summarizer-source] unions overlapping intervals, clips them to
phase markers, and can align CPU intervals with `--cpu`. In the newer traces,
each marker's `bytes` field holds the host steady-clock timestamp paired with
the CUPTI timestamp. The older `profile-gpu.summary.json` derives from a trace
without that calibration and must not supply CPU/GPU overlap claims.

The final and serial raw traces contain no `error` or `dropped` records. This
does not turn kernel duration into SM utilization, achieved memory bandwidth,
or a counterfactual speedup. Name-based kernel categories are descriptive;
category duration sums can double-count concurrent kernels and are not the
union columns below.

### Observed CPU/GPU overlap

The tables below use the stable reviewed summaries
[profile-final-gpu.tooling-review.summary.json][review-final] and
[profile-serial-gpu.tooling-review.summary.json][review-serial]. Their CPU/GPU
alignment uses the **median CUPTI-minus-host offset across three markers**,
rather than the first marker alone. CPU core-millisecond sums and expert-call
wall unions are unchanged by this correction for these fully contained calls.

| Trace / phase | Phase wall ms | GPU kernel+copy union ms | Useful GPU union ms | CPU expert-call wall union ms | CPU expert/useful-GPU overlap ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Retained, prefill + first token | 2,209.724 | 1,394.498 | 1,385.245 | 845.091 | 126.192 |
| Retained, following decode | 3,690.118 | 3,038.076 | 1,904.450 | 1,012.096 | 6.554 |
| Serial projections / host affinity off, prefill + first token | 2,560.926 | 1,430.756 | 1,421.012 | 896.196 | 116.019 |
| Serial projections / host affinity off, following decode | 3,576.563 | 3,132.698 | 1,898.744 | 1,066.798 | 0.004 |

| Clock-alignment sensitivity of overlap | Nominal ms | Range over observed marker offsets, ms |
| --- | ---: | ---: |
| Retained, prefill + first token | 126.192108 | 126.168249–126.206265 |
| Retained, following decode | 6.554276 | 5.525114–7.170444 |
| Serial, prefill + first token | 116.018999 | 116.016269–116.042009 |
| Serial, following decode | 0.003625 | 0.003555–0.004215 |

The retained marker offsets span 384 ns; serial offsets span 330 ns. Small
shifts can accumulate over thousands of clipped call/activity boundaries.
These ranges describe **clock-alignment sensitivity**, not statistical
confidence intervals or guaranteed measurement-error bounds. In particular,
the serial decode overlap is effectively negligible at this resolution.

The serial trace changes host affinity as well as projection batching and has
much larger profiled grouping time. It is useful for launch counts and schedule
inspection, not a clean one-variable causal timing comparison.

| True CPU expert phase sums | Gate/up CPU ms | Intermediate quantization CPU ms | Down CPU ms |
| --- | ---: | ---: | ---: |
| Retained, prefill + first token | 16,726.142 | 366.846 | 9,170.185 |
| Retained, following decode | 14,005.979 | 79.837 | 8,503.150 |
| Serial, prefill + first token | 17,851.112 | 390.480 | 9,550.446 |
| Serial, following decode | 14,062.463 | 104.623 | 8,592.420 |

For example, 26,263.173 summed CPU ms over 845.091 ms of expert-call wall time
is about 31.1 core-equivalents during those intervals. It is not 31.1 cores
busy throughout the whole request, and it is not a vector-unit utilization
measurement. CPU grouping, combine, host waits and pool overhead are outside
these phase CPU sums.

**Instrumentation overhead is material, on the order of 20% wall time.** The
retained profiled request wall was 5,899.835 ms versus a 5,115.103 ms median in
the early unprofiled pinned case, about +15.3%; decode was 3,719.434 versus
2,960.771 ms, about +25.6%. The serial profiled wall was 6,137.479 ms versus
5,287.445 ms in its early unprofiled case, about +16.1%. These are descriptive
cross-run comparisons, not a randomized overhead estimate. Performance claims
should use the unprofiled matrix; profiler results explain the schedule.

### Per-layer observations

The final per-layer CSV contains a two-token warmup plus the measured 511-token
prefill. Filtering `prompt_tokens == 511` leaves two chunks, 256 and 255 rows,
and **80 layer records**. Their totals are:

| Measured prefill interval/counter | Sum |
| --- | ---: |
| GPU dense event intervals | 1,050.639 ms |
| GPU GDN mixer event intervals | 102.953 ms |
| GPU dense-attention mixer event intervals | 46.624 ms |
| CPU expert-call wall time | 840.639 ms |
| CPU combine/submission wall time | 57.500 ms |
| CPU grouping wall time | 16.050 ms |
| Host routing-copy stream waits | 1,167.290 ms |

The dense event intervals include normalization, projection, router and
intervening copy/stream work up to the routing synchronization. They do not
contain the subsequently submitted shared expert as a separately bounded
bucket. Calling this table "GPU dense + shared compute" would misdescribe it.

Across both chunks, the GDN layers' mixer intervals range from **3.383 to
3.569 ms/layer**; the ten dense-attention layers range from **4.618 to 4.779
ms/layer**. Representative rows below are sums over the two chunks, not
single-token latency:

| Layer | Mixer | Dense event intervals ms | Mixer event intervals ms | CPU expert wall ms |
| ---: | --- | ---: | ---: | ---: |
| 0 | GDN | 28.365 | 3.562 | 22.197 |
| 1 | GDN | 28.371 | 3.549 | 20.168 |
| 2 | GDN | 28.406 | 3.569 | 19.906 |
| 3 | Dense causal | 23.925 | 4.779 | 19.895 |
| 7 | Dense causal | 22.855 | 4.640 | 20.184 |
| 15 | Dense causal | 22.770 | 4.635 | 21.119 |
| 23 | Dense causal | 23.112 | 4.699 | 21.820 |
| 31 | Dense causal | 22.668 | 4.634 | 21.126 |
| 38 | GDN | 27.140 | 3.418 | 21.348 |
| 39 | Dense causal | 22.835 | 4.641 | 20.873 |

All layers remain available in [profile-final-layers.csv][layers-final]. The
per-layer CSV does not provide absolute GPU/CPU overlap timestamps; the CUPTI
and CPU traces provide those. Dense attention's history-dependent cost should
be remeasured at longer actual prompts rather than inferred from context
allocation alone.

## 7. Retained changes and initial end-to-end observations

### Independent eight-token projection groups in one launch

The retained BF16 kernel maps independent eight-token groups to `grid.y`.
Each token retains its ordered FP32 FMA and reduction sequence. The GDN A/B
projection uses the same independent-group pattern. This changes launch
organization, not the temporal GDN recurrence or the decode kernels for
windows of at most eight tokens. Sources: [native_bf16.cu][bf16-source] and
[verify_kernels.cu][verify-kernels-source].

For 511 prefill tokens at B256, the traces directly show:

| Prefill-only projection launches | Serial groups | Batched launch groups |
| --- | ---: | ---: |
| BF16 router + shared scalar-gate | 5,120 | 160 |
| GDN alpha/beta | 1,920 | 60 |

The prefill-through-first-token trace additionally contains the unchanged
first-token path: 80 single-row BF16 launches and 30 short GDN A/B launches.
Those should not be counted as failures to batch prefill. In the subsequent
decode window both traces have 10,160 single-row BF16 and 3,810 GDN A/B launches,
consistent with 127 further one-token evaluations.

`ResolvedPrefillKernels` binds `gdn_ab` and `router_projection` through the
[kernel registry][registry-source] while preserving the frozen 7-pointer
`PrefillKernelTableV1` layout. New projection entry points are chained through
`PrefillKernelTableV1::next_ext` via `PrefillKernelExtensionV1` (validated by
`size >= sizeof(PrefillKernelExtensionV1)` and `tag == 0x50464b45u`) so older
ABI v1 plugins without the extension cannot be misread. Per-operation capability
queries (`guild_resolve_prefill_max_tokens_for_op`) advertise up to 512 tokens
for batched projections while keeping `PrefillOp::Attention` at 8 tokens.
`GUILD_PREFILL_BATCH_PROJECTIONS=0` selects the serial control. Kernel count
reduction does not imply a 32× request speedup: work inside the launches, CPU
experts, attention, dependencies and host overhead remain.

### Existing initial CSV medians

These rows have short warmup and ten measured samples each. They are preliminary
observations, separate from the final matrix placeholder. "Serial projections"
still uses batched token ingestion and grouped CPU experts.

| Case / CSV basename | Prefill ms | Prefill tok/s | Decode ms | Decode tok/s |
| --- | ---: | ---: | ---: | ---: |
| Fresh starting optimized baseline, `baseline-b256` | 2,220.460 | 230.133 | 3,013.371 | 42.477 |
| Baseline B128, `baseline-b128` | 2,340.767 | 218.305 | 3,024.049 | 42.327 |
| Global interleave B128, `baseline-interleaved-b128` | 2,294.844 | 222.673 | 2,967.160 | 43.139 |
| Batched projections B256, `batch-projections-b256` | 2,147.643 | 237.935 | 2,963.810 | 43.188 |
| Serial projections B256, `serial-projections-b256` | 2,241.200 | 228.003 | 3,038.278 | 42.129 |
| Host pinned + batching, `host-pinned-b256` | 2,147.111 | 237.994 | 2,960.771 | 43.232 |
| Host pinned + serial projections, `host-pinned-serial-b256` | 2,208.519 | 231.377 | 2,975.947 | 43.012 |

All basenames refer to [the raw measurement directory][raw]. Baseline B256 has
a CSV but lacks its own companion command JSON in this directory; its provenance
is correspondingly less complete than the later captured commands.

The pinned batched result is +3.42% in prefill throughput relative to the fresh
starting optimized baseline. This total combines changes and run conditions.
The pinned batched/serial comparison is the more relevant early projection A/B;
the nearly identical unpinned/pinned batched medians do not establish a separate
large affinity benefit. Decode-rate differences are observed even though these
projection changes preserve the short-window kernels; frequency, placement,
warmup and ordinary run variation remain possible influences.

## 8. NUMA and request-thread placement

The resident expert fallback arena now respects the caller's process memory
policy and the allowed-node mask. On Linux it automatically applies range-only
interleave across allowed nodes when the process policy is `MPOL_DEFAULT` and
multiple nodes are allowed. `GUILD_EXPERT_NUMA=process` retains process/first-touch
behavior; `interleave` explicitly requests interleave within the allowed mask.
This policy concerns the pageable fallback arena, not arbitrary CUDA host
allocations or all process memory. See [expert_source.cpp][expert-source].

| Initial B256 placement case, ten samples | Prefill tok/s | Decode tok/s |
| --- | ---: | ---: |
| Starting arena-interleaved baseline | 230.133 | 42.477 |
| `GUILD_EXPERT_NUMA=process` | 215.544 | 38.586 |
| `numactl --membind=0` + process policy | 220.482 | 38.792 |

The node-0 case binds memory, not the pool's worker CPUs to one socket. These
measurements support retaining the arena-interleaved default on this dual-socket
machine. They do not measure remote-read fraction or establish a universal
interleave policy for other topologies. The B128 global-interleave case also
changes placement of other allocations and is not an expert-arena-only test.

The request thread pins to the pool-reserved host core—core 0 on this
machine—via `pin_current_thread_if_allowed`, which verifies that the target CPU
is permitted in the caller's existing affinity mask before pinning and restores
the caller's prior mask on scope exit via RAII (`ScopedThreadAffinity`).
`GUILD_HOST_AFFINITY=0` is the A/B control. Workers already reserve that physical
core for the host, which also drains expert tasks. Pinning avoids the host
competing with a worker on another selected core; the measured benefit depends
on the surrounding case. Sources: [session.cpp][session-source] and
[pool affinity/host participation][pool-source].

No remote-DRAM/UPI counter result or replica-local expert-placement experiment
is available here. NUMA costs should not be assigned a percentage from these
end-to-end deltas alone.

## 9. CPU streaming, instruction count, and frequency

### Real-expert microbenchmark shape

[cpu_expert_workload.cpp][cpu-workload-source] copies real layer-0 GGUF experts
into a CPU blob array, builds fixed deterministic activation rows, warms ten
iterations, and repeats the same expert jobs. It makes no CUDA calls. These are
controlled CPU expert measurements, not whole-model CPU inference.

The one-expert case repeatedly reuses approximately 2 MB of weights. The
256-expert case has **522,190,848 logical weight bytes/iteration**, much larger
than the CPU LLC, and is the streaming case. Tested core counts are 1 for the
one-expert case and 8, 16, 32 for the 256-expert case; tokens/expert are 1 and 8.
Each available CSV has ten measured repetitions. The microbenchmark's allocated
blob array uses the captured `numactl --interleave=all` launch policy; its
allocation behavior should not be substituted for Guild's full resident arena.

| 256 experts: cores | Tokens/expert | AVX-512-dispatch ms/iteration | AVX2-capped ms/iteration | AVX-512 logical weight GB/s | AVX2 logical weight GB/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 1 | 11.667 | 12.084 | 44.758 | 43.213 |
| 16 | 1 | 8.821 | 8.546 | 59.202 | 61.102 |
| 32 | 1 | 5.519 | 5.381 | 94.624 | 97.045 |
| 8 | 8 | 35.456 | 49.232 | 14.728 | 10.607 |
| 16 | 8 | 18.863 | 26.071 | 27.683 | 20.030 |
| 32 | 8 | 10.020 | 14.358 | 52.116 | 36.370 |

Sources are the `cpu-avx{512,2}-g256-c{8,16,32}-t{1,8}.csv` records in [raw][raw].
Logical GB/s is `weight_bytes_per_iteration / median_iteration_seconds / 1e9`.
It counts an expert's weights once per microbatch, not once for every activation
row, and is **not a DRAM counter**. The one-expert/eight-token medians were
0.809 ms under AVX-512 dispatch and 1.138 ms under the AVX2 cap; the corresponding
one-token cases were 0.191 and 0.186 ms.

### ISA and measured clocks

For 256 experts/eight tokens/32 cores:

| Whole-command counter / sustained clock | AVX-512 dispatch | AVX2 cap |
| --- | ---: | ---: |
| Retired instructions | 556.175 billion | 1,025.210 billion |
| Cycles | 314.929 billion | 567.769 billion |
| Instructions/cycle | 1.766 | 1.806 |
| Sustained busy CPU MHz samples | 1,915–1,922 | Mainly 2,410–2,421; one late busy sample 2,385 |

The AVX-512 case executes about **45.8% fewer retired instructions**, while the
median timed iteration is about **30.2% shorter**. The lower busy clock is real
in the `turbostat` records; a nominal 2.1 GHz label would obscure it. Higher
AVX2 clock and slightly higher IPC do not compensate for the larger instruction
count in this eight-row workload.

The [K-quant source][kq-source] uses AVX-512BW integer pairwise multiply/add
operations with FP32 accumulation/reduction. Existing-object disassembly below
confirms the generated instruction forms. The Xeon 6130 lacks VNNI; no VNNI
dot instructions appear in the eight inspected functions. Vector/port
utilization still requires dynamic counters.

The control is a **Guild dispatch cap**, not a separately rebuilt AVX2-only
ggml library. [native_expert.cpp][native-cpu-source] uses Guild's multi-token
K-quant kernels for `nt >= 2`; single-token jobs fall through to ggml's
`vec_dot`. Guild's bundled ggml was compiled with `-march=native` in this
environment. Consequently the one-token rows do not isolate AVX-512 versus
AVX2 instruction execution. Their nearly unchanged instruction counts are
consistent with the shared fallback path. The full-request forced-AVX2 result
in section 10 is an empirical dispatch-policy comparison with the same caveat.

### Generated assembly: NT4/NT8 dual-row kernels and bundled ggml

[assembly-summary.json][assembly-result] records object hashes, symbol sizes,
instruction/mnemonic/form counts, register references, stack-memory sites,
largest-backward-branch spans and the generated flags. The approximately 17 KB
[assembly-excerpts.txt][assembly-excerpts] archives aligned representative
windows rather than a full objdump. The reproducer is
[tools/research_workload_artifacts.py][research-source]; it invokes
`objdump -d -C --no-show-raw-insn` on the existing objects without building or
running model code.

The inspected objects are:

```text
build-cuda12/CMakeFiles/guild_kernels_cpu.dir/src/kernels/cpu/kq_avx2.cpp.o
build-cuda12/ggml/src/CMakeFiles/ggml-cpu.dir/ggml-cpu/arch/x86/quants.c.o
```

Counts below are **static whole-function instruction sites**, excluding nops.
"Stack R/W" counts vector instructions with explicit `%rsp`/`%rbp` memory
operands. It includes accumulator homes and scratch as well as possible spills;
it does not classify compiler spills separately. "Shuffle" includes byte/lane
shuffle, permute, insert/extract, unpack and selected move-shuffle forms.

| Function | Code bytes | Static instruction sites | `sub rsp` immediate, B | Vector stack R/W sites | Shuffle sites | Referenced SIMD register indices |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| Q4 dual-row AVX-512BW NT4 | 3,306 | 542 | 448 | 32 / 48 | 88 | 0–31 |
| Q4 dual-row AVX-512BW NT8 | 5,953 | 946 | 1,024 | 79 / 98 | 140 | 0–31 |
| Q4 dual-row AVX2, runtime NT | 4,343 | 821 | 1,760 | 94 / 46 | 116 | 0–15 |
| Q6 dual-row AVX-512BW NT4 | 2,694 | 403 | 200 | 17 / 25 | 64 | 0–31 |
| Q6 dual-row AVX-512BW NT8 | 4,529 | 675 | 456 | 34 / 49 | 96 | 0–31 |
| Q6 dual-row AVX2, runtime NT | 5,399 | 949 | 1,824 | 122 / 83 | 56 | 0–15 |
| ggml Q4_K × Q8_K single dot | 803 | 161 | 32 | 0 / 0 | 16 | 0–22 |
| ggml Q6_K × Q8_K single dot | 850 | 142 | 32 | 0 / 1 | 11 | 0–28 |

The AVX2 functions accept runtime NT and are shared by NT4 and NT8 calls; there
are no distinct AVX2 NT4/NT8 specializations to compare. Their guarded blocks
and inner loops mean their static footprint is not a per-call or per-weight-
block execution count. For example, Q4 AVX-512 NT8 has more static sites than
the AVX2 function even though the recorded multi-token workload retires fewer
instructions overall. Unrolling and loop trip counts matter.

Stack pressure is present inside the weight-loop spans, not only at function
entry. Selecting the largest backward-branch address span gives these vector
stack read/write site counts: **Q4 NT4 16/16, Q4 NT8 47/34, Q4 AVX2 62/45;
Q6 NT4 9/8, Q6 NT8 18/16, Q6 AVX2 98/79**. These address spans may contain
guarded paths and nested loops. The archived examples show stack-source
`vfmadd213ps` followed by accumulator stores. Q4 NT8 also has greater static
stack allocation and stack-site counts than NT4. This supports a register/home-
storage tradeoff; it does not measure its time cost.

Specific generated forms establish what the compiler chose:

- For each of Q4 and Q6, dual-row AVX-512BW **NT4 has 32 and NT8 has 64 ZMM
  `vpmaddubsw` sites**. They combine those byte products with ZMM `vpmaddwd`,
  reduce integer partials into YMM values and use XMM/YMM FP32 FMA accumulation.
  Byte unpacking/shuffles are not uniformly 512-bit. Examples also use
  `vinserti64x4`, `vextracti32x8` and, in Q6, `vpternlogq`.
- Q4 NT4/NT8 contain **40/80 total `vpmaddwd` sites**, including XMM
  minimum/scale correction; Q6 NT4/NT8 contain **40/80**, including YMM
  correction. The full JSON breaks selected forms down by explicit operand
  width. Whole-function shuffle counts cannot be read as executed shuffles
  per token.
- All four AVX-512BW dual-row functions reference all **32 architectural SIMD
  register indices**. Q4 source declares two YMM accumulator arrays and two
  XMM correction arrays: at NT8 that is 32 accumulator vectors before weight
  unpacking and dot temporaries; NT4 has 16. Q6 has two YMM accumulator arrays,
  or 16 accumulator vectors at NT8. Those source-level counts explain why
  wider-register code can still keep accumulators on the stack; they do not
  prove all named values are simultaneously live in the emitted schedule.
- The AVX2 dual-row functions reference indices 0–15 and have no ZMM operands.
  In contrast, bundled ggml's Q4/Q6 single-dot functions reference **23/29
  architectural SIMD indices**, including indices above 15, despite having
  **no ZMM operands**. Its generated EVEX/VL forms include extended registers,
  and Q6 uses `vpternlogq`. Thus even its single-dot code is not accurately
  described as an AVX2-only binary on this `-march=native` build.

Architectural register references are not **physical rename-register usage**,
peak live-register counts, or a register-stall counter. The `sub rsp` immediate
also excludes alignment/push effects and may coexist with red-zone accesses;
it is not the number of spilled bytes. Direct-stack counting misses accesses
through stack aliases in other registers. No dynamic spill rate, memory-level
traffic, vector-lane occupancy, port pressure or physical-register-file
occupancy was measured. The existing instruction/frequency measurements and
this disassembly are complementary evidence, not substitutes for those counters.

Object hashes identify this inspection snapshot. A match between every
inspected object and every older benchmark executable has not independently
been established; the pending corrected build requires its own provenance.

### DRAM hardware counters

The `imc-read-write-t{1,8}` commands run a longer streaming workload: 256 experts,
32 cores, 200 iterations/repetition, ten repetitions. `perf stat -a --per-socket
-I 1000` measures `uncore_imc/cas_count_read/` and `cas_count_write/`.

The raw records are scaled **MiB for each reporting interval**, approximately
one second. For a rate, sum S0 and S1 for each event, divide by the difference
between successive elapsed timestamps, then convert MiB to decimal GB. Do not
interpret the already scaled field as an unscaled CAS count and multiply by
64 again. For complete intervals after elapsed time 2 s (`interval > 0.9 s`),
the medians are:

| Workload | Included intervals | DRAM read GB/s | DRAM write GB/s | Logical expert GB/s from the separate timed CSV |
| --- | ---: | ---: | ---: | ---: |
| One token/expert | 10 | 99.764 | 33.169 | 96.554 |
| Eight tokens/expert | 19 | 57.065 | 21.035 | 52.249 |

Sources: [one-token IMC records][imc-one], [eight-token IMC records][imc-eight],
and matching CSVs/commands. The selection omits the startup interval and the
last short interval; it is an elapsed-time filter, not exact application phase
marking. These counters are system-wide and include runtime reads/writes and
any background work. They are not a process-attributed measurement.

The extra writes and the difference between logical weight rate and IMC reads
show why `expert_weight_bytes` must not be labeled measured DRAM traffic.
At eight rows, weight reuse reduces bytes per activation while quantized dot
product/activation work grows. The ISA result shows that instruction cost
matters there. The measurements do not establish a complete memory/compute
roofline or a percentage of theoretical memory-controller saturation.

For the full-model prefill, 163,520 token/expert evaluations
(`511 × 40 × 8`) become **25,822 expert microbatch jobs**, about 6.33 entries/job,
with **49,148,092,416 logical expert-weight bytes**. This is format-dependent
scheduled weight accounting, not 49 GB measured at DRAM or transferred to GPU.

## 10. Rejected trials

The following results are existing unprofiled early CSV medians on the same
machine and 512/128 request. Their trial code was removed. These are reasons
not to retain those implementations on this measured workload, not proof that
the approaches cannot work on other models or GPUs.

| Trial / CSV basename | Samples | Prefill tok/s | Decode tok/s | Decision supported by the observation |
| --- | ---: | ---: | ---: | --- |
| GPU paired rows, `paired-rows-b256` | 10 | 223.430 | 42.385 | Below the retained baseline/batched path |
| GPU column tile 32, `column32-b256` | 3 | 218.678 | 42.350 | No prefill improvement |
| GPU column tile 64, `column64-b256` | 3 | 226.192 | 42.302 | No prefill improvement |
| GPU column tile 128, `column128-b256` | 3 | 228.852 | 43.360 | No prefill improvement; small decode change is not a demonstrated kernel gain |
| CPU groups of 32, `expert32-b256` | 10 | 227.541 | 42.824 | Larger groups did not improve end-to-end prefill |
| CPU single-row trial, `single-row-b256` | 10 | 225.841 | 42.203 | No improvement |
| Forced AVX2 dispatch, `forced-avx2-b256` | 10 | 202.855 | 42.037 | Keep the measured AVX-512-capable multi-token default |

The filenames and [command records][raw] identify each trial. The CPU K-quant
assembly inspected above does not establish the cause of the removed GPU tile
regressions. Without trial-specific GPU assembly, occupancy, cache-traffic and
phase-isolated measurements, those trials' register-pressure/cache explanations
remain hypotheses. Three-sample column trials are particularly limited.

## 11. Controlled llama.cpp comparison and the missing Guild CPU control

The current [reference driver][llama-source] supports `--runs`, `--batch`,
`--aligned`, `--host-kv`, and `--cpu-experts`. For these rows, context is 32,768,
batch/ubatch 256, threads/batch threads 32, FP16 KV, the same prompt-ID file,
greedy generation and one full-shaped warmup. All ten rows per case have 511
prefill tokens and 128 completed tokens/128 decode evaluations.

| Reference case / CSV | GPU configuration | Prefill ms | Prefill tok/s | Decode ms | Decode tok/s |
| --- | --- | ---: | ---: | ---: | ---: |
| `llama-cpu-only.csv` | NGL 0; op/KQV offload disabled; `CUDA_VISIBLE_DEVICES=-1` | 4,162.630 | 122.773 | 9,506.200 | 13.465 |
| `llama-ngl16.csv` | NGL 16; host KV | 4,398.525 | 116.175 | 15,518.200 | 8.248 |
| `llama-dense-gpu.csv` | NGL 99; host KV; routed expert tensors overridden to CPU buffers | 7,133.375 | 71.635 | 12,273.300 | 10.429 |

The high-NGL case deliberately uses `--cpu-experts` to leave routed expert
weights on CPU while offloading eligible dense work. A large NGL number alone
does not mean all model computation or all weights fit on this GPU. Host KV
storage also does not make the engines' cache managers and graph schedules
identical.

**CPU-only is fastest among these three llama.cpp cases.** This controlled
result must be retained even though Guild itself benefits from its hybrid
schedule and projection-launch optimization. Differences between engines
include kernel selection, CPU scheduling, batching, weight representation,
KV handling, graph boundaries and linked ggml revisions. Guild-versus-llama
ratios do not isolate a GPU acceleration factor.

The often-quoted historical **177.31 prefill / 10.86 decode tok/s** row was
**NGL 16, 16 threads, context 2,048**, with the historical prefill convention
using all 512 prompt tokens. It was not an NGL99/32-thread/context32K row.
Its configuration is documented in
[bench/results/ornith-prefill/README.md][historical-report], section
"NUMA-Interleaved and llama.cpp Comparison". It should not be copied into the
new controlled matrix.

Guild has no working whole-model CPU-only runtime control here:
[model_stub.cpp][stub-source] returns `Guild GPU runtime is disabled in CPU-only
build` from model load. A full 40-layer CPU implementation would be a separate
backend project, including GDN, attention/KV, dense projections and head. The
CPU expert microbenchmark and llama.cpp CPU-only run are useful controls with
different scopes; neither is a measurement of Guild with the GPU removed.

## 12. Numerical representation and validation evidence

### Same GGUF, different execution representation

[workload_weight_audit.cpp][audit-source] compares pack BF16 tensors against
the original GGUF data/dequantization. [weight-audit.stdout.txt][audit-result]
contains **142 tensors**: 82 router/shared-gate tensors and 60 GDN alpha/beta
tensors. The audit includes the pack's extra block-40 router/shared-gate entries;
that does not imply 41 layers execute in the active main network.

- The audited original F32 router/shared-gate weights are exactly represented
  by the pack's BF16 values: zero measured difference for these tensors.
- The 60 alpha/beta tensors originate as Q4_K in the GGUF. Guild's pack contains
  BF16 values obtained by rounding their dequantized values. Relative L2
  differences range from **0.00162955 to 0.00167824**, approximately 0.166%.
- There are **zero BF16 round-to-nearest-even conversion mismatches** across
  the audited tensors. That verifies the audited conversion, not equivalence
  between BF16 execution and llama.cpp's quantized graph operations.

This establishes a real weight-representation difference. It is **not proof
that all cross-engine first-logit error is caused by alpha/beta rounding**.
Kernel arithmetic, activations, reduction order and backend differences also
require isolation. The independent comparison below uses existing current
dumps; it does not substitute historical numerical results or validate a later
corrected binary.

### Independent first-token logits and 128 generated IDs

[numerical-comparison.json][numerical-result] was computed directly from the
existing `.f32` files with Python's standard-library `array('f')`, interpreted
as little-endian binary32. Error differences/norms use binary64 arithmetic and
`math.fsum`; relative L2 is
`sqrt(sum((candidate − baseline)^2)) / sqrt(sum(baseline^2))`.
All ten inputs contain exactly **248,320 logits**, **128 numeric output IDs**,
and **zero NaNs or infinities**. Every input's finite-logit top-1 is numeric
token ID **760**. No token text is decoded or emitted.

Guild baseline is the reference for every row. The six Guild comparisons are:

| Candidate dump prefix | Bitwise-equal first-token logits | Numerically equal logits | Max abs / relative L2 | Matching IDs | First mismatch position |
| --- | --- | ---: | --- | ---: | --- |
| `batch-projections` | Yes | 248,320 / 248,320 | 0 / 0 | 128 / 128 | None |
| `serial-projections` | Yes | 248,320 / 248,320 | 0 / 0 | 128 / 128 | None |
| `forced-avx2` | Yes | 248,320 / 248,320 | 0 / 0 | 128 / 128 | None |
| `host-pinned` | Yes | 248,320 / 248,320 | 0 / 0 | 128 / 128 | None |
| `expert32` | Yes | 248,320 / 248,320 | 0 / 0 | 128 / 128 | None |
| `paired-rows` | Yes | 248,320 / 248,320 | 0 / 0 | 128 / 128 | None |

Their logit-file SHA-256 is
`341c9297d6f9a1d1a7a65872f3ebf16311dfe632279d6433cad304f31ea69fc6`,
identical to baseline. The ID files are also byte-identical. These checks
support numerical preservation for the saved request, including the rejected
trials; they are not a model-wide accuracy evaluation.

| llama.cpp candidate vs Guild baseline | First-logit max abs | Relative L2 | Numerically equal logits | Matching IDs | Identical prefix length / first mismatch position, zero-based |
| --- | ---: | ---: | ---: | ---: | --- |
| `llama-cpu-only` | 10.599987984 | 0.2261956903 | 0 / 248,320 | 106 / 128 | 64 / 64 |
| `llama-ngl16` | 10.931639194 | 0.2262743434 | 0 / 248,320 | 103 / 128 | 64 / 64 |
| `llama-dense-gpu` | 5.723895550 | 0.1615394663 | 0 / 248,320 | 125 / 128 | 80 / 80 |

The independent comparisons share top-1, but neither full first-logit numerical
equivalence nor identical full continuations is established between engines.
The dense-offload dump is closer under these metrics for this request; this
alone does not identify an arithmetic error in either engine or attribute the
remaining difference to one weight conversion. Once greedy histories diverge,
later matching positions are agreement statistics, not comparisons under the
same evolving context.

The JSON archives file hashes, counts, nonfinite checks, numeric top-1 IDs,
error norms, match counts and mismatch **positions**, with no decoded strings
or raw continuation-ID arrays. Guild's benchmark captures logits at the first
callback and overwrites dumps on later runs; its benchmark checks generated-ID
consistency across runs. The llama driver writes first-token logits at measured
run 0 but rewrites the ID file each run. These file comparisons do not separately
establish oracle cross-run consistency. They apply to the saved files, not a
future rebuild or resumed matrix row.

### Checks present in the raw record

| Check | Existing result and scope |
| --- | --- |
| Projection-width parity | [validation-projection-widths.stdout.txt][width-validation]: BF16 and GDN A/B bitwise parity at 19 widths through 512, PASS |
| Kernel registry / ABI | [validation-registry.stdout.txt][registry-validation] and `validation-registry-serial-env.stdout.txt`: modular ABI/registry test PASS under both default and `GUILD_PREFILL_BATCH_PROJECTIONS=0`, including `PrefillKernelExtensionV1` (`next_ext`) and per-op token caps |
| Caller-safe thread affinity | `validation-pool-affinity.stdout.txt` (`pool_affinity_test`): verifies `pin_current_thread_if_allowed` respects restricted caller CPU masks and restores affinity on scope exit, PASS |
| Batched-projection real model | [validation-batched-projections.stdout.txt][projection-validation]: prompt 512, batches 1/2/8/16/32/64/128/256/512, zero reported logit/state errors and bitwise matches; KV/greedy checks PASS |
| Short real-model suite | `validation-real-short.stdout.txt` and `short-parity.json`: prompts 1/16/127/128/129/512 PASS |
| Long real-model suite | `validation-real-long.stdout.txt`, `validation-real-long-16384-resume.stdout.txt`, and `long-parity.json`: prompts 2050/2051/2052/4096/8192/16384 all PASS (`exit_code = 0`, B128 for prompts above 512) |
| Broad CTest record | `validation-ctest.stdout.txt`: 95 tests enumerated, six skipped, one failed (`ple_parity`, missing external GGUF fixture); 88 passed |
| Compute Sanitizer | `validation-memcheck.stdout.txt`: Ubuntu `compute-sanitizer` wrapper failed to locate `/usr/lib/nvidia-cuda-toolkit/compute-sanitizer/libsanitizer-collection.so` without `--injection-path`; documented below |

The current [real-model parity test][parity-source] generates eight output
tokens, then copies logits and recurrent/KV state. Its logit comparison is
therefore **after that checked continuation**, not an independently captured
first-token vector. It also compares output IDs and used KV cells; for prompts
up to 129 it checks per-layer routing IDs/weights and residuals. The standalone
benchmark's `--dump` captures the first-token logits at the first callback,
which is the artifact independently compared above. Its first-token capture is
distinct from the parity test's post-continuation capture.

The unit tests (`prefill_projection_test`, `modular_abi_test`, `pool_affinity_test`),
512-token real-model parity across all nine batch sizes, and the 95-test CTest
run were re-executed on the final committed runtime (`c4b0746` / `87fc64f`,
`ornith_prefill_bench` SHA-256 `a13c606cc6b350a18c02902cbf009d1498e9bfdef7567886d0055d48d5370733`).
The multi-hour `short-parity.json` and `long-parity.json` (`2050..16384`) suites
were completed on the pre-extension post-optimization build (`ornith_prefill_test`
SHA-256 `71fb408ca923de386b24d1805d9a16d2eb64b13d651361ca66dd055e59d2fc7d`),
which shares identical CUDA/CPU projection kernels.

## 13. Current blocks and recommended next work

1. **Generalize the GPU expert path before claiming GPU expert residency for
   Ornith.** The measured runtime has zero GPU-cache groups. Its routed expert
   formats include Q4_K/Q6_K down projections, while the native grouped GPU
   down-format list does not cover them. [performance.cpp][performance-source]
   gates cache eligibility through `native_expert_supported`; the format lists
   and predicate are in [iq_kernels.cu][iq-source]. Separately, the larger GPU
   prefill implementation and scratch layouts contain fixed intermediate width
   640 (`src/prefill/prefill.cpp`, `src/prefill/moe_fused_iq.cu`), versus this
   model's 512. Registry selection alone cannot supply missing format/geometry
   support. Add a validated implementation and accurate capability/scratch
   descriptions before evaluating such an offload.
2. **Follow static assembly evidence with actual bottleneck counters.** The
   existing-object inspection now establishes instruction forms, stack-resident
   accumulators, and NT4/NT8 code/storage growth. Measure executed spill/home
   traffic and relevant vector/port or stall events before assigning their
   cycle cost or vector utilization. Measure UPI/remote-memory traffic before
   quantifying NUMA penalties. Instruction counts, clocks and static disassembly
   justify keeping the measured default but do not explain every cycle.
3. **Improve useful overlap, guided by dependencies.** The trace establishes
   limited CPU/useful-GPU overlap, especially in one-token decode. Candidate
   work includes shared-expert scheduling, host launch/doorbell overhead and
   graph handshake simplification. Evaluate useful activity separately from
   flag polling and preserve the ordered combine and recurrent arithmetic.
   An overlap trace alone does not promise an attainable speedup.
4. **Measure real host-link traffic.** Explicit-copy CUPTI records omit mapped
   KV and decode activation/result accesses. Actual PCIe/I/O or appropriate
   device counters are needed for a host-link bandwidth/saturation claim.
   Attention staging should be measured at longer actual histories; allocation
   capacity is not a proxy for bytes read.
5. **Explore deeper GPU batching only with numerical and end-to-end evidence.**
   The retained change reduces launch count without changing per-row arithmetic.
   Larger tiles and paired rows did not improve this Pascal workload. A GEMM
   redesign, different quantization or associative GDN scan would need separate
   parity/accuracy gates and measurements; GTX 1070 has no BF16 Tensor Cores.
6. **External test environment gaps.** The external `ple_parity` GGUF fixture
   (`../../Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf`) is absent
   from this checkout, and the installed CUDA 12.0 `compute-sanitizer` (`2022.4.1`)
   cannot intercept the newer `580.178.04` (CUDA 13.0 ABI) driver (`Target
   application terminated before first instrumented API call`). Current native
   dense-attention execution is also bounded by the validated single-tile
   32,768-cell history cap; the session refuses requests exceeding that
   allocated staging capability.

### Evidence index and recalculation

The directory [bench/results/hybrid-workload-optimization][raw] holds the raw
CSVs, per-command environment/binary records, stderr, deterministic `.gz`
archives of raw `.tsv` and `.telemetry.jsonl` traces (indexed in
`archive-manifest.json`), and validation outputs. The recursive summary is in
[`summary.json`][raw-summary] and the 28-case full-warmup matrix report is in
[`matrix-owner-20261016T033335Z-2eb5322b/report.md`][matrix-report]. A
lightweight median check, run from the repository root, is:

```python
import csv
from pathlib import Path
from statistics import median

root = Path("bench/results/hybrid-workload-optimization")
for name in ("batch-projections-b256", "serial-projections-b256",
             "host-pinned-b256", "host-pinned-serial-b256"):
    with (root / f"{name}.csv").open() as f:
        rows = list(csv.DictReader(f))
    print(name, "samples", len(rows),
          "pp", median(float(r["prefill_tps"]) for r in rows),
          "tg", median(float(r["decode_tps"]) for r in rows))
```

For the aligned llama CSVs, derive per-row rates as
`1000 * prefill_tokens / prefill_ms` and
`1000 * generated / decode_ms` before taking medians. For expert microbenchmarks,
first divide each row's `wall_ms` by its `iterations`. For trace overlap, union
intervals on a common clock rather than summing overlapping kernel durations.

Relevant source locations are linked throughout; they describe the reviewed
working tree and should be pinned to the final revision when publishing the
completed report.

The lightweight existing-object/dump analysis can be regenerated with
`python3 tools/research_workload_artifacts.py`. It archives compact assembly
JSON/excerpts and numerical JSON, records input identities, and checks that
inputs did not change during inspection. It does not build or execute inference.

[raw]: ../../bench/results/hybrid-workload-optimization/
[raw-readme]: ../../bench/results/hybrid-workload-optimization/README.md
[raw-summary]: ../../bench/results/hybrid-workload-optimization/summary.json
[matrix-dir]: ../../bench/results/hybrid-workload-optimization/matrix-owner-20261016T033335Z-2eb5322b/
[matrix-report]: ../../bench/results/hybrid-workload-optimization/matrix-owner-20261016T033335Z-2eb5322b/report.md
[environment]: ../../bench/results/hybrid-workload-optimization/environment.json
[placement]: ../../bench/results/hybrid-workload-optimization/host-pinned-b256.stderr.txt
[batch-command]: ../../bench/results/hybrid-workload-optimization/batch-projections-b256.command.json
[profile-final]: ../../bench/results/hybrid-workload-optimization/profile-final.csv
[trace-final]: ../../bench/results/hybrid-workload-optimization/profile-final-gpu.tsv.gz
[trace-serial]: ../../bench/results/hybrid-workload-optimization/profile-serial-gpu.tsv.gz
[cpu-final]: ../../bench/results/hybrid-workload-optimization/profile-final-cpu.csv
[layers-final]: ../../bench/results/hybrid-workload-optimization/profile-final-layers.csv
[review-final]: ../../bench/results/hybrid-workload-optimization/profile-final-gpu.tooling-review.summary.json
[review-serial]: ../../bench/results/hybrid-workload-optimization/profile-serial-gpu.tooling-review.summary.json
[assembly-result]: ../../bench/results/hybrid-workload-optimization/assembly-summary.json
[assembly-excerpts]: ../../bench/results/hybrid-workload-optimization/assembly-excerpts.txt
[numerical-result]: ../../bench/results/hybrid-workload-optimization/numerical-comparison.json
[imc-one]: ../../bench/results/hybrid-workload-optimization/imc-read-write-t1.perf.txt
[imc-eight]: ../../bench/results/hybrid-workload-optimization/imc-read-write-t8.perf.txt
[audit-result]: ../../bench/results/hybrid-workload-optimization/weight-audit.stdout.txt
[width-validation]: ../../bench/results/hybrid-workload-optimization/validation-projection-widths.stdout.txt
[registry-validation]: ../../bench/results/hybrid-workload-optimization/validation-registry.stdout.txt
[projection-validation]: ../../bench/results/hybrid-workload-optimization/validation-batched-projections.stdout.txt
[historical-report]: ../../bench/results/ornith-prefill/README.md
[bench-source]: ../../tests/server/ornith_prefill_bench.cpp
[session-source]: ../../src/runtime/session.cpp
[prefill-source]: ../../src/runtime/batched_prefill.cpp
[verify-source]: ../../src/core/verify.cpp
[pool-source]: ../../src/kernels/cpu/pool.cpp
[native-cpu-source]: ../../src/kernels/cpu/native_expert.cpp
[kq-source]: ../../src/kernels/cpu/kq_avx2.cpp
[shared-source]: ../../src/kernels/cuda/shared_expert.cu
[bf16-source]: ../../src/kernels/cuda/native_bf16.cu
[verify-kernels-source]: ../../src/kernels/cuda/verify_kernels.cu
[registry-source]: ../../src/kernels/registry.cpp
[kv-source]: ../../src/core/layer.cpp
[expert-source]: ../../src/core/expert_source.cpp
[performance-source]: ../../src/runtime/performance.cpp
[iq-source]: ../../src/kernels/cuda/iq_kernels.cu
[stub-source]: ../../src/runtime/model_stub.cpp
[archetype-source]: ../../src/model/archetype.cpp
[collector-source]: ../../tools/cuda_activity_trace.cpp
[summarizer-source]: ../../tools/summarize_workload_activity.py
[llama-source]: ../../tools/llama_prefill_reference.cpp
[cpu-workload-source]: ../../tools/cpu_expert_workload.cpp
[audit-source]: ../../tools/workload_weight_audit.cpp
[parity-source]: ../../tests/core/ornith_prefill_test.cpp
[research-source]: ../../tools/research_workload_artifacts.py
