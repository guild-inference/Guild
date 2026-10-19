# Phase 4 Performance Profiling & Ranked Optimization Report

All measurements in this report were recorded on the test machine before applying Phase 4 optimizations (commit `e60d56f`, Phase 3 baseline):

- **CPU**: Dual-socket Intel Xeon E5-2697A v4 (32 cores / 64 threads total, AVX2 + FMA, 2 NUMA nodes)
- **System RAM**: 251 GiB DDR4 across 2 NUMA nodes (`node0`: 125 GiB, `node1`: 126 GiB)
- **GPU**: NVIDIA GeForce GTX 1070 (8 GiB GDDR5, Compute Capability 6.1, PCIe Gen3 x16)
- **Storage**: NVMe SSD (`/mnt/models-ssd`) for model packs, Qwen3.8 GGUF shards, and GPACK containers

---

## 1. Instrumented Startup Telemetry (`StartupProfile`)

`GuildModel::load` (`include/guild/runtime/types.hpp` and `src/runtime/model.cpp`) records wall-clock latency and resource counters across 22 startup stages:

| Metric | Ornith-1.5-35B (GGUF + pack) | Qwen3.8-Flash-Next (3-shard GGUF + pack) | Qwen3.8-Flash-Next + MTP (`spec=2`) |
| :--- | ---: | ---: | ---: |
| **`total_load_ms`** | **7,252.03 ms** | **38,967.12 ms** | **39,884.37 ms** |
| `numa_expert_pin_ms` | 5,232.29 ms (72.1%) | 33,738.47 ms (86.6%) | 33,962.01 ms (85.2%) |
| `expert_calibration_post_ms` | 0.01 ms | 1,500.83 ms (3.9%) | 1,546.83 ms (3.9%) |
| `native_dense_load_ms` | 312.52 ms (4.3%) | 1,011.15 ms (2.6%) | 1,039.88 ms (2.6%) |
| `prefill_init_ms` | 26.16 ms | 548.14 ms (1.4%) | 547.76 ms (1.4%) |
| `embed_load_ms` | 247.85 ms (3.4%) | 545.86 ms (1.4%) | 541.93 ms (1.4%) |
| `mtp_init_ms` | 0.00 ms | 0.00 ms | 509.79 ms (1.3%) |
| `dense_index_and_pack_ms` | 56.62 ms | 502.06 ms (1.3%) | 557.28 ms (1.4%) |
| `kv_session_init_ms` | 399.29 ms (5.5%) | 254.05 ms (0.7%) | 253.68 ms (0.6%) |
| `tokenizer_load_ms` | 346.18 ms (4.8%) | 342.40 ms (0.9%) | 340.66 ms (0.9%) |
| `runner_and_plan_ms` | 226.12 ms (3.1%) | 179.96 ms (0.5%) | 183.75 ms (0.5%) |
| `verifier_init_ms` | 178.75 ms (2.5%) | 94.15 ms (0.2%) | 96.42 ms (0.2%) |
| `native_head_load_ms` | 148.70 ms (2.1%) | 93.42 ms (0.2%) | 96.16 ms (0.2%) |
| `expert_calibration_pre_ms` | 0.01 ms | 55.66 ms | 54.48 ms |
| `expert_source_open_ms` | 17.25 ms | 35.12 ms | 37.90 ms |
| `model_open_parse_ms` | 17.24 ms | 26.42 ms | 28.43 ms |
| `ple_bind_ms` | 17.82 ms | 19.37 ms | 19.90 ms |
| `kernel_provider_resolve_ms` | 15.60 ms | 15.96 ms | 15.88 ms |
| `cuda_init_ms` | 5.02 ms | 3.56 ms | 3.74 ms |
| `quant_layout_prep_ms` | 3.75 ms | 0.42 ms | 0.43 ms |
| **Container file opens** | **6** | **22** | **25** |
| **Text index parses** | **4** | **4** | **4** |
| **Dense `cudaMalloc` calls** | **262** | **305** | **305** |
| **Expert copy calls** | **30,720** | **73,728** | **73,728** |
| **Peak host RSS (`ru_maxrss`)** | **40.49 GiB** | **121.02 GiB** | **122.41 GiB** |

---

## 2. First-Inference, Prefill, and Decode Profile (`RuntimeTelemetry`)

Measured with `ornith_prefill_bench` (`prompt=512`, `generate=128`, `batch=64` for Ornith; `prompt=512`, `generate=64`, `batch=512` for Qwen3.8):

### Ornith-1.5-35B (Phase 3 Baseline)
- **First-token CUDA graph capture overhead**: `200.28 ms` (`first_window_capture_ms`) on the first verification step.
- **Prefill (512 tokens, batch 64)**: `2,078.79 ms` (`246.30 tok/s`)
  - CPU routed experts: `1,303.75 ms` (62.7% of prefill time; 8,370 expert group evaluations moving 61.96 GiB of expert weights)
  - CUDA attention / GDN recurrent state: `434.48 ms` (20.9%)
  - CUDA dense projections (`mmvq`): `271.56 ms` (13.1%)
- **Decode (128 tokens)**: `2,680.17 ms` (`47.76 tok/s`, median `20.90 ms/tok`, p95 `21.25 ms/tok`)
  - CPU routed experts (`decode_cpu_gu_ms` + `decode_cpu_down_ms`): `1,659.72 ms` (`12.97 ms/tok`, 61.9% of decode time)
  - Verifier GPU window launch (`decode_gpu_window_ms`): `1,004.73 ms` (`7.85 ms/tok`, 37.5% of decode time)

### Qwen3.8-Flash-Next (Phase 3 Baseline)
- **First-token CUDA graph capture overhead**: `158.18 ms` (verifier 1-token window) + `351.40 ms` (3-token window when MTP `spec=2` is active).
- **Prefill (512 tokens)**: `3,664.42 ms` (`139.72 tok/s`)
- **Decode (64 tokens, default `--pcie-frac -1` auto)**: `13.21 tok/s` (degraded by unthrottled 55.43 GiB `cudaHostAllocMapped` page-locking on a single NUMA socket + PCIe Gen3 x16 contention).

---

## 3. Ranked Optimization Opportunities

Based on the measured bottlenecks above, Phase 4 opportunities are ranked by measured wall-clock and memory impact:

1. **Eliminate Unbounded `cudaHostAllocMapped` Pinning & Multi-Shard Page Cache Duplication (`numa_expert_pin_ms`: 5.23s on Ornith, 33.74s on Qwen3.8; Peak RSS: 40.5 GiB / 121.0 GiB)**
   - *Root Cause 1*: `GuildModel::load` passed `gpu_needs_pinned = true` whenever `expert_performance.gpu_layers` was non-empty, even when `pcie_expert_fraction == 0` or when `budget > 24 GiB` in auto mode. On Qwen3.8 (55.43 GiB expert complement), `cudaHostAllocMapped` took ~26 seconds to pin 14.5 million 4 KiB pages in the NVIDIA driver, disabled `MPOL_INTERLEAVE` across the two Xeon NUMA sockets, and slowed `calibrate_experts` post-pinning to 1.50 seconds.
   - *Root Cause 2*: `pin_cache_complement` copied 30,720 (Ornith) to 73,728 (Qwen3.8) disjoint `gate`/`up`/`down` slices across 8 threads and only called `madvise(..., MADV_DONTNEED)` when `role_ptr_.empty()` (`experts.bin`), leaving 18–55 GiB of GGUF shard pages faulted in the OS page cache alongside the resident arena (doubling host RSS to 40.5 GiB on Ornith and 121.0 GiB on Qwen3.8).
   - *Root Cause 3*: `calibrate_experts` ran twice during startup even when the pre-pinning calibration already measured CPU/GPU latencies and `gpu_needs_pinned` was false.
   - *Phase 4 Fix*: Gate `gpu_needs_pinned` on active PCIe offload (`force_pin || pcie_expert_fraction > 0 || (pcie_expert_fraction < 0 && budget <= 24 GiB)`), release source GGUF shard pages per layer via page-aligned `MADV_DONTNEED`, scale copy workers up to 16 threads, enable 2 MiB transparent hugepages (`MADV_HUGEPAGE`) automatically when expert arenas are >= 32 GiB (where every expert blob > 2 MiB still stripes across both NUMA nodes), and support zero-copy contiguous `ExpertBlobs` in GPACK (`GUILD_EXPERT_MMAP=1`).

2. **Consolidate 6–25 Container Opens, 4 Text Index Parses, and 262–305 Device Allocations into GPACK v1 (`native_dense_load_ms`, `dense_index_and_pack_ms`, `embed_load_ms`, `native_head_load_ms`, `model_open_parse_ms`: ~800 ms on Ornith, ~2,200 ms on Qwen3.8)**
   - *Root Cause*: Loading from GGUF + pack directory opens the GGUF shards 5 separate times (`GgufModel`, `NativeDense`, `NativeHead`, `NativeEmbed`, `FileExpertSource`, `PleTable`), parses `index.txt` 3 times and `native_experts.txt` twice with `sscanf`, and calls `cudaMalloc` + `cudaMemcpy` 262–305 individual times in `NativeDense::load`.
   - *Phase 4 Fix*: Introduce GPACK v1 (`GPCK`/`GPKE`), a single 4096-byte page-aligned mmap container with pre-parsed binary `TensorIndexBin`, `DenseIndexBin` (with precomputed skip flags and compacted pool size), `NativeDenseIndexBin` + contiguous `NativeDenseBlob` (loaded with a **single** `cudaMalloc` arena and a **single** `cudaMemcpyAsync`), pre-validated `EmbedBlob`/`HeadBlob`/`PleBlob`, and `ExpertLayoutBin` + optional contiguous `ExpertBlobs`.

3. **Precompute Binary Tokenizer (`GBT1`) & Integer-ID BPE Merge Table (`tokenizer_load_ms`: 346 ms on Ornith, 342 ms on Qwen3.8)**
   - *Root Cause*: Every model load parses 11 MiB of `vocab.json` (248,320 UTF-8 keys), `merges.txt` (247,587 string pairs), and `token_type.json` (248,320 integers), building a `std::unordered_map<std::pair<std::string, std::string>, int32_t>` with hundreds of thousands of heap allocations.
   - *Phase 4 Fix*: Serialize the resolved vocabulary offsets, token types, chat template, and pre-resolved `(left_id, right_id) -> (rank, merged_id)` integer BPE merge table into the portable `GBT1` binary section inside GPACK, reducing 248k-vocab tokenizer load time from `~355 ms` to `~69 ms` (**5.1x faster**) and accelerating BPE encoding via integer token IDs.

4. **Pre-Capture Common 1-Token Verifier CUDA Graph at Startup (`verifier_init_ms` / `first_window_capture_ms`: ~200 ms on first inference)**
   - *Root Cause*: `Verifier::init` previously captured both 1-token and 2-token windows unconditionally during `init` (`178.75 ms` on Ornith) or deferred capture to the first token.
   - *Phase 4 Fix*: Make startup graph pre-capture configurable (`GUILD_PRECAPTURE_GRAPHS=1` pre-captures the 1-token decode graph during model load so the first inference token has 0 ms graph-capture jitter; default lazy capture saves `178 ms` during `Verifier::init`).

5. **Hardware-Specific Expert Row Interleaving & Kernel Autotune Locking (GEXEC Layer, Phase 5 Preparation)**
   - *Root Cause*: CPU expert decoding (`62%` of prefill and decode time) reads `gate_exps` and `up_exps` from two separate row blocks (`gate_off` and `up_off`) for the same input activation vector, doubling hardware prefetcher streams per worker thread.
   - *Phase 4 Fix*: Keep GPACK strictly hardware-independent (preserving original GGUF block layouts for GPU `xcache` compatibility) and implement `pack_expert_gate_up_interleaved`, `GExecAutotuneTable`, and `GExecHeterogeneousPlan` in the GEXEC contract layer (`include/guild/gexec/gexec_contract.hpp`).
