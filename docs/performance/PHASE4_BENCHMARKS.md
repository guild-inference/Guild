# Phase 4 Benchmarks: GPACK Startup, Memory, and Inference Parity

All measurements below were collected on the same dual-socket Intel Xeon E5-2697A v4 (32C/64T, 251 GiB DDR4 across 2 NUMA nodes) + NVIDIA GeForce GTX 1070 (8 GiB GDDR5, SM 6.1) + NVMe SSD (`/mnt/models-ssd`) machine as the Phase 3 baseline (`docs/performance/PHASE4_PROFILING.md`).

Raw telemetry JSON profiles are stored in `build-cuda12/phase4-profile/`.

---

## 1. Ornith-1.5-35B (`ornith:35b`, 40 layers, 256 experts/layer, 8 active)

Workload: `ornith_prefill_bench --model ornith:35b --prompt 512 --generate 128 --runs 3 --batch 64`

| Metric | Phase 3 Baseline (GGUF + pack) | Phase 4 GPACK (Embedded, Default 4K NUMA) | Phase 4 GPACK (External Experts) | Phase 4 GPACK (`GUILD_EXPERT_THP=1`) | Phase 4 GPACK (`GUILD_EXPERT_MMAP=1` Zero-Copy) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **Total Model Load (`load_ms`)** | **7,252.08 ms** | **5,396.36 ms (1.34x)** | **2,953.97 ms (2.46x)** | **2,344.23 ms (3.09x)** | **999.94 ms (7.25x)** |
| `model_open_parse_ms` | 17.24 ms | **1.00 ms (17.2x)** | **0.80 ms (21.6x)** | **1.02 ms (16.9x)** | **1.01 ms (17.1x)** |
| `dense_index_and_pack_ms` | 56.62 ms | **30.67 ms (1.85x)** | **30.92 ms (1.83x)** | **30.62 ms (1.85x)** | **31.65 ms (1.79x)** |
| `native_dense_load_ms` | 312.52 ms | **275.46 ms (1.13x)** | **273.84 ms (1.14x)** | **276.00 ms (1.13x)** | **275.59 ms (1.13x)** |
| `numa_expert_pin_ms` | 5,232.29 ms | **3,766.52 ms (1.39x)** | **1,329.36 ms (3.94x)** | **697.81 ms (7.50x)** | **0.00 ms (zero-copy)** |
| `verifier_init_ms` | 178.75 ms | **0.78 ms (229x)** | **0.75 ms (238x)** | **0.76 ms (235x)** | **0.76 ms (235x)** |
| `tokenizer_load_ms` (`GBT1`) | 346.18 ms | **80.70 ms (4.29x)** | **83.24 ms (4.16x)** | **81.53 ms (4.25x)** | **81.77 ms (4.23x)** |
| **Container File Opens** | 6 | **1** | **2** | **1** | **1** |
| **Text Index Parses** | 4 | **0** | **0** | **0** | **0** |
| **Dense `cudaMalloc` Calls** | 262 | **3 (87.3x fewer)** | **3 (87.3x fewer)** | **3 (87.3x fewer)** | **3 (87.3x fewer)** |
| **Peak Host RSS (`ru_maxrss`)** | 40.49 GiB | **22.92 GiB (-17.57 GiB)** | **22.91 GiB (-17.58 GiB)** | **22.93 GiB (-17.56 GiB)** | **21.25 GiB (-19.24 GiB)** |
| **Prefill Throughput (p512)** | 246.30 tok/s | **245.55 tok/s** | **247.37 tok/s** | 222.84 tok/s | 164.35 tok/s |
| **Decode Throughput (g128)** | 47.76 tok/s | **47.90 tok/s** | **47.82 tok/s** | 43.25 tok/s | 28.59 tok/s |
| **Bitwise Token Parity (128 tok)** | Reference | **100% Identical** | **100% Identical** | **100% Identical** | **100% Identical** |

*Note on NUMA page size for Ornith-1.5-35B*: Because Ornith's expert blobs are `1.69–1.95 MiB` (`< 2 MiB`), 4 KiB page interleaving (`MPOL_INTERLEAVE`, the default for arenas `< 32 GiB`) stripes every single expert blob across both Xeon memory controllers (`47.90 tok/s` decode), whereas 2 MiB hugepages (`GUILD_EXPERT_THP=1`) place an entire `< 2 MiB` expert blob onto a single socket (`2.34 s` load, `43.25 tok/s` decode).

---

## 2. Qwen3.8-Flash-Next (`qwen3.8-flash-next`, 48 layers, 512 experts/layer, 10 active, PLE 28 GiB)

Workload: `ornith_prefill_bench --model qwen3.8-flash-next --batch 512 --prompt 512 --generate 64 --runs 2 --context 8192`

| Metric | Phase 3 Baseline (3-shard GGUF + pack) | Phase 4 GGUF + Pack (Auto Pinned Fix + THP) | Phase 4 GPACK v1 (`qwen3.8-flash-next-ext.gpack`) | Phase 3 Baseline + MTP (`spec=2`) | Phase 4 GPACK v1 + MTP (`spec=2`) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **Total Model Load (`load_ms`)** | **38,967.12 ms** | **10,606.87 ms (3.67x)** | **9,610.80 ms (4.05x)** | **39,884.37 ms** | **9,259.65 ms (4.31x)** |
| `model_open_parse_ms` | 26.42 ms | 29.87 ms | **1.40 ms (18.9x)** | 28.43 ms | **1.54 ms (18.5x)** |
| `dense_index_and_pack_ms` | 502.06 ms | 501.36 ms | **469.51 ms (1.07x)** | 557.28 ms | **470.51 ms (1.18x)** |
| `native_dense_load_ms` | 1,011.15 ms | 1,019.01 ms | **969.92 ms (1.04x)** | 1,039.88 ms | **973.17 ms (1.07x)** |
| `numa_expert_pin_ms` | 33,738.47 ms | 7,403.52 ms | **6,803.16 ms (4.96x)** | 33,962.01 ms | **5,952.86 ms (5.71x)** |
| `expert_calibration_post_ms` | 1,500.83 ms | **0.00 ms (eliminated)** | **0.00 ms (eliminated)** | 1,546.83 ms | **0.00 ms (eliminated)** |
| `verifier_init_ms` | 94.15 ms | **1.14 ms (82.6x)** | **1.14 ms (82.6x)** | 96.42 ms | **1.17 ms (82.4x)** |
| `tokenizer_load_ms` (`GBT1`) | 342.40 ms | 350.85 ms | **71.36 ms (4.80x)** | 340.66 ms | **71.40 ms (4.77x)** |
| **Container File Opens** | 22 | 22 | **3 (7.3x fewer)** | 25 | **3 (8.3x fewer)** |
| **Text Index Parses** | 4 | 4 | **0** | 4 | **0** |
| **Dense `cudaMalloc` Calls** | 305 | 305 | **3 (101.7x fewer)** | 305 | **3 (101.7x fewer)** |
| **Peak Host RSS (`ru_maxrss`)** | 121.02 GiB | **64.75 GiB (-56.27 GiB)** | **61.18 GiB (-59.84 GiB)** | 122.41 GiB | **61.71 GiB (-60.70 GiB)** |
| **Prefill Throughput (p512)** | 139.72 tok/s | **144.49 tok/s** | **143.47 tok/s** | 136.50 tok/s | **138.30 tok/s** |
| **Decode Throughput (g64)** | 13.21 tok/s | **25.99 tok/s (1.97x)** | **26.06 tok/s (1.97x)** | 12.80 tok/s | **28.59 tok/s (2.23x)** |
| **Bitwise Token Parity (64 tok)** | Reference | **100% Identical** | **100% Identical** | Reference | **100% Identical** |
| **Logit Parity (993,280 B `.f32`)** | Reference | **0 bytes diff (`cmp -l`)** | **0 bytes diff (`cmp -l`)** | Reference | **0 bytes diff (`cmp -l`)** |

---

## 3. Standalone Precomputation Microbenchmarks

- **`GBT1` Binary Tokenizer (`gpack_test`)**:
  - 248,320-token Qwen3.5/Ornith vocabulary + 247,587 BPE merges:
    - JSON/text load (`vocab.json` + `merges.txt` + `token_type.json`): `354.70 ms`
    - `GBT1` binary load (`6.76 MiB` section): **`68.97 ms` (5.14x faster)**
    - Token ID output across all 19 golden multilingual/code/special-token test cases and 8 Jinja chat template cases: **100% exact match**.
- **GEXEC Gate/Up Expert Row Interleaving (`gexec_contract_test`)**:
  - 1,204 KiB routed expert blob (`n_ff = 512`, `gate_row = 784 B`, `up_row = 784 B`, `down = 430,080 B`):
    - Interleave time: **`128.22 us` per expert blob** (~9.4 GB/s single-thread) with **100% lossless round-trip**.
