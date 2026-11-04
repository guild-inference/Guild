# GTX 1070 decode kernels: multi-column MMVQ, the hyper-connection read, the top-k router

Measured 2026-10-24 on this PC: Ubuntu, two Xeon Gold 6130 (2 NUMA nodes), 376 GiB RAM, GTX 1070 8 GiB (sm_61),
driver 580.178.04, CUDA 12 Release build of `guild-generate` at `a400927` (`-DGUILD_EXPERIMENTAL_SM60=ON`) with
`GUILD_CUDA_SOURCE_DIR` on a `guild-kernels-cuda` checkout: `fe4c080` (the locked revision) before, branch
`perf/sm61-decode` after (`a1416ed`, `3d94934`, `1d18a1b` on top of `fe4c080`). That branch is local, not pushed,
and `backends.lock.json` still pins `fe4c080`.

Every change keeps the outputs bit for bit: the same operations in the same order per value, only who does what
changes. Kernel times come from a benchmark that runs the old and new kernels interleaved in one process (the
GTX 1070's clocks moved the same untouched kernel by up to 6% between processes) and compares every output.
Decode rows are medians of three 256-token greedy generations from `tools/bench_runtime.py` on its prose and code
prompts, with the settings of [2026-10-24-numa-iq3s](../2026-10-24-numa-iq3s/README.md); every pair of runs gave
the same text.

## Decode

| Model | Before | After |
| --- | ---: | ---: |
| Qwen3.8-Flash-Next UD-IQ4_XS, MTP draft, `--spec 4` (prose / code) | 26.14 / 33.86 tok/s | 27.40 / 35.96 tok/s |
| Ornith-1.5-35B Q4_K_M, its MTP drafter (prose / code) | 63.67 / 73.69 tok/s | 66.88 / 77.78 tok/s |
| Ornith-1.5-35B Q4_K_M, no drafter, router change only (two runs) | 54.46, 54.25 / 54.86, 55.03 | 55.16, 54.73 / 55.58, 55.57 |

## 1. Multi-column MMVQ (`a1416ed`)

The exact multi-column kernel loaded every column's activation values again for each row of its block, inside
that row's bounds check, so blocks of two rows saved nothing. The formats with a split decode (Q8_0, Q6_K, Q4_K,
Q5_K, IQ4_XS, Q4_0, Q5_0, IQ4_NL) now load each column once per block step for all the block's rows: 4 rows for
Q8_0 up to 4 columns, 2 otherwise. One-row kernel to this, us per call:

| Shape | 2 columns | 3 columns | 4 columns |
| --- | ---: | ---: | ---: |
| Q8_0 2560 -> 10240 | 171 -> 154 | 203 -> 166 | 212 -> 170 |
| Q6_K 2560 -> 248320 (Qwen's output head) | 4297 -> 4143 | 5197 -> 4669 | 6137 -> 5190 |
| IQ4_XS 2560 -> 10240 | 123 -> 107 | 175 -> 114 | 213 -> 153 |
| Q5_K 2048 -> 4096 | 48 -> 43 | 62 -> 50 | 67 -> 53 |

Four rows lost against two on every format but Q8_0, and on Q8_0 beyond four columns (register pressure). One
column keeps its own kernels: this path does not touch it.

## 2. The hyper-connection read (`3d94934`)

The staged down projection stages two 1280-float tiles per token: 40 KB of shared memory at four tokens, so a
Pascal SM (96 KB) held two of the 41 blocks, and five or more tokens took two launches, each reading all 6.5 MB of
`w_down`. Before sm_70 it now runs 512-float tiles (eight tokens fit 32 KB in one launch), and the up projection
runs 8 columns per block (320 blocks instead of 160, which left a third of a second wave on 15 SMs) with `lo` in
shared memory as two planes, so its float4 reads hit no bank twice. `fused_gr_check` compares it with the plain
read on the card at start (it passes here); `gr_parity` passes. Other cards keep the old kernels: they were not
measured.

| Tokens | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| before, us | 85 | 96 | 111 | 151 | 204 | 224 | 240 | 282 |
| after, us | 81 | 90 | 104 | 115 | 132 | 146 | 157 | 174 |

The Qwen gain is small (+1.3% decode on its own) because the split verify window gives the read two or three
tokens at a time. With these kernels one group (`GUILD_SPEC_SPLIT=0`) was 27.33 / 36.32 against 27.17 / 36.15
tok/s with the split, inside the noise, so the split stays on.

## 3. The top-k router on CUDA (`1d18a1b`)

`router_top10_fast_kernel` (bitwise the block kernel, see its comment) was compiled for HIP only; it is now the
CUDA default too, `GUILD_ROUTER_OLD=1` keeps the block kernel. The HIP test's rows ported to the CUDA runtime
(65,536 rows of 512 experts / k 10 and 8,192 of 256 / k 8, with ties, near-ties, NaN rows and the forced serial
path) all match here. A single-token call took 14.3 instead of 33.5 us (512 / k 10) and 9.4 instead of 19.6 us
(256 / k 8). Ornith uses this router; Qwen's (`native_router_top10`) is another kernel: 7.9 us per call on its
own, 18.7 us in the decode profile, where it shares the GPU with the mapped-memory copies.

## Tried and not kept

All bitwise equal; none was faster where it counts.

- One-column calls through the multi-row kernel (1, 2, 4, 8 rows): the Q6_K heads 5-15% slower, Q8_0 even,
  IQ4_XS 10% faster (no hot IQ4_XS dense matrix in either model).
- One row per warp at one column, each lane doing the work of the four warps' threads and summing them in the
  shared-memory order: Q4_0 / Q5_0 8-11% and Ornith's 2048-wide Q4_K 7% faster, the Q6_K heads 23% slower.
- `__launch_bounds__` minimum blocks 12 or 16 on the one-column K-quant kernels (37-39 -> 32 registers): within
  the +-3% noise, IQ4_XS and Q5_K slower.
- Activations staged in shared memory for the multi-column kernel: slower at four columns (every block copies all
  columns from L2).
- The norm folded into the down kernel: each of the 41 blocks would re-read R for every token before its first
  dot; at most about 0.5% of a window.
- Q6_K blocks padded from 210 to 212 bytes so the one-column kernel loads 32-bit words instead of two 16-bit
  halves (`q6a.cu`, a standalone kernel): 2731 -> 2560 us for Ornith's head, 3334 -> 3078 us for Qwen's, bitwise
  equal. Worth about 1% of Ornith's decode; it needs a padded weight format in the upload, not done.

## Where the GPU time goes now

nsys, 200 tokens. Qwen with MTP (8.0 s of GPU time with the prompt; section 1 in, section 2 not yet): the Q8_0
projections about 37%, the hyper-connection down and up kernels 11% and 10%, waiting on CPU experts 4.8%. Ornith without
drafting (3.99 s GPU for a 3.98 s run): waiting on CPU experts 24%, the one-column Q6_K and Q4_K kernels about 41%
(the Q6_K head alone 2.69 ms per token, 158 GB/s against a 222 GB/s read ceiling), the router 5.9% (before
section 3).
