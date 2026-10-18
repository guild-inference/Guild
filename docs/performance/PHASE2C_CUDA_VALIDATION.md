# Phase 2C CUDA backend performance and distribution validation

Measured on Ubuntu 24.04, dual Intel Xeon Gold 6130 (32 physical cores, 64 threads;
AVX-512F/BW/DQ/VL, no VNNI), 377 GiB RAM, NVIDIA GeForce GTX 1070 8 GiB (SM 6.1),
driver 580.178.04, CUDA 12.0.140 (`-DGUILD_EXPERIMENTAL_SM60=ON`,
`-DCMAKE_CUDA_ARCHITECTURES=61`, `-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12`).
Model: Ornith-1.5-35B Q4_K_M, context 32,768, prefill batch 256, 128 generated
tokens, host-only FP16 KV cache, 512 MiB VRAM reserve, 18.16 GiB pageable resident
expert cache complement (`mlock`, NUMA interleaved across 2 nodes), zero runtime
file reads (`file_reads = 0`).

## Binary and package provenance

- **Preserved Phase 1 reference (`preserved`)**:
  `bench/results/external-kernels/phase1/ornith_prefill_bench`
  (`71dca3597ba64f135517e0d32f5dd585171d4913b18a1919a537f926c2decc7c`).
- **Rebuilt Phase 2C Core (`builtin`)**:
  `build-cuda12/ornith_prefill_bench`
  (`1b171687b46296bc5db48b4004ab8e923683d1d61e302ef21ca911b2be7bc244`),
  `build-cuda12/guild`
  (`7b002121227f7f8fae6d83bae97c7c5dd5ba14abfef042b4307cdfd9bc5d1cfa`),
  compiling pinned `guild-kernels-cpu` `264227c74d0151b914d826b0fa35a0f576a117b8` (`1.1.1`)
  and pinned `guild-kernels-cuda` `b8b675b5fd7605366c04facd7ced079de1ac6f69` (`1.3.0`,
  `dense_bundled: true`, `attn_bundled: true`, `moe_bundled: true`).
- **External `.gkernel` overrides (`all` and `cuda`)**:
  - `org.guild.kernels.cpu@1.1.1` (`libguild_cpu_plugin.so` `abe857c2e85bcc20cc10425c57f4c8b6f20f328f28ec3603ccd5e790aa5dcaec`)
  - `org.guild.kernels.cuda@1.3.0` (`libguild_cuda_plugin.so` `89991dd06fdd1344ae087712f4d49b9b0bd1fa407a84a1cf2a7ef5cf05e1888d`, exporting `external.cuda.router.bf16.v1`, `external.cuda.dense.v1`, `external.cuda.attn.v1`, `external.cuda.moe.v1`)
  - `org.guild.runners.qwen@1.0.0` (`libguild_qwen_runner.so` `0ab424ef00e5eaf240a0d6800077029795b7df1e7a4cae613bf8488896e1f1ae`)

All raw CSVs, per-sample JSON receipts, before/after CPU governor/frequency and
GPU clock/temperature/power snapshots, and bootstrap summaries are stored in:

- `bench/results/phase2c-cuda-complete/`
- `bench/results/phase2c-cuda-interruption-check/`
- `bench/results/phase2c-cuda-isolation/`
- `bench/results/phase2c-cuda-codegen/`

## 10-round counterbalanced paired results

```sh
python3 tools/bench_paired_plugins.py --cases preserved builtin all \
  --prompts 512 4096 8192 16384 --rounds 10 \
  --output bench/results/phase2c-cuda-complete
```

Each sample runs in a fresh process with one full-shaped warmup followed by one
measured repetition. Odd rounds reverse case order, and consecutive forward/reverse
pairs rotate the starting case. Reported relative differences are the **median of
within-round ratios** with 10,000-draw paired bootstrap 95% intervals (`seed = 20261017`).

### Median throughput (prefill / decode, tok/s)

| Prompt | Preserved Phase 1 | Rebuilt built-in (`1.3.0`) | All external overrides (`all`) |
|---:|---:|---:|---:|
| 512 | 236.281 / 43.241 | 237.375 / 43.227 | 237.022 / 43.079 |
| 4096 | 214.472 / 33.030 | 214.094 / 32.978 | 214.493 / 32.859 |
| 8192 | 196.362 / 25.983 | 195.974 / 25.937 | 196.615 / 26.047 |
| 16384 | 171.727 / 17.960 | 171.026 / 17.975 | 171.274 / 17.924 |

### Within-round paired deltas (median % and 95% bootstrap interval)

| Prompt | Comparison | Prefill delta | Decode delta |
|---:|---|---|---|
| 512 | `builtin` vs `preserved` | +0.29% [-0.79%, +0.80%] | -0.08% [-0.48%, +0.23%] |
| 512 | `all` vs `preserved` | +0.16% [-0.30%, +0.53%] | -0.43% [-1.88%, +0.21%] |
| 512 | `all` vs `builtin` | -0.21% [-0.45%, +0.58%] | -0.48% [-1.27%, +0.08%] |
| 4096 | `builtin` vs `preserved` | -0.22% [-0.54%, +0.44%] | -0.24% [-0.58%, +0.06%] |
| 4096 | `all` vs `preserved` | +0.03% [-0.57%, +0.66%] | -0.30% [-0.88%, -0.06%] |
| 4096 | `all` vs `builtin` | +0.03% [-0.62%, +0.40%] | -0.15% [-0.71%, +0.15%] |
| 8192 | `builtin` vs `preserved` | -0.17% [-0.55%, +0.24%] | -0.01% [-0.49%, +0.15%] |
| 8192 | `all` vs `preserved` | +0.13% [-0.25%, +0.71%] | -0.03% [-0.58%, +0.55%] |
| 8192 | `all` vs `builtin` | +0.40% [+0.10%, +0.62%] | +0.42% [-0.39%, +0.67%] |
| 16384 | `builtin` vs `preserved` | -0.60% [-1.40%, -0.16%] | +0.11% [-0.43%, +0.51%] |
| 16384 | `all` vs `preserved` | -0.30% [-1.03%, +0.07%] | -0.09% [-0.35%, +0.23%] |
| 16384 | `all` vs `builtin` | +0.21% [-0.08%, +0.58%] | -0.02% [-0.65%, +0.33%] |

## Follow-up attributions and sensitivity checks

### 1. 4096-token CUDA-only plugin isolation

In the primary table at 4,096 tokens, `all` vs `preserved` decode was -0.30%
[-0.88%, -0.06%], while `all` vs `builtin` decode was -0.15% [-0.71%, +0.15%].
To isolate the external CUDA plugin (`org.guild.kernels.cuda@1.3.0`) from the CPU
and runner packages, a separate 10-round counterbalanced comparison ran `builtin`
against `cuda` at 4,096 tokens (`bench/results/phase2c-cuda-isolation/`):

- `builtin` median: `214.790 / 32.960` tok/s
- `cuda` median: `215.327 / 32.964` tok/s
- `cuda` vs `builtin` prefill: `+0.41% [-0.21%, +0.95%]`
- `cuda` vs `builtin` decode: `-0.01% [-0.51%, +0.39%]`

Loading `org.guild.kernels.cuda@1.3.0` as an external `.gkernel` DSO shows no
measurable prefill or decode penalty relative to bundled compilation.

### 2. 16384-token round-3 session-pause sensitivity check

During the 16,384-token run, round 3 (`r02`) straddled an outer session pause
between `builtin`/`all` and `preserved` (within-round start spread 35,589.8 s vs
423.4–463.1 s for the other nine rounds), after which the first resumed
`preserved` sample ran on an idle, cooled machine (`179.098` tok/s vs `171.727`
median). All original receipts are retained in `bench/results/phase2c-cuda-complete/`,
and a complete uninterrupted replacement round (`builtin`, `all`, `preserved`) was
recorded in `bench/results/phase2c-cuda-interruption-check/` (`171.549 / 17.950`,
`171.490 / 17.877`, `171.837 / 17.884` tok/s). Substituting that uninterrupted
round yields:

- `builtin` vs `preserved` prefill: `-0.50% [-0.75%, -0.08%]`, decode: `+0.11% [-0.43%, +0.41%]`
- `all` vs `preserved` prefill: `-0.24% [-0.58%, +0.07%]`, decode: `-0.09% [-0.36%, +0.22%]`
- `all` vs `builtin` prefill: `+0.21% [-0.08%, +0.58%]`, decode: `-0.02% [-0.45%, +0.33%]`

Keep the remaining ~0.2–0.6% 16K `builtin` vs `preserved` prefill difference
documented as an open observation rather than claiming exact prefill equality.

### 3. Resource footprint and CPU codegen inspection

- Across all 40 rounds, `vram_sample_bytes` is constant at `2,145,583,104` bytes
  for `preserved` and `builtin`, and `2,156,068,864` bytes (+10.0 MiB for the
  loaded external CUDA DSO module image) for `all`. `host_kv_allocated_bytes`
  (`671,098,880`), `kv_staging_bytes` (`67,125,248`), and `prefill_scratch_bytes`
  (`212,471,812`) are byte-identical across all three cases.
- `tools/audit_binary_codegen.py` (`bench/results/phase2c-cuda-codegen/codegen.json`)
  confirms identical instruction counts between `phase1` and `phase2c` for all
  inspected SIMD and native expert kernels: `kq256_rows` (284), `iq512_gu_rows`
  (2748), `kq256_gu_rows` (113), `native_gu_rows` (338), `native_down_rows` (220),
  `native_quant_act` (17), and `native_quant_h` (17). `ExpertPool::drain` is 1,462
  instructions vs 1,290 in `phase1` due to the CPE1/CPU-backend callback dispatch
  paths introduced in Phase 2B.

## Packaged Linux CUDA distribution verification

`tools/package_linux_cuda.py` staged `build-cuda12` into
`build-cuda12/phase2c-distribution/guild-linux-x86_64-cuda12` and produced
`guild-linux-x86_64-cuda12.tar.gz`
(`88cc9183723eb728603fd9d1300422695ae5ea0cc357c919e89d2d85ca84df8b`), bundling
`libcudart.so.12`, `libcublas.so.12`, `libcublasLt.so.12`, and license files
(`Guild-Strata-MIT.txt`, `CPU-MIT.txt`, `CUDA-Backend-MIT.txt`, `GGML-MIT.txt`,
and CUDA toolkit EULA/copyright).

`tools/test_distribution_inference.py` launched the packaged `guild` wrapper on
`127.0.0.1` against `build-workflow-tools/distribution-fixture.json` (`ornith:35b`,
context 32,768, prefill batch 256, 16 greedy tokens) and verified an exact match
to the maintained greedy completion golden.
