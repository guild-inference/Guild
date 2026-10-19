# Phase 3 runtime module performance and parity

Validation host: dual Intel Xeon Gold 6130 (32 physical cores, 64 threads),
377 GiB RAM, GTX 1070 8 GiB (SM61), CUDA 12.0.140, driver 580.178.04.

Raw receipts, CSVs, hardware state and module provenance are in
[`bench/results/phase3-modules`](../../bench/results/phase3-modules) and were
collected with [`bench_phase3_modules.py`](../../tools/bench_phase3_modules.py).

## Binaries and model

| Role | Path | SHA-256 |
|---|---|---|
| Phase 2E baseline (`88c8117...`) | `build-cuda12/phase3-baseline/ornith_prefill_bench` | `444dcfc31c39c6f568952c8ba07f3f29523550ae5b898fc06a22b5e4211a218e` |
| Phase 3 candidate | `build-cuda12/ornith_prefill_bench` | `71cfe17157c8c21b7040ecb18d3531638505729520b674f0fcaaecbf0104211a` |
| Phase 3 CLI | `build-cuda12/guild` | `d13543478c627fa94c6c852d42286445ff60173df8c1573e425158c6d70df828` |

Model: Ornith-1.5-35B Q4\_K\_M GGUF + native pack, 32,768 context, batch 256,
128 generated tokens, resident expert cache complement (18.16 GiB RAM, 0 file
reads), full-shaped warmup before each measured run. Three counterbalanced rounds
compare:

- `preserved`: frozen Phase 2E binary.
- `builtin`: Phase 3 binary with an empty store.
- `module`: Phase 3 binary with the signed `example.scan@1.0.0` `.gkmodule`
  installed, enabled and loaded in process.

## Results (medians of 3 fresh processes)

| Prompt | Case | Prefill (tok/s) | Decode (tok/s) | Prefill vs preserved | Decode vs preserved | Sampled VRAM |
|---:|---|---:|---:|---:|---:|---:|
| 512 | preserved | 240.660 | 46.456 | — | — | 2,749,562,880 B |
| 512 | builtin | 240.981 | 46.767 | +0.03% | +0.67% | 2,749,562,880 B |
| 512 | module | 240.505 | 46.685 | +0.42% | +0.36% | 2,749,562,880 B |
| 4,096 | preserved | 223.600 | 44.840 | — | — | 2,749,562,880 B |
| 4,096 | builtin | 222.444 | 44.803 | -0.49% | -0.08% | 2,749,562,880 B |
| 4,096 | module | 222.192 | 44.148 | -0.30% | -1.56% | 2,749,562,880 B |

Every sample matched first-token logits and all 128 greedy token IDs bit-for-bit:

- 512 tokens: `.f32` `aa87131639c6411fa7a1474d61d5630a8ec0f14e55bd4c974cb20d0ebff64a92`,
  `.ids` `8965d507bf4fc36593b6db0965da99c0357b42b9d13022b0933b934337a6754b`.
- 4,096 tokens: `.f32` `0c42ca2601e0f1eddbbeaa1d311bf1efcd1f080da8ca1e2e51ec7bd01a7e7b11`,
  `.ids` `a34517a7a42ef1c8a5671397da789000f063c8b2886ef74137ed2da88e9b71c0`.
