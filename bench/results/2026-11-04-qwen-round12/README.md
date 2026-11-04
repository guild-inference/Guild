# Qwen3.8-Flash-Next, twelfth round: the prompt's dense products through MMQ, and where decode stands

PC: 2x Xeon Gold 6130 (16 cores each, AVX-512 BW without VNNI, 2 NUMA nodes, 376 GiB RAM), GTX 1070 8 GB (sm_61,
15 SMs, 151 W cap, PCIe 3.0 x16), CUDA 12. Branch `feat/nemotron-transcribe`, guild-kernels-cpu `9b7281a` and
guild-kernels-cuda `de65d60` (the pinned revisions). Unsloth UD-IQ4_XS with setup's settings: the DFlash 2 drafter
(Q4_K weights, the multilingual head subset, `--spec 8`), `--dense q5_k --hc q8_0`, `--prefill auto`;
`tools/bench_runtime.py`, `--context 32768`, FP16 KV in host memory, `--resident-budget-gib 56`.

## The prompt: where 19.6 s went

The 4,007-token prompt (`qwen4k.json`, no prefix cache) with `GUILD_PREFILL_TIMING=1`. The load leaves 2,954 MiB
free, so the chunk is 4,096 tokens, computed as chunks of 512 (round 11):

| Phase | ms | Share |
| --- | ---: | ---: |
| GDN input projections (qkv, gate, alpha, beta) | 3,659 | 18.7% |
| expert gate/up products (MMQ) | 3,146 | 16.1% |
| waiting for expert copies | 2,856 | 14.6% |
| hyper-connection read | 1,904 | 9.7% |
| QSA projections | 1,551 | 7.9% |
| QSA attention | 1,532 | 7.8% |
| expert down products (MMQ) | 1,321 | 6.8% |
| GDN output projection | 1,140 | 5.8% |
| the rest | 2,499 | 12.8% |

**Dense products.** On Pascal the prompt path multiplies a quantized dense matrix as an FP16 copy through cuBLAS,
with FP32 arithmetic.
- The GDN input projections are 12.1 TFLOP for this prompt, done in 3.66 s: about 3.3 TOPS.
- The card's int8 dot products peak near 26 TOPS, and the experts already use them (llama.cpp's MMQ).

**Expert copies.** nsys (`pf.sqlite`, not kept): the prompt moves 50 GB of experts in 2.4 MB copies, one per expert,
at 7.6 GB/s.
- Pinned copies on this PC peak at 8.0 GB/s from node 0 and 7.2 GB/s from node 1, at any copy size (`bw.cu`).
- The copy engine is busy 40% of the prompt, but only during the MoE half of each layer: 85-90% there, with MMQ ~42%
  busy, and 0% during the dense half. With `--prefill-rows 512` the expert ring is the 8-slot ring of a 512-token
  chunk, so nothing streams ahead of the routing.
- Sizing the ring by the chunk instead gives the same tokens. But it gains little, and on this card it does not fit:

| Ring slots | Prompt | Waiting for copies | VRAM after |
| --- | ---: | ---: | ---: |
| 8 (today) | 19.66 s | 2,858 ms | 7,353 MiB |
| 32 | 19.75 | 2,989 | 7,449 |
| 64 | 19.67 | 2,646 | 7,577 |
| 128 | 19.24 | 2,140 | 7,833 |
| 160 | 19.01 | 1,918 | 7,961 |
| the byte budget's (~190) | out of memory | | |

  Not kept: 3% for the last 230 MiB of VRAM.

## `GUILD_PREFILL_DENSE_MMQ` (opt-in; it rounds differently)

`GUILD_PREFILL_DENSE_MMQ=1`: each dense matrix of a type MMQ covers (Q5_K here) multiplies its own GGUF blocks
against q8_1 activations, a chunk's rows in one launch, instead of an FP16 copy through cuBLAS. That is decode's MMVQ
rounding. `=2` adds the hyper-connection's Q8_0 matrices (`--hc q8_0`; rebuilt from Guild's planes as GGUF blocks,
the activations widened from BF16). The buffers are its own, grown on first use: ~130 MB at a 4,096-token chunk.

| Prompt | Today | Mode 1 | Mode 2 |
| --- | ---: | ---: | ---: |
| 4,007 tokens | 19.61 s | 16.08 s | 15.76 s |
| 31,918 tokens | 171.4 s | 142.5 s | |

Mode 1 at 4K: the GDN input projections take 3,659 -> 1,520 ms, the QSA projections 1,551 -> 763 and the GDN
output projection 1,140 -> 746. Mode 2 takes the hyper-connection read from 1,903 to 1,510 ms. Decode after the
prompt is unchanged (46-47 tok/s at 4K, 40-42 at 32K), but the text after the prompt changes from its fourth token.

**What it does to the predictions.** A new `qwen_kl` option, `QWEN_KL_PREFIX=1024 QWEN_KL_PREFILL=1`, reads a text's
first 1,024 tokens through the prompt path and teacher-forces the rest through the verifier. The reference is the
same text read entirely by the verifier, decode's own arithmetic. `GUILD_DENSE=q5_k GUILD_HC=q8_0`; the two KL texts
of round 5, positions 1,024 on (1,060 and 1,149 predictions):

| Prompt path | Text 1: top-1 same | KL mean | 99th pct | Text 2: top-1 same | KL mean | 99th pct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| today (cuBLAS) | 96.04% | 0.0118 | 0.144 | 96.95% | 0.0141 | 0.159 |
| mode 1 | 96.51% | 0.0138 | 0.146 | 96.26% | 0.0181 | 0.179 |
| mode 2 | 95.75% | 0.0160 | 0.144 | 95.56% | 0.0224 | 0.250 |

The perplexity of the text from position 1,024 is 2.5036 / 2.6170 with the verifier's prefix:
- today: 2.5179 / 2.6184
- mode 1: 2.4982 / 2.6437
- mode 2: 2.5230 / 2.6103

Mode 1 moves the predictions after a prompt by about as much again as today's prompt path already does. It stays
opt-in for that decision; mode 2 costs more for its last 2%.

## Decode

The bench's prose and code prompts, 2 runs x 256 tokens, alternated over 2 rounds:

| Arm | prose tok/s | code tok/s |
| --- | --- | --- |
| today | 45.64 / 45.51 | 60.46 / 60.18 |
| Q5_K products with per-column sums (kernel branch `perf/q5k-cols`, local) | 45.83 / 45.88 | 60.56 / 60.35 |
| `--dense q6_k` instead of `q5_k` | 42.94 / 42.99 | 56.39 / 56.27 |

- **Where a window goes.** nsys of decode (not kept): the GPU is busy ~89% of a window once nsys's own graph launch
  time is taken out.
  - Its bytes (the dense weights twice, once per token group, the HC, the head, the drafter) are ~7.5 GB, about 36 ms
    at 210 GB/s, against 58-65 ms measured.
  - The gap is the products at several columns. `glm5_kernels_test shapes`, Qwen's 2560 x 10240 qkv, us at 1 / 2 /
    3 / 4 columns:

    | Format | 1 | 2 | 3 | 4 |
    | --- | ---: | ---: | ---: | ---: |
    | Q5_K | 89.7 | 113.0 | 145.6 | 183.3 |
    | Q6_K | 104.4 | 117.1 | 128.1 | 145.8 |
    | Q8_0 | 133.4 | 135.7 | 138.5 | 160.7 |

    Past two columns Q5_K's arithmetic, not its bytes, sets the time. The verify head (Q6_K, 2560 x 248320) takes
    2.43 ms at one column and 7.02 at eight: 3.55 ms a window on average.
- **`perf/q5k-cols`.** The Q5_K dot takes its activation sums once per column (the staged records' spare int) and
  its scale products in 16 bits; staged kernels run up to 4 columns. The same ints and float operations, so the same
  bits: Guild's mmvq parity tests pass and the decode texts are identical. Two columns ran 5-12% faster in the
  shapes bench (its runs differ by ~8% on this power-capped card); decode +0.6% / +0.2%. Not pinned: too small to
  move the lock.
- **`--dense q6_k`.** It reads 20% more bytes and loses 5-6% here, though its products scale better with columns.
  Its texts differ (it is closer to the GGUF's weights: round 5's KL 0.028 against q5_k's 0.048).
- Nsight Compute cannot read the card's counters here (`ERR_NVGPUCTRPERM`, needs root), so the multi-column stall
  reasons are not measured.

## Files

`runs/`: the reports (`b-*.json`, every run's ids and timings), the KL outputs (`k?1.out`, `k?2.out`), the scripts
(`*.sh`; `ph.sh`'s run used mode 2 under its earlier name, `=1` before the modes were split) and `bw.cu` (the
pinned copy test).
