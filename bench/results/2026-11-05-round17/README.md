# Seventeenth round: Drama 3's tower in fewer bits, measured and not taken; a slow GLM run that did not repeat

PC: 2x Xeon Gold 6130 (16 cores each, AVX-512 BW without VNNI, 2 NUMA nodes, 376 GiB RAM), GTX 1070 8 GB (sm_61,
15 SMs, 151 W cap, PCIe 3.0 x16), CUDA 12. Branch `perf/next`, the pinned kernel revisions.

## Drama 3's tower: K-quants against Q8_0

Round 16 found the tower the largest read on the GPU: 13 layers and 10 heads, all Q8_0, about 1.1 GB a frame. New,
opt-in, and off unless set:
- **`GUILD_DRAMA3_TOWER=q6_k|q5_k|q4_k`** (or `kind=type,...` over the kinds `eh_proj`, `wqkv`, `wo`, `w1`, `w2`,
  `w3` and `head`) re-quantizes the tower's matrices at load. The re-quantization is `q8_0_requant`, now shared in
  `guild/core/dense_quant.hpp`; Qwen's `--dense` uses the same function, and its tokens are unchanged.
- **Two test hooks.** `GUILD_DRAMA3_DUMP_FRAMES` writes each decoded frame in the form `GUILD_DRAMA3_FORCE` reads.
  `GUILD_DRAMA3_DUMP_LOGITS` writes each step's tower logits.

The Q8_0 run with both dumps on gives the same WAV as without them (`6e04c9b1`).

**Loss against fish-speech's own float32 tower.** `drama3_tower_test` feeds the tower the reference's hidden states
and teacher-forces its 40 frames (the reference of 2026-10-24); KL of the 10 heads' distributions, 400 rows:

| Tower | VRAM of its matrices | KL mean / worst, nats | Argmax agreement | Worst layer cosine |
| --- | ---: | --- | ---: | ---: |
| Q8_0 (the default) | 1,420 MiB | 0.0015 / 0.0066 | 94.0% | 0.99950 |
| Q6_K | 1,181 | 0.0088 / 0.044 | 84.5% | 0.99788 |
| Q5_K | 1,049 | 0.026 / 0.10 | 78.0% | 0.99030 |
| Q4_K | 925 | 0.097 / 0.65 | 58.5% | 0.97248 |

Teacher-forced on a whole Guild run's 480 frames, against the Q8_0 tower (`runs/kl.py`):

| Tower | Main-token head KL | Code heads' KL, mean |
| --- | ---: | ---: |
| Q6_K | 0.0003 | 0.017 |
| Q5_K | 0.0010 | 0.058 |
| Q4_K | 0.0039 | 0.21 |

The code heads' KL grows from code 1 to code 9 in every format (`runs/tow.progress`).

**Speed.** The bench text, seeds 11 and 12, 2 rounds alternated, frames/s:

| Tower | Round 1 | Round 2 | Mean | Change |
| --- | --- | --- | ---: | ---: |
| Q8_0 | 46.22 / 46.06 | 46.21 / 46.08 | 46.1 | |
| Q6_K | 47.58 / 47.51 | 48.94 / 48.74 | 48.2 | +4.5% |
| Q5_K | 49.45 / 49.68 | 50.60 / 50.71 | 50.1 | +8.7% |
| Q4_K | 52.65 / 52.50 | 50.83 / 52.55 | 52.1 | +13% |

Every format's audio differs, since its codes differ. Whisper-small transcribes all eight round-1 WAVs as the whole
text (`runs/round1.txt`); its word error rates of 3.8-7.5% are only how it writes "lighthouse keeper" and
"ninety-seven". Q4_K's speech is the longest: 24.6 and 25.6 s against Q8_0's 22.2 and 19.8.

**Not taken.** The cheapest step, Q6_K, has 6 times Q8_0's loss against the reference for 4.5% more frames, so Q8_0
stays. The option stays for anyone who would trade audio for speed; a listening test, not these numbers, would have
to decide that.

## A GLM run at 13.9 tok/s that did not repeat

The first GLM-5.3-Flash run of the round decoded at 13.93 tok/s. The step median was 101 ms and the 90th percentile
236 ms, against 79 ms in every other run. Its CPU experts read their rows at 92.8 GB/s.

It ran after this round's page-cache fills (round 16), and its load read the GGUF from disk in 582 s. The same
binary, run twice more, was normal:

| Run | Load | Step 10th / 50th / 90th | CPU experts a step, median | Expert rows | tok/s |
| --- | ---: | --- | ---: | ---: | ---: |
| first | 582 s, from disk | 78.9 / 101.2 / 236.2 ms | 85.4 ms | 92.8 GB/s | 13.93 |
| again, files cached | 35 s copy | 78.5 / 79.0 / 80.2 | 64.0 | 184.2 | 23.25 |
| files evicted, 163 GB of other models read first | 1,150 s, from disk | 78.6 / 79.1 / 80.7 | 64.3 | 183.0 | 23.13 |

- **The third run** repeated the first's conditions as far as they are known (`runs/repro.sh`). Its process never
  swapped (`VmSwap` 0), and it was normal.
- **What is left.** Nothing in this round's changes is on GLM's path. The most likely cause is something else on
  the machine during the first run, which these samples cannot show.

GLM's `cpu_moe` arenas are bound per node but not `mlock`ed, unlike the other models' resident experts. Locking them
was not tried, since nothing here shows it would help.

## Files

`runs/`:
- Drama 3: `tow.sh`, `kl.py`, `tow.progress`, `tower_test_*.txt`, `wer.py`, `text.txt`, `round1.txt`.
- GLM: `repro.sh`, `swp.sh`, `glm_base[123].txt`.
