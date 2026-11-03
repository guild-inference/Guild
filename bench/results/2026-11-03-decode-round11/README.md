# Eleventh round: longer prompt chunks on an 8 GB card, Drama 3's tower and codec beside the backbone

Two changes, each with the same outputs as before:

- **Qwen3.8-Flash-Next's prompt chunks grow into the free VRAM.** On an 8 GiB card the planner gives prompt chunks
  of 512 tokens, and each chunk streams every routed expert the GPU does not hold over PCIe. When the chunk is the
  planner's (`--prefill auto`), the load now grows it into the VRAM left after the weights and the drafter: up to
  4096 tokens, about 0.40 MiB a token (168 B x `n_embd`), with 768 MiB kept back. The longer chunk is computed as
  chunks of 512 (`--prefill-rows`), which on Pascal (llama.cpp's MMQ without stream-k) gives the 512-token chunks'
  results bit for bit. Newer cards keep 512 unless `GUILD_PREFILL_GROW=1`; `--prefill N` keeps N. Setup writes
  `--prefill auto` instead of 512 for a card of 8 GiB or less when the engine's `--help` says the chunk grows.
- **Drama 3 runs the tower's last eight residual layers, and the codec, beside the next backbone row.** The backbone's
  next row reads only the frame's main token and code 1. So `Drama3Tower::step_begin` runs the shared layers and the
  first residual layer (main, c0, c1), and `step_finish` (c2..c9) runs on a second thread while the backbone takes
  the next row. The codec decodes each 16-frame chunk on a third thread while the next frames are made.
  `GUILD_DRAMA3_OVERLAP=0` turns both off.

PC: 2x Xeon Gold 6130 (16 cores each, AVX-512 BW without VNNI, 2 NUMA nodes, 376 GiB RAM), GTX 1070 8 GB (sm_61,
15 SMs, 151 W cap), CUDA 12. Branch `feat/nemotron-transcribe`, guild-kernels-cpu `9b7281a` and guild-kernels-cuda
`de65d60` (the revisions `backends.lock.json` pins).

## Qwen3.8-Flash-Next: prompt chunks

Unsloth UD-IQ4_XS with setup's settings (the DFlash 2 drafter with Q4_K weights and the multilingual head subset,
`--spec 8`, `--dense q5_k --hc q8_0`), `tools/bench_runtime.py`, `--context 32768`, FP16 KV in host memory,
`--resident-budget-gib 56`, no prefix cache (`GUILD_NO_KV_CACHE=1`). The load left 2,954 MiB of VRAM free.

A 4,008-token prompt (the all-models bench's `qwen4k.json`), then 256 tokens, one run a setting:

| Prompt chunk | Prompt | Decode after it | VRAM after the run |
| --- | ---: | ---: | ---: |
| 512 (before; the all-models runs, `38ac0b6`) | 52.19 / 52.34 s | 46.5-46.7 tok/s | 5,925 MiB |
| 1024 | 34.78 s | 46.26 | 6,145 MiB |
| 1536 | 29.86 s | 45.64 | 6,359 MiB |
| 2048 | 24.87 s | 45.58 | 6,583 MiB |
| 3072 | 24.75 s | 46.38 | 6,965 MiB |
| 3584 (`auto`, a first, looser size rule) | 24.35 s | 46.96 | 7,155 MiB |
| 4096 | 19.66 s | 46.27 | 7,353 MiB |
| `auto` (grows to 4096) | 19.63 s | 46.91 | 7,353 MiB |

Every setting generated the same 256 tokens. 2048 and 3072 take the same time: a 4,008-token prompt is two chunks
either way.

**A 31,918-token prompt** (the decode-roofline lanes' 32K prompt), then 256 tokens: chunks of 512, 432.58 s; `auto`
(4096), 169.69 s (188 tok/s). Decode after it was 40.70 and 41.00 tok/s, with the same tokens.

**Short prompts.** The bench's prose and code prompts, alternated over 2 rounds, chunks of 512 against 2048 (both
`--prefill-rows 512`): prose 44.36 / 44.86 against 45.27 / 45.44 tok/s, code 58.06 / 58.75 against 59.71 / 59.79,
the same tokens. The longer chunk does not slow decode.

## Ornith-1.5-35B: MTP draft settings

Ornith's server runs its MTP layer with `--spec 4` and a minimum draft probability of 0.5. `tools/bench_runtime.py`
with its server settings, prose and code, 256 tokens, one run a setting, tok/s:

| `--spec` | min-p 0.3 | min-p 0.5 | min-p 0.7 |
| --- | --- | --- | --- |
| 3 | 79.33 / 91.02 | 80.33 / 90.91 | 78.08 / 86.84 |
| 4 | 76.82 / 92.42 | **79.47 / 92.07** (the server's) | 77.18 / 88.19 |
| 5 | 72.86 / 89.48 | 77.74 / 91.12 | 77.39 / 89.56 |
| 6 | 71.56 / 87.41 | 77.21 / 90.11 | 76.78 / 88.59 |

Every setting generated the same texts. No setting beats the server's by more than 1.1% on both prompts, which is
within one run's spread (the all-models runs measured 79.52 / 80.19 and 92.84 / 92.79 at the server's setting), so
the config stays.

## Fish Audio Drama 3: the tower and the codec beside the backbone

**Where a frame went before.** An nsys capture of a 195-frame decode: 25 ms a frame, of which the backbone took
~13.5 ms, the tower ~9 ms and the codec ~2.7 ms, one after another.
- The GPU did useful work 66% of the time. The backbone's kernels covered 33% of it (not counting their waits on the
  CPU experts), the tower's 25% and the codec's 10%.
- The tower is near its bound: its Q8_0 projections and heads read 1.1 GB a frame in 5.7 of its 7.6 ms of kernels
  (about 92% of the card's bandwidth).
- Its ten host samples a frame (a 4,096-way softmax in doubles, top-k 30) take ~47 us each with the GPU idle.

Faster sampling would gain about 0.5 ms a frame. Overlap gains more: the backbone's next row needs only main and c1,
and during its step the GPU waits about 5 ms on the CPU experts.

**Results.** `guild-generate --drama3 --say`, the 290-character passage of the all-models runs, the model card's
sampling, `--max-context 10240 --kv fp16 --kv-host-only --resident-budget-gib 24`. Three arms alternated over 2
rounds: the new binary with both overlaps, the binary with only the tower's (`ab.sh`'s), and the new binary with
`GUILD_DRAMA3_OVERLAP=0`. Frames/s:

| Voice, seed | Neither | Tower beside | Tower and codec beside |
| --- | --- | --- | --- |
| none, 11 (479 frames, 22.24 s) | 39.28 / 39.07 | 45.65 / 44.39 | 46.49 / 45.75 |
| none, 12 (426 frames, 19.78 s) | 39.21 / 39.04 | 44.39 / 44.24 | 45.94 / 45.51 |
| en0, 11 (402 frames, 18.67 s) | 38.79 / 38.90 (`ab.sh`) | 44.64 / 44.16 | 45.39 / 45.43 |

- **Same audio.** Every WAV of a seed and voice has the same bytes in every arm, and the same bytes as the
  all-models runs of 2026-11-03 (`wav.md5`).
- **Real time.** 1.81-1.82x real time becomes 2.11-2.16x: frames/s +16.6-18.4%.
- **Waiting.** The time the decode loop waits on the tower fell from 3.7-4.2 s a run to 1.5-1.8 s, and on the codec
  from 1.1-1.3 s to 16-40 ms.
- **The codec gains less than its time.** Its large SGEMMs share the GPU with the backbone and the tower's first
  layers, which are on the critical path.

**The server path.** `serve_test.py` drives `guild-generate --serve`. A `SAY` streamed 30 chunks with the same
samples as the `--wav` run. A `SAY` stopped after 3 chunks (`STOP`) ended with `SDONE ... cancel` after 4 chunks,
and the engine exited cleanly.

`on_audio` (`GuildModel::synthesize`) is now called from the codec's thread: one call at a time, in order, and all
before `synthesize` returns. Both callers (`guild-generate --serve` and `--say`) only append or print the chunk.

**Open.**
- Sampling on the GPU would remove the ten host round trips a frame. Its `exp` differs from the host's in the last
  bits, so the draws, and with them the audio, would change; that needs a decision.
- A higher stream priority for the backbone than for the codec would change no numbers. It touches the verify
  streams every model shares, so it is not tried here.

## Files

`qwen/` holds the Qwen runs (`b-*.json`: every run's ids and timings, `*.sh` the scripts), `ornith/` the Ornith
sweep and `drama3/` the Drama 3 A/B (`ab.sh`, `ab2.sh`, their logs, and the WAVs' MD5 sums).
