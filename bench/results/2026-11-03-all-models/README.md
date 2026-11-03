# Every model on the dual Xeon + GTX 1070, 2026-11-03

The six models this PC runs, measured one after another with the branch's head and its pinned kernels, each the
way its own results measured it before.

PC: 2x Xeon Gold 6130 (16 cores each, AVX-512 BW without VNNI, 2 NUMA nodes, 376 GiB RAM), GTX 1070 8 GB (sm_61,
15 SMs, 151 W cap), CUDA 12. Guild `feat/nemotron-transcribe` at `38ac0b6`, guild-kernels-cpu `9b7281a` and
guild-kernels-cuda `de65d60` (the revisions `backends.lock.json` pins).

| Model | Decode, with its drafts | Decode, no drafts | A longer prompt | Other |
| --- | --- | --- | --- | --- |
| Qwen3.8-Flash-Next UD-IQ4_XS, setup's settings | prose 45.9, code 60.3 tok/s; story 42.9, sql 64.9, Chinese 32.2, Russian 36.0 | 25.8 tok/s | 4,008 tokens in 52.2 s (77 tok/s) | first token 0.41-0.46 s after a new 34-41-token prompt |
| GLM-5.3-Flash UD-IQ4_XS, 256K context | prose 22.1, code 22.1, story 20.1, sql 23.4 tok/s | 15.2-15.7 tok/s | 1,974 tokens in 27.6 s (72 tok/s) | |
| Ornith-1.5-35B Q4_K_M | prose 79.9, code 92.8 tok/s | 60.3 tok/s | 1,024 tokens at 442 tok/s | |
| StartLux-Decision Q8_0 | | | a 940-token prompt in 2.15 s | a 503-token question in 1.12 s |
| Nemotron 3.5 Transcribe Q8_0 | a 29.6 s clip in 2.31 s (0.078 of real time) | 3.03 s (0.102) | | a 5.0 s clip in 0.62 s (0.82 s without drafts) |
| Fish Audio Drama 3 Q4_K_M | | | | speech at 1.79-1.85x real time (38.6-39.8 frames/s) |

Qwen's two runs of each prompt, and Nemotron's five of each clip, gave the same token ids; Qwen and Ornith also
give the same texts with their drafts as without.

## Qwen3.8-Flash-Next (Unsloth UD-IQ4_XS)

`tools/bench_runtime.py` (`qb.sh`) with setup's run config for this model: the DFlash 2 drafter (Q4_K weights, the
multilingual head subset, `--spec 8`), `--dense q5_k --hc q8_0`, and the bench's usual `--context 32768`, FP16 KV in
host memory, `--resident-budget-gib 56`. 2 runs x 256 tokens a prompt, tok/s:

| Prompt | With the drafts | Tokens a window | Without (`decode_strategy=normal`) |
| --- | --- | ---: | --- |
| prose | 45.81 / 45.93 | 2.64 | 25.75 / 25.84 |
| code | 59.97 / 60.63 | 3.88 | 25.74 / 25.78 |
| story | 42.92 / 42.82 | 2.39 | |
| sql | 64.70 / 65.07 | 4.27 | |
| Chinese | 32.26 / 32.20 | 1.71 | |
| Russian | 35.99 / 35.99 | 1.83 | |

- **A ~4K-token prompt** (`qwen4k.json`: the first 4,000 ids of the decode-roofline lanes' 32K prompt and its last 8;
  no prefix cache): 4,008 tokens in 52.19 / 52.34 s, then 46.48 / 46.68 tok/s. The prompt is PCIe-bound: an 8 GB
  card streams the experts from RAM for every 512-token chunk.
- **The first token** after a new short prompt: 0.41-0.46 s; when the server already holds the prompt, 0.07 s.
- **Without drafts** a token takes 38.8 ms; the drafter makes a window of 2.6-3.9 tokens on English text and code in
  57-65 ms. The texts are the same with the drafts and without.

## GLM-5.3-Flash (Unsloth UD-IQ4_XS)

`glm5_run --spec --prompts` with GLM's defaults: the NextN block drafting one token, the attention re-quantized to
Q6_K at load, the shared experts of a window's second token on the CPU, a 262,144-token context. The four chat
prompts of round 9, 256 tokens each, one run; "no drafts" is `--plain` in the same process:

| Prompt | With NextN | Kept | Median step | No drafts | Median step |
| --- | --- | ---: | ---: | --- | ---: |
| prose | 22.12 | 78% | 79.4 ms | 15.20 | 66.4 ms |
| code | 22.07 | 82% | 79.7 ms | 15.56 | 63.9 ms |
| story | 20.09 | 62% | 80.2 ms | 15.62 | 63.6 ms |
| sql | 23.42 | 91% | 80.0 ms | 15.73 | 63.5 ms |

- **A 1,974-token prompt** (text 1 of round 6's KL measurements): 27.56 s (71.6 tok/s), then 32 tokens at 20.0
  tok/s.
- **The DFlash drafter** (opt-in; its checkpoint is for research and evaluation) with verify windows of up to four
  tokens measured prose 21.2, code 23.3, story 18.2 and sql 26.0 tok/s, with `GUILD_GLM_SHEXP_SPLIT=0`
  ([round 10](../2026-11-03-decode-round10/README.md)).

## Ornith-1.5-35B (Q4_K_M)

`tools/bench_runtime.py` with Ornith's server settings: its MTP layer (`--mtp`, `--spec 4`), `--context 32768`,
FP16 KV in host memory, `--resident-budget-gib 24`. 2 runs x 256 tokens:

| Prompt | With MTP | Tokens a window | Without (`--decode-strategy normal`) |
| --- | --- | ---: | --- |
| prose | 79.52 / 80.19 | 2.03 | 60.45 / 59.93 |
| code | 92.84 / 92.79 | 2.44 | 60.44 / 60.49 |

- The texts with and without the drafts are the same, token for token.
- **A 1,024-token prompt** (`ornith_prefill_bench --prompt 1024 --batch 256 --context 4096`, 3 runs): 441.96 tok/s,
  first token 2.34 s. The same bench's 256 tokens after it: 59.3 tok/s without drafts, 137.2 with MTP; its prompt is
  synthetic and drafts unusually well, so the bench prompts above are the ones to quote.

## StartLux-Decision (Q8_0)

`decision_run --repeat 5 --fresh` on the four short requests (`perf-short.jsonl`), ms, mean of 5:

| Request | Tokens | Mean | Runs |
| --- | ---: | ---: | --- |
| warm-up | 243 | 570 | 602 / 578 / 565 / 561 / 543 |
| one question | 503 | 1115 | 1096 / 1110 / 1136 / 1125 / 1109 |
| three questions on one state (898 tokens shared) | 1470 | 1555 | 1534 / 1493 / 1536 / 1624 / 1588 |
| a 940-token prompt | 940 | 2146 | 2177 / 2131 / 2148 / 2119 / 2156 |

## Nemotron 3.5 Transcribe (Q8_0)

`transcribe_run --repeat 6` on the 5.0 s and 29.6 s LibriSpeech clips, greedy and with 3 MTP drafts, the median of
the 5 runs after a warm-up:

| Clip | Greedy: total (prompt, decode) | Real-time factor | 3 drafts: total (prompt, decode) | Real-time factor |
| --- | --- | ---: | --- | ---: |
| 5.0 s | 817 ms (346, 426) | 0.163 | 623 ms (360, 217) | 0.125 |
| 29.6 s | 3030 ms (890, 1957) | 0.102 | 2311 ms (947, 1194) | 0.078 |

## Fish Audio Drama 3 (Q4_K_M)

`guild-generate --drama3 --say`, the 290-character passage of the Drama 3 results, the model card's sampling, seeds
11 and 12, `--max-context 10240 --kv fp16 --kv-host-only --resident-budget-gib 24`:

| Voice | Audio | Frames/s | Real time | Prompt |
| --- | --- | ---: | ---: | --- |
| none | 22.24 / 19.78 s | 39.84 / 39.02 | 1.85x / 1.81x | 86 rows, 0.50 / 0.54 s |
| en0 (3.9 s clip, encoded in 0.40-0.41 s) | 18.67 / 17.14 s | 39.40 / 38.56 | 1.83x / 1.79x | 206 rows, 1.33 / 1.34 s |

The audio lengths are the 2026-10-25 runs' for these seeds. `run.sh`'s en0 runs passed the voice's `en0.codes`, a
format `--ref-codes` does not read (one frame of 10 codes a line); `run2.sh` passes the WAV instead.

## Not measured here

- The GSQ-RCO quants (setup's default family), Swift 1.5 and the Coder: their files are not on this PC; nor is a
  pack for the Qwen Q4_K_M GGUF in `~/models`.
- Long contexts: GLM's 128K prompt and 256K decode, and Qwen's 32K and 128K, are in
  [2026-11-01-glm-256k](../2026-11-01-glm-256k/README.md) and
  [2026-10-30-decode-roofline](../2026-10-30-decode-roofline/README.md).
- Load times: the six models do not fit the OS file cache together, so a load took 7-11 s (Ornith) to 334 s (GLM),
  depending on what the run before it left there.

## Files

`run.sh` and `run2.sh` (the runs, in order), `qb.sh` (the Qwen and Ornith bench wrapper), `more.json` (the four
extra prompts' ids), `summary.py`; `runs/` holds the reports (`b-*.json`: every run's ids and timings) and the other
models' outputs.
