# Speed across the models: overlapping the prefill, the CPU expert kernels

After the Nemotron 3.5 Transcribe pass (`../2026-11-01-nemotron-perf/`), the same two ideas on the other engines:
keep the GPU and the CPU busy at once where a layer used to wait on one of them, and make the CPU expert rows of
the formats this PC runs stream at the memory's speed.  Every change keeps each token's arithmetic: the outputs are
the same bits.

PC: 2x Xeon Gold 6130 (AVX-512 without VNNI, 2 NUMA nodes), GTX 1070 8 GB, CUDA 12.  Guild
`feat/nemotron-transcribe` (`c104042`, `5485490`, `a12a30e`, `cdf65cb`); guild-kernels-cpu `perf/q8-avx512`
(`c12ae99`, local, on the pinned `5147f69`: **not in `backends.lock.json`** - a build takes it only through
`GUILD_CPU_SOURCE_DIR`).

| Model | What | Before | After | Outputs |
| --- | --- | ---: | ---: | --- |
| StartLux-Decision Q8_0 | a 503-token question | 2.41 s | 1.60 s (pairs) / 1.19 s (+ Q8_0 rows) | the same answers |
| | a 940-token prompt | 4.55 s | 3.11 s / 2.23 s | |
| Ornith-1.5-35B Q4_K_M | 1,024-token prefill | 244 tok/s | 345 tok/s | the same ids |
| GLM-5.3-Flash | decode step (2 tokens) | 84.7 ms (CPU experts 69.6) | 83.9 ms (67.1) | the same ids |
| Nemotron 3.5 Transcribe | 29.6 s clip, 3 drafts | decode 1195 ms | 1170 ms | the same ids |
| Qwen3.8-Flash-Next | decode / a 4K prompt | | unchanged | the same texts |

## 1. The batched prefill: two halves of a chunk at a time (`cdf65cb`)

Ornith-family models (Ornith, StartLux-Decision) read a prompt with `BatchedPrefill`: per layer of a chunk, the
GPU's mixer and routing, then the CPU's routed experts, then their combine, each waiting for the other.  On
StartLux-Decision's 940-token prompt (`GUILD_PREFILL_PROFILE_CSV`, per run): GPU projections 1415 ms, the mixers
261, the CPU experts 1648, one after the other.

Now each chunk runs as two halves of rows: at each layer this thread runs one half's experts while the GPU runs
the other half's layer up to its routing.  The stream keeps every layer's order (half A before half B: the
recurrent states and the attention cells pass from one to the next), so each half computes what a chunk of its rows
computes alone, and the batched prefill already equals the sequential ingestion bit for bit at every batch size.
Expert jobs hold at most 8 tokens, so a half reads no more expert bytes than the chunk did.  `GUILD_PREFILL_PAIRS=0`
is the control; profiling and per-layer callbacks run one half.

- `ornith_prefill_test` passes: 512 tokens at batch 1-512 against the sequential ingestion, the logits, the GDN/conv
  state, the KV cells and the output bitwise (Ornith at batch 512: 353 tokens/s).
- Ornith-1.5-35B Q4_K_M (`ornithab.sh`, `ornith_prefill_bench --prompt 1024 --batch 256`, 3 runs, alternated twice):
  prefill 4.19 / 4.20 s -> 2.97 / 2.98 s (244 -> 345 tokens/s); the CPU's wait for the GPU 2.44 -> 1.07 s; the
  expert rows the same; decode 59.5-59.9 tokens/s in both.
- StartLux-Decision Q8_0 (`abpairs.sh`, `decision_run --repeat 5 --fresh`, alternated twice, ms):

| Request | Pairs off | Pairs | Pairs + Q8_0 rows (section 2) |
| --- | ---: | ---: | ---: |
| warm-up, 243 tokens | 1181 / 1196 | 800 / 809 | 593 / 598 |
| one question, 503 tokens | 2406 / 2417 | 1596 / 1601 | 1184 / 1198 |
| three questions on one ~430-token state | 2913 / 2941 | 2092 / 2095 | 1541 / 1551 |
| a 940-token prompt | 4534 / 4556 | 3115 / 3110 | 2224 / 2229 |

Every answer's probabilities are the same bytes as before the change, in every arm.

Ornith is now GPU-bound in the prefill (the CPU still waits 1.07 s of 2.97): the dense projections run the decode's
MMVQ arithmetic (kept so that the prefill equals the sequential ingestion), already reading each weight row from DRAM
once for many columns.  A faster prefill there needs a tiled kernel that sums every output in the MMVQ's order: a
guild-kernels-cuda change, not made here.

**Qwen3.8-Flash-Next** reads prompts with its own engine (`prefill::Prefill`, its routed experts on the GPU, the
non-resident ones streamed from RAM over PCIe): a ~4K-token prompt takes 53.9 s with or without this change
(`qwenlong.sh`), the same text.  Its experts on the CPU instead (as the Ornith path does) would be another
arithmetic, and so other texts: not done.

## 2. Q8_0 expert rows on AVX-512 in guild-kernels-cpu (`c12ae99`, local)

StartLux-Decision's Q8_0 experts went, in guild-kernels-cpu's native path, through ggml-cpu's one-token dot for the
gate/up rows of every token (kq256's fused gate/up takes only Q4_K and Q6_K) and through kq256's AVX-2 kernel for the
down rows.  The branch adds the same two-rows-to-a-register AVX-512BW kernel as Guild's `q8_0_rows512`
(`../2026-11-01-nemotron-perf/`), bit-exact against ggml's AVX-2 dot for every token, and routes both to it.
`iq_ggml_order_test` checks the rows and the gate/up against ggml's dot at 1-8 tokens (AVX-512BW, and AVX-2 with
`GUILD_Q8_512=0`) and the Q8_0/Q8_0 dispatch; the repository's 4 tests pass (`kcpu_test.sh`).

StartLux-Decision Q8_0 with it (`abq8.sh`, before the pairs, 5 runs, alternated twice): a question 2.41 -> 1.90 s,
three questions 2.92 -> 2.36 s, a 940-token prompt 4.54 -> 3.57 s; with the pairs, the table above.  The same
answers.  Qwen3.8-Flash-Next (Q8_0 down rows in two layers): the same texts and decode speed.

**Using it needs the CPU lock moved** to a pushed revision with this commit; the branch is local and unpushed.

## 3. GLM-5.3-Flash's CPU experts (`c104042`, `5485490`)

GLM's decode keeps 92-94% of the expert threads busy inside the phases (`GUILD_GLM_TIMING` now reports it), so the
phase ends cost little; the rows streamed at ~164 GB/s against the ~210 one thread per core reaches on its own.  Two
changes:
- the rows of ggml-cpu's one-token dot (GLM's IQ4_XS down rows, mostly one token an expert) are asked for two rows
  ahead, and a down item's last rows ask for the next group's first;
- the IQ3_S gate/up rows (NL4, Guild's own kernel) two to an AVX-512BW register, bitwise the AVX-2 rows
  (`glm5_nl4_test`: 23,976 results); every core streaming its own 4,096-value rows, two tokens 163 -> 200 GB/s, one
  token 208 -> 209 (asking one row ahead instead of two: 187, which first made the engine 11% slower).

`glmab2.sh`, `glm5_run --spec` (5-token prompt, 256 tokens), HEAD / both / the prefetch alone, three rounds, the
first HEAD run left out (its median step was 120 ms: something else on the machine):

| Arm | Median step, ms | CPU experts, ms a step | gate/up, s | down, s |
| --- | ---: | ---: | ---: | ---: |
| HEAD (rounds 2, 3) | 84.9 / 84.5 | 70.2 / 69.0 | 5.98 / 5.81 | 3.74 / 3.66 |
| the prefetch | 84.1 / 84.2 / 84.1 | 67.9 / 67.8 / 67.6 | 5.92 / 5.85 / 5.85 | 3.48 / 3.46 / 3.44 |
| the prefetch + NL4 AVX-512 | 83.8 / 84.0 / 83.9 | 67.1 / 67.2 / 66.9 | 5.77 / 5.76 / 5.75 | 3.45 / 3.46 / 3.45 |

The same ids in all nine runs.  The step moves less than its CPU part: ~17 ms of a step lies outside the CPU
experts (the drafts, the accept, the verify window's GPU work that does not overlap them).

## 4. Nemotron 3.5 Transcribe: the accepted rows with the next draft (`a12a30e`)

`accept()` ran the MTP block over a verify window's kept rows on its own; they now join the next draft's first
step (at most four rows, each computed as alone, so the same drafts).  The 60 gate clips: the same ids, drafted and
accepted counts with 1-3 drafts.  The 29.6 s clip with three drafts: accept 1.5 -> 0.1 ms a step, a draft step
+0.9 ms, decode 1195 -> 1170 ms.

## Files

`abpairs.sh` (the parity test, StartLux pairs), `abq8.sh` (StartLux, the kernel branch), `ornithab.sh`, `glmab2.sh`,
`qwenlong.sh` (with `qwen4k.json`: the first 4,000 ids of the decode-roofline lanes' 32K prompt and its last 8), `kcpu_test.sh` (the kernel repository's tests on the branch); `runs/` holds the StartLux answers and timings and
the Ornith CSVs.
