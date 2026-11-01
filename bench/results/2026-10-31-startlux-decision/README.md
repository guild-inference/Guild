# StartLux-Decision-v1-35B-A3B: typed decisions on the GTX 1070 + dual Xeon

Measured 2026-10-31 on this PC: Ubuntu (kernel 6.8), two Xeon Gold 6130 (2 NUMA nodes, 32 cores / 64 threads,
AVX-512 without VNNI or BF16), 376 GiB RAM, GTX 1070 8 GiB (sm_61), driver 580.178.04, CUDA 12.0.140, GCC 13.3.0.
Guild branch `feat/startlux-decision` (on `feat/glm5` `196527e`), Release, `-DGUILD_EXPERIMENTAL_SM60=ON`, with
`GUILD_CUDA_SOURCE_DIR` on guild-kernels-cuda `perf/sm61-decode` `b511c94` and `GUILD_CPU_SOURCE_DIR` on
guild-kernels-cpu `5147f69` (the lock's).

[StartLux-Decision-v1-35B-A3B](https://huggingface.co/startlux-models/StartLux-Decision-v1-35B-A3B) (CC BY-NC 4.0)
answers typed questions about a state (the TypeSafe `/v1/systemone` format): a probability for every option of a
choice, a yes / no (noul) or a score, read from the next-token logits of the option letters.  Nothing is generated.

## What was reused and what is new

The checkpoint is `qwen3_5_moe`: 40 layers (30 Gated DeltaNet, 10 gated full attention), 256 experts with 8
active and a shared expert, hidden 2048, the Qwen3.5 tokenizer.  That is Ornith-1.5-35B's topology, which Guild
runs as `qwen35moe` (`runner.qwen35moe.v1`), so every kernel and the whole engine are reused: the GDN and attention
kernels, the MoE path (GPU dense, CPU experts on both NUMA nodes), the prefill engines and the verifier.  No kernel
was written.

| Piece | Reused | New |
| --- | --- | --- |
| Weights | llama.cpp's Qwen3.5-MoE converter, `llama-quantize`, `tools/iq_pack.py` (the pack and its tokenizer export) | the vision tower and the MTP layer are not used |
| Engine | all of it | `GuildSession::score`: a prompt's next-token logits without generating, with a checkpoint of a shared beginning (`generate`'s prefix reuse and prompt reading, moved into two helpers both call; GLM-5.3-Flash through its own engine) |
| Prompt | `ChatTemplate` (the Qwen3.5 template with thinking off), `Tokenizer` | `src/decision`: a port of the checkpoint's `jevfmt.py` (rendering) and `model.py` (readout, wide choices, TypeSafe answers), with the Python text conversions that shape a prompt (`json.dumps` of a JSON state, `str.strip`, `str.lower`, `float` for legends); `JsonValue` keeps a number's literal |
| Runner | guild-generate's model options and load, moved to `src/program/model_options.cpp` | `decision_run` |
| Registry | `ModelRegistry`, `ModelManifest` | `license` and `commercial_use` fields; the model's entry is `cc-by-nc-4.0`, commercial use off (`guild show` prints it) |

Prepare the model once:

```bash
python <llama.cpp>/convert_hf_to_gguf.py StartLux-Decision-v1-35B-A3B --outtype bf16 --outfile SL-BF16.gguf   # 71 GB
llama-quantize SL-BF16.gguf SL-Q8_0.gguf Q8_0                                                               # 37.8 GB
python tools/iq_pack.py --gguf SL-Q8_0.gguf --out pack-Q8_0 --compat-bf16
cp StartLux-Decision-v1-35B-A3B/decision_config.json pack-Q8_0/tokenizer/
decision_run --native SL-Q8_0.gguf --pack pack-Q8_0 --resident-budget-gib 40 --requests in.jsonl --out out.jsonl
```

`in.jsonl` holds one request a line (`{"id", "state", "questions"}`); `--detail` adds each question's prompt tokens,
letter logits and probabilities.  Requests with images are refused with an error.

## The gate

`tests/data/startlux_decision`: 244 requests, 258 questions:
- the 231 public JevBench items (Intern-Decision's `benchmarks/accuracy-v1/jevbench`, MIT);
- the model card's example and edge cases of the rendering;
- four long inputs of 8.9K, 25K, 31K (Guild docs and C++) and 23K tokens (a JSON state).

The reference is the checkpoint's own package (`startlux_decision/`) with transformers 5.20.2 and torch 2.14.1 on the
CPU (`tools/decision_reference.py`), in bf16 (its own setting) and in fp32.  The package's shared-prefix path fails
on this transformers version (its linear-attention cache keeps its states in dicts), so the four long requests read
each prompt whole, the same computation.

- **Prompts:** all 255 prompts the reference recorded render to the same tokens in Guild (`decision_render_test`), the
  31K-token one and the JSON states included.
- **The readout.** The letters' logits are the final hidden state times the checkpoint's own BF16 head rows for
  the 26 letters, in double precision, as the package's `model.py` reads them (`h.float() @ letter_rows.T`):
  `tools/decision_letter_rows.py` writes the rows beside `decision_config.json`, and `GuildSession::score` returns the
  hidden state with the logits.  Against Guild's Q8_0 head it moved one question across 0.01 (59 -> 60) and the 95th
  percentile of max |dp| from 0.03722 to 0.03680: the head was not where the error comes from.
- **The gate as decided** (`gate.json`, a fixed test; `decision_gate_test` runs it when the model's paths are
  configured).  The reference is the package in fp32; the bounds are the package's own bf16 run's distance from it:
  1. the top answer equals fp32's wherever fp32's top two probabilities differ by 0.05 or more (near-ties may flip);
  2. no more questions with max |dp| over 0.01 than bf16 has (52 of 258), and a 95th percentile of max |dp|
     (nearest rank) no larger than bf16's (0.03685);
  3. the same prompt tokens.

| Condition | bf16 (the bound) | Guild Q8_0 | |
| --- | --- | --- | --- |
| 1. flips where fp32's margin >= 0.05 | none | `hard-opus-c-temporal_numeric-04` (margin 0.0614) | **fails** |
| near-tie flips (allowed), fp32 margin | `temporal_numeric-07` 0.0111, `temporal_numeric-12` 0.0033, `long-json-20k`/`common` 0.028 | `temporal_numeric-07` 0.0111, `sol-a-multi_hop-09` 0.0138, `long-json-20k`/`common` 0.028 | |
| 2a. questions over 0.01 | 52 | 60 | **fails** |
| 2b. 95th percentile of max \|dp\| | 0.03685 | 0.03680 | passes |
| 3. prompt tokens equal | | 255 of 255 | passes |

  The test fails, and gives the same result every run (regress2's ctest: the same flips, counts and percentile).
  `temporal_numeric-04` flips under every Q8_0 computation measured: llama.cpp on the same GGUF, and the package in
  fp32 with every matrix rounded to Q8_0.
- **All 258 questions** against fp32:

| Run | Same top answer | max \|dp\| <= 0.01 | Median / 90th / 95th / max \|dp\| |
| --- | ---: | ---: | --- |
| Guild Q8_0 (the default) | 254 / 258 | 198 / 258 | 0.0014 / 0.022 / 0.0368 / 0.144 |
| the reference package in bf16 | 255 / 258 | 206 / 258 | 0.0013 / 0.026 / 0.0369 / 0.086 |
| Guild Q4_K_M (before the letter-rows readout) | 249 / 258 | 155 / 258 | 0.0059 / 0.057 / - / 0.537 |

On the 69 questions of 500 tokens or more that llama.cpp was run on, against fp32 (before the letter-rows readout):

| Run | max \|dp\| <= 0.01 | Median / 90th / max \|dp\| | Same top answer |
| --- | ---: | --- | ---: |
| Guild Q8_0 | 32 / 69 | 0.0120 / 0.049 / 0.143 | 66 / 69 |
| llama.cpp `11fe02151`, the same Q8_0 GGUF (`llama-server` log-probabilities, as the package's `gguf_server.py` reads them) | 28 / 69 | 0.0125 / 0.035 / 0.080 | 65 / 69 |
| the reference in fp32 with every matrix rounded to Q8_0 (blocks of 32, no activation quantization) | 38 / 69 | 0.0089 / 0.043 / 0.064 | 66 / 69 |
| the reference in bf16 | 36 / 69 | 0.0090 / 0.041 / 0.070 | 67 / 69 |

- **The quant.** On a first 21-question set against fp32, Q8_0 stayed within 0.0066, Q6_K moved up to 0.032 and
  Q4_K_M up to 0.060 (the same top answers).  Q8_0 is the default.
- **Where the error comes from.** The probabilities move with the precision of the arithmetic.
  - The model's own bf16 run is as far from fp32 as Guild's Q8_0, and so is llama.cpp on the same file.
  - Rounding the weights to Q8_0 alone, with fp32 activations, already moves 31 of the 69 longer questions by more
    than 0.01.
  - The error grows with the prompt: Guild Q8_0's median \|dp\| is 0.0005 under 200 tokens and 0.012 at 2K-4K.

- **The 0.143 outlier** (`hard-opus-b-multi_hop-04`, 2,359 tokens; condition 3 of the decision).  fp32 is fairly
  confident there and Guild is not, so it was followed down to the layers:

| Run | chen | ana | bjorn | esra | max \|dp\| |
| --- | ---: | ---: | ---: | ---: | ---: |
| the package in fp32 (the reference) | 0.479 | 0.229 | 0.184 | 0.060 | |
| the package in bf16 | 0.503 | 0.170 | 0.204 | 0.080 | 0.058 |
| fp32 with Q8_0-rounded weights | 0.433 | 0.264 | 0.137 | 0.105 | 0.046 |
| llama.cpp, the same Q8_0 GGUF | 0.442 | 0.309 | 0.161 | 0.058 | 0.080 |
| Guild Q8_0 | 0.335 | 0.324 | 0.191 | 0.091 | 0.144 |

  - **Per layer.** The residual stream at the last position, relative L2 against fp32: Guild stays within the
    envelope of the other runs up to layer 20 (0.025; fp32 with Q8_0 weights 0.020, bf16 0.033), and against fp32
    with Q8_0 weights it stays at 0.012-0.023 (Q8_0 activations on top of the same weights).  Then it jumps at
    layer 21 (0.025 -> 0.041) and again at layer 35.
  - **The routing explains the jump.** The last token's router (top 8 of 256 experts) has its 8th and 9th
    probabilities within 0.001 of each other in 30 of the 40 layers in fp32, and within 0.0001 in 7 (layers 10, 12,
    21, 22, 25, 27, 35; 0.00004 at layer 21).  Every reduced precision picks some of those near-ties the other way:
    Guild swaps one expert in 7 layers (21, 22, 25, 26, 27, 33, 39), fp32 with Q8_0 weights in 9 (22-28, 30, 35),
    and only 2 of those swaps are the same (layers 22 and 26).  Guild's first swap is layer 21, where its error
    jumps; before it, nothing in its arithmetic stands out.
  - **So:** not a kernel defect that shows anywhere else, but a prompt whose answer hangs on a chain of expert
    choices that are ties at fp32's own precision; which computation lands nearer fp32 is luck of the draw (bf16
    0.058, llama.cpp 0.080, Guild 0.144).  Noted, not fixed.


- **JevBench accuracy**, correct of easy / original / hard:

| Run | Easy | Original | Hard |
| --- | ---: | ---: | ---: |
| the reference in fp32 | 48 / 48 | 71 / 72 | 89 / 111 |
| the reference in bf16 | 48 / 48 | 71 / 72 | 91 / 111 |
| Guild Q8_0 (the letter-rows readout) | 48 / 48 | 71 / 72 | 88 / 111 |
| Guild Q4_K_M (before it) | 48 / 48 | 71 / 72 | 88 / 111 |

## Speed

`decision_run --repeat 5 --fresh` (each run from a reset session; mean and standard deviation of the five), the
Q8_0 GGUF (37.8 GB, its experts in RAM split over both NUMA nodes), `perf_requests.jsonl`'s prose and C++ states:

| Request | Tokens | Q8_0 | Q4_K_M |
| --- | ---: | --- | --- |
| one question | 503 | 2441 ms (stdev 13) | 2105 ms (stdev 2) |
| three questions on one ~430-token state | 1470 | 2951 ms (stdev 21) | 2565 ms (stdev 15) |
| prefill, 1K prompt | 940 | 205.3 tok/s (stdev 0.4) | 235.4 tok/s (stdev 1.2) |
| prefill, 8K prompt | 7953 | 183.9 tok/s (stdev 0.2) | 206.3 tok/s (stdev 0.4) |
| prefill, 32K prompt | 32003 | 134.8 tok/s (stdev 0.2) | 147.8 tok/s (stdev 0.3) |

The three questions' prompts share their first 430 tokens: those are read once and checkpointed, and each question
reads only its own ~60 tokens, so two more questions add 0.5 s.  A prompt's speed is Ornith's prefill on this PC (the
CPU experts bound it, as `docs/performance/PHASE6_LONG_CONTEXT_OPTIMIZATION.md` measured), not something this model
adds.

## Nothing else moved

The branch against `build-cuda12` (`feat/glm5` `196527e`), one machine lock, `regress.sh`, run twice: before the
letter-rows readout (regress1) and at the branch's head `66d5356` (regress2, 2026-10-31 21:44-23:02 UTC):
- **ctest (regress2):** 125 tests, 3 failures:
  - `ple_parity`: its fixtures are not on this machine, as on `feat/glm5`.
  - `decision_gate_test`: the gate above, failing as it should.
  - `multi_model_isolation_test`: timed out at 183 s against its 180 s limit (173 s in regress1).  Run alone right
    after, it passed.
- **Qwen3.8-Flash-Next:** qwen-bench, server flags, base / branch / branch / base, tok/s:
  - prose: 34.04 / 34.28 / 34.23 / 34.19;
  - code: 44.74 / 44.76 / 44.75 / 44.63.
  - All texts identical.
- **GLM-5.3-Flash:**
  - glm5_run, base / branch alternated, tok/s:
    - base 21.23 (the first run, colder), 21.68, 21.81;
    - branch 21.54, 21.90, 21.80.
    - The same ids in all six.
  - guild-generate (the runtime's GLM path): the same text, 20.75 against 21.45 tok/s.
  - `session_score_test` on GLM: the argmax and repeat checks pass; reading a prompt in two parts differs from one
    read (max \|dlogit\| 3.03), as before this branch (below).

## Known gaps

- **The gate fails** at Q8_0 on two of its conditions (above).  No full-precision path was built, as decided.
- **Prompt length.** Guild's dense attention for this topology holds 32,768 cells (`kTopkMaxCells`), so a prompt
  is at most 32,768 tokens.  The model card's 262,144 needs multi-tile attention.
- **Images** are refused.
- **Text conversions not reproduced.** `str.lower` is reproduced for Latin, Greek, Cyrillic, Armenian and fullwidth
  letters only (used only to test whether a criterion repeats its option id).  `str()` of a list or object inside
  a criteria list is refused.
- **Prefix reuse is exact only on Ornith's topology** (found, not changed).  `session_score_test` reads a prompt in
  two parts (a checkpoint after 300 of 348 tokens) and in one:
  - Ornith-1.5-35B: bit for bit.
  - Qwen3.8-Flash-Next: max \|dlogit\| 1.35-6.4.
  - GLM-5.3-Flash: 0.97-3.0.

  `generate`'s prefix cache does the same, before this branch too: `build-cuda12`'s guild-generate gives
  Qwen3.8-Flash-Next another text from token 18 after reusing a 299-token checkpoint than from a fresh read.

Files: `perf_requests.jsonl` and `perf_{Q8_0,Q4_K_M}.jsonl` (the speed runs), `gate_Q8_0.jsonl` (decision_run's
answers to the fixture, `--detail` without the prompt tokens, which the fixture holds), `regress.sh`.
