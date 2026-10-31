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
- **Probabilities** against the fp32 reference (`tools/decision_gate.py`):

| Run | Same top answer | max \|dp\| <= 0.01 | Median / 90th / max \|dp\| |
| --- | ---: | ---: | --- |
| Guild Q8_0 (the default) | 253 / 258 | 199 / 258 | 0.0014 / 0.022 / 0.143 |
| Guild Q4_K_M | 249 / 258 | 155 / 258 | 0.0059 / 0.057 / 0.537 |
| the reference package in bf16 | 255 / 258 | 206 / 258 | 0.0013 / 0.026 / 0.086 |

On the 69 questions of 500 tokens or more, against fp32:

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
  - The flips sit on near-ties: two of Guild's five are also bf16's (`hard-opus-a-temporal_numeric-07`,
    `long-json-20k` / `common`).
- **Not passed as set.** A 0.01 bound on every question against fp32 holds only for an engine close to fp32: bf16
  weights with fp32 activations throughout.
- **JevBench accuracy**, correct of easy / original / hard:

| Run | Easy | Original | Hard |
| --- | ---: | ---: | ---: |
| the reference in fp32 | 48 / 48 | 71 / 72 | 89 / 111 |
| the reference in bf16 | 48 / 48 | 71 / 72 | 91 / 111 |
| Guild Q8_0 | 48 / 48 | 70 / 72 | 88 / 111 |
| Guild Q4_K_M | 48 / 48 | 71 / 72 | 88 / 111 |

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

The branch against `build-cuda12` (`feat/glm5` `196527e`), one machine lock, `regress.sh`:
- **ctest:** all 123 of the branch's tests ran: 116 passed, 6 skipped for missing hardware features or fixtures.
  `ple_parity` failed: its fixtures are not on this machine, as on `feat/glm5`.
- **Qwen3.8-Flash-Next:** qwen-bench, server flags, base / branch / branch / base.
  - ms a window, prose: 62.95-63.24 against 63.08-63.26.
  - code: 66.10-66.41 against 66.29-66.42.
  - All 24 texts identical.
- **GLM-5.3-Flash:**
  - glm5_run, base / branch alternated: the branch's warm runs 21.59 / 21.85 / 21.85 tok/s against 21.88 / 21.87;
    the first base run, cold after the Qwen runs, 10.09.  The same ids in all six.
  - guild-generate (the runtime's GLM path): the same text, 21.62 against 21.61 tok/s.

## Known gaps

- **Precision.** The 0.01-on-every-question gate is not met at Q8_0 (above).
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
