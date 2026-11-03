# Decode, ninth round: the DFlash drafter's kernels and head, and a GLM-5.3-Flash version

The DFlash 2 drafter became Qwen3.8-Flash-Next's default in round 8's follow-up (`e934ee5`: `--dflash`, and setup
fetches the checkpoint). This round makes its drafts cheaper and its load quicker, and adds a GLM-5.3-Flash version.
With setup's new settings, prose decodes at 41.2 tok/s and code at 54.5, against 40.2 and 53.2 for round 8's DFlash
and 39.7 and 49.9 for MTP, with the same texts.

- **Two rows a task in the drafter's product kernels.** With four rows, the Q4_K kernel spilled registers. A draft
  went from 6.50-6.57 to 5.43 ms (int8 weights, the whole head), with the same drafts.
- **A token subset for the draft head**, which setup now passes (`--dflash-vocab`):
  - `data/dflash_vocab.bin`: 163,778 tokens of English, code, Chinese, Japanese, Korean, Russian and multilingual chat
    text. The same acceptance as the whole head on the four English and code texts; 4% fewer accepted on Chinese
    and 1% on Russian. A draft takes 4.77-4.82 ms.
  - `data/dflash_vocab_en.bin`: 65,536 tokens of English and code, passed by hand only. 0-2% fewer accepted on
    English and code, 81-88% fewer on Chinese and Russian. A draft takes 4.03-4.09 ms.
- **The drafter's weights quantized in up to 32 threads at load.** A `dflash_check` run with Q4_K drafter weights
  took 59 s, now 12 s (int8: 9.5 s).
- **Q4_K drafter weights**, the default since the round (`GUILD_DFLASH_WEIGHTS=int8` keeps the int8 rows). With two
  rows a task they draft in 4.55 ms against 5.43. End to end they gained 1-4% on four of six prompts, and lost 0.2%
  on Russian and 3.2% on sql.
- **GLM-5.3-Flash** gets the drafter too, opt-in (`--dflash`, incoai's GLM-5.3-Flash DFlash 2 checkpoint). On this
  PC its NextN block stays the better drafter: GLM's verify takes at most two tokens, so DFlash's long blocks go
  unused, its draft costs 7 ms against 4.5, and its 1 GB of VRAM moves 35 shared experts to the CPU. It led on code
  by 1%; in the cleaner of two rounds it ran 2-6% slower on prose, story and sql.

PC: 2x Xeon Gold 6130 (16 cores each, AVX-512 BW without VNNI, 2 NUMA nodes), GTX 1070 8 GB (sm_61, 15 SMs, 151 W
cap), CUDA 12. Branch `feat/nemotron-transcribe`, guild-kernels-cpu `9b7281a`, guild-kernels-cuda `de65d60` (all
local). Drafter checkpoints: incoai's DFlash 2 checkpoints for Qwen3.8-Flash-Next and GLM-5.3-Flash (bf16
safetensors, 1.2 GB and 2.3 GB).

## Results

`tools/bench_runtime.py` with the server's settings: `--dense q6_k --hc q8_0`, `--context 32768`,
`--resident-budget-gib 56`, 2 runs x 256 tokens a prompt. One binary; the arms alternated over 2 rounds. DFlash runs
at `--spec 8` with the cumulative-probability rule (>= 0.1), MTP at `--spec 4 --spec-min-p 0.5`.

| Setting | prose tok/s | code tok/s | prose: tokens, ms a window (draft) | code: tokens, ms a window (draft) |
| --- | --- | --- | --- | --- |
| MTP | 39.82 / 39.67 | 50.00 / 49.75 | 2.23, 55.9-56.1 (5.7) | 2.98, 59.5-59.9 (6.2) |
| DFlash as round 8 left it: four rows a task, the whole head | 40.15 / 40.21 | 53.18 / 53.24 | 2.51, 62.3-62.7 (7.6) | 3.71, 69.6-70.0 (7.6) |
| two rows a task | 40.92 / 40.77 | 53.91 / 54.06 | 2.51, 61.2-61.7 (6.5) | 3.71, 68.6-69.1 (6.5) |
| two rows, `dflash_vocab.bin` (setup's default) | 41.26 / 41.18 | 54.60 / 54.48 | 2.51, 60.8-61.0 (5.8-5.9) | 3.71, 67.9-68.1 (5.9) |
| two rows, `dflash_vocab_en.bin` | 41.81 / 41.68 | 55.18 / 54.84 | 2.51, 60.0-60.4 (5.1) | 3.71, 67.2-67.8 (5.2) |
| two rows, `dflash_vocab.bin`, Q4_K drafter weights | 42.89 / 42.96 | 56.15 / 56.29 | 2.53, 59.0-59.2 (5.0) | 3.82, 67.9-68.1 (5.0) |

- **Output.** The same 256 tokens in every run, both prompts, all six settings.
- **Against round 8's DFlash**, setup's new default runs prose 2.6% and code 2.5% faster; against MTP, 3.7% and
  9.4%. Two rows a task alone gives 1.7% and 1.5%.
- **The draft in the engine** costs about 1 ms more than alone (`dflash_check`): the verifier's asynchronous commit
  runs beside it.

Four more prompts, the replay texts' (story, sql, Chinese, Russian), the same way (`--prompts`, token ids), the
arms alternated over 2 rounds:

| Setting | story | sql | Chinese | Russian |
| --- | --- | --- | --- | --- |
| two rows, the whole head | 38.15 / 37.99 | 59.44 / 59.24 | 30.42 / 30.25 | 32.24 / 31.60 |
| two rows, `dflash_vocab.bin` | 38.45 / 38.36 | 59.86 / 59.61 | 30.25 / 30.00 | 32.11 / 31.88 |
| two rows, `dflash_vocab_en.bin` | 38.65 / 38.56 | 60.62 / 60.51 | 21.76 / 21.58 | 22.20 / 22.07 |
| two rows, `dflash_vocab.bin`, Q4_K drafter weights | 38.81 / 38.75 | 57.90 / 57.77 | 30.85 / 30.65 | 32.07 / 31.82 |

- **Output.** The same 256 tokens in every run, all four prompts, all four settings.
- **The multilingual subset** runs within 1% of the whole head on Chinese (1.77 against 1.79 tokens a window, in
  0.5 ms less) and ahead of it elsewhere.
- **The English subset** costs Chinese 28% and Russian 31%: 1.15 and 1.11 tokens a window, against 1.77 and 1.80.
- **Q4_K drafter weights**, against int8 with the same subset, over all six prompts: prose +4.1%, code +3.1%, story
  +1.0%, Chinese +2.1%, Russian -0.2%, sql -3.2% (3.88 against 4.06 tokens a window).

## Two rows a task

`dflash_check` on the prose text (257 drafts), one binary with a switch for the rows a task, int8 drafter weights:

| `k4_mm` (the head, Q4_K, 248,320 rows, 7 columns) | 4 rows | 2 rows | 1 row |
| --- | ---: | ---: | ---: |
| ms a draft | 6.50-6.57 | 5.53 | 5.65-5.73 |

| `d8_mm` (the five layers' products, int8) | old rule (4 rows for 2048+ rows) | 2 rows | 1 row |
| --- | ---: | ---: | ---: |
| ms a draft (head at 2 rows) | 5.54-5.56 | 5.43 | 5.69 |

With Q4_K drafter weights every product goes through `k4_mm`: 6.56-6.58 ms a draft with four rows, 4.53-4.57 with
two.

**Why four rows lost.** `__launch_bounds__(256, 2)` caps a thread at 128 registers. Registers and spills
(`cuobjdump -res-usage`):

| `k4_mm_k<C, R>` | R = 4 | R = 2 | R = 1 |
| --- | --- | --- | --- |
| C = 1, 2 | 115, 127 | 72, 87 | 46, 56 |
| C = 3 .. 8 | 128 + 40-128 bytes of stack | 101-111 | 76-95 |

The kernel unpacks every row's nibbles once, outside the column loop. That takes registers per row, and at four rows
it spills from three columns on. Round 8 measured two rows a task before that unpack was hoisted (7.23 against 6.36
ms a draft), and kept four.

Isolated, on random rows (`dflash_kernels_test rates`: `d8_mm` 15360 x 2560, `k4_mm` 248,320 x 2560, the median
of 20 runs):

| Columns | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `d8_mm` GB/s | 211 | 210 | 209 | 207 | 205 | 204 | 204 | 203 |
| `k4_mm` GB/s, two rows a task | 214 | 210 | 208 | 205 | 203 | 201 | 198 | 194 |
| `k4_mm` GB/s, four rows a task (round 8) | 216 | 213 | 199 | 185 | 161 | 150 | 139 | 120 |

`k4_mm` is no longer compute-bound: at seven columns it reads the head at 198 GB/s, the int8 kernel's rate.

Odd row counts take one row a task, so a subset of any size stays on these kernels: before, an odd count made
`k4_mm` hand the head to MMQ.

## The head's token subset

The draft head is the target's output head (copied as Q4_K), so its cost scales with its rows: 7.6 µs per 1000 rows
at seven columns, against 11.3 with four rows a task. A token outside the subset can never be drafted; the verify
window still decides every token, so the output does not change.

**How the subsets were built.** `tools/draft_vocab.py` (new options: a share a file, `.jsonl` / `.parquet`
corpora, the `punct`, `caps` and `single` token classes, `--size`, `--multiple`, `--jobs`; the commands are in its
docstring). The corpus: up to 40 million characters from each of nine files:

| Kind of text | File (Hugging Face Hub, revision) | Share covered | Tokens outside the subset |
| --- | --- | ---: | ---: |
| English chat | HuggingFaceH4/ultrachat_200k, train_sft shard 0 (8049631) | 0.9995 | 0.018% |
| English encyclopedia | Salesforce/wikitext, wikitext-103-raw-v1 train shard 0 (b08601e) | 0.999 | 0.034% |
| Code | ise-uiuc/Magicoder-OSS-Instruct-75K (5f839b1) | 0.9995 | 0.016% |
| Code | m-a-p/CodeFeedback-Filtered-Instruction (a08c213) | 0.9995 | 0.006% |
| Multilingual chat | CohereForAI/aya_dataset (f9ea045) | 0.99 | 0.392% |
| Chinese encyclopedia | wikimedia/wikipedia 20231101.zh shard 2 (b04c8d1) | 0.99 | 0.440% |
| Japanese | 20231101.ja shard 14 | 0.98 | 0.202% |
| Korean | 20231101.ko shard 2 | 0.98 | 0.132% |
| Russian | 20231101.ru shard 7 | 0.99 | 0.225% |

To that union come every ASCII punctuation token (3,593), every all-caps ASCII word (6,709: SQL's keywords,
constants), the 256 one-byte tokens and the 33 control tokens, and one more frequent token for an even count:
163,778 tokens. The English subset keeps the same classes and fills to 65,536 with the four English and code files'
most frequent tokens.

**Acceptance.** Offline replays (`dflash_check`) of the target's greedy texts under the server's numerics
(`--dense q6_k --hc q8_0`), accepted drafts of 7, int8 drafter weights:

| Head | prose | code | story | sql | Chinese | Russian | ms a draft | VRAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| whole, 248,320 | 1.537 | 2.459 | 1.097 | 3.918 | 1.082 | 0.864 | 5.42-5.45 | 923 MiB |
| `dflash_vocab.bin`, 163,778 | 1.537 | 2.459 | 1.097 | 3.918 | 1.039 | 0.856 | 4.77-4.82 | 799 MiB |
| `dflash_vocab_en.bin`, 65,536 | 1.537 | 2.444 | 1.078 | 3.903 | 0.202 | 0.105 | 4.03-4.09 | 655 MiB |

The VRAM is the whole drafter's: its weights, the head and the context rings.

Tokens of the texts' generated parts outside the subset: none of the English and code texts' for the multilingual
subset (25 of the Chinese text's, 8 of the Russian's); 1-2 a text for the English one (315 and 337).

How the shares were chosen, from earlier subsets built with the same counts (a tenth of each file held out):

| Subset | Rows | sql | Chinese | Russian | ms a draft |
| --- | ---: | ---: | ---: | ---: | ---: |
| English and code by frequency | 65,536 | 3.767 | 0.307 | 0.125 | 4.04-4.08 |
| the shares above, no token classes | 159,402 | 3.770 | 1.039 | 0.856 | 4.75-4.78 |
| the same, with `punct` and `caps` | 162,672 | 3.903 | 1.039 | 0.856 | 4.77-4.82 |
| higher shares (0.9999 English and code, 0.995 the rest), with `punct` and `caps` | 181,048 | 3.918 | 1.078 | 0.864 | 4.91-4.96 |

Prose, code and story drafted the same with every subset but the first. The sql text needs its all-caps keywords
(`VARCHAR`, `PRIMARY`): without them it accepted 4% fewer drafts.

**Setup** passes the multilingual subset whatever `--draft-vocab` says: that option, and the low-VRAM plan's `en` on
an 8 GB-class card, now choose only the MTP layer's subset. The English subset is passed by hand. An engine without
`--dflash-vocab` drafts with the whole head.

## Q4_K drafter weights

The drafter's matrices as Q4_K: 684 MiB of VRAM against 923, and with two rows a task, a draft in 4.53-4.61 ms
against 5.42-5.45. The drafts change:

| Weights (whole head) | prose | code | story | sql | Chinese | Russian |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| int8 | 1.537 | 2.459 | 1.097 | 3.918 | 1.082 | 0.864 |
| Q4_K | 1.521 | 2.502 | 1.062 | 3.693 | 1.031 | 0.864 |

Four texts lose 1-6% of their accepted drafts, code gains 2%. End to end (Results), the cheaper draft outweighs that
on five of the six prompts; sql loses 3.2%. They are the default (`GUILD_DFLASH_WEIGHTS=int8` for the int8 rows).

## GLM-5.3-Flash

`--dflash <dir>` (or `GUILD_DFLASH`; `glm5_run --dflash`) drafts with incoai's GLM-5.3-Flash DFlash 2 checkpoint in
place of the NextN block, which is then not loaded. Setup does not fetch it: the checkpoint's licence is CC BY-NC-ND
4.0 (research and evaluation). The drafter reads GLM's layers 5, 14, 24, 33 and 42 (the four hyper-connection streams
averaged, as for Qwen); its head is GLM's own, 154,880 rows copied as Q4_K.

GLM's verify takes at most two tokens: the KDA layers keep one rollback point. So only the first draft of a block is
used, where NextN drafts that one token for less.

`glm5_run --spec --prompts`: four chat prompts with thinking closed, 256 tokens each from a reset context, a
262,144-token context, `GUILD_GLM_SHEXP_SPLIT=0` (the verify equals greedy decoding). Separate loads, alternated over
2 rounds, Q4_K drafter weights:

| Prompt | NextN: accepted, tok/s | DFlash: accepted, tok/s |
| --- | --- | --- |
| prose | 78.3%, 21.01 / 20.76 | 70.0%, 19.88 / 20.04 |
| code | 84.8%, 21.41 / 21.65 | 86.1%, 21.58 / 21.90 |
| story | 71.1%, 19.98 / 20.01 | 61.4%, 17.49 / 18.82 |
| sql | 92.5%, 22.31 / 22.48 | 90.3%, 20.71 / 21.99 |

- **A draft** takes 4.5 ms with NextN and 6.9-7.0 ms with DFlash, which took 10.5 ms before two rows a task.
- **VRAM.** The drafter takes 1017 MiB. The 262,144-token plan keeps 13,312 cells in VRAM, but 35 shared experts move
  to their CPU copies: the CPU's experts take 68-70 ms a step against 58-61 with NextN (one run of each slower).
  With int8 drafter weights (1558 MiB), the plan fell back to 5,376 VRAM-only cells and 42 shared experts on the CPU.
- **The texts.** Each arm's texts are the same in both rounds, and the DFlash arm's equal its first measurement's,
  before two rows a task. The two arms' texts part, at tokens 6 to 184, because the shared experts' CPU
  copies round differently. So the acceptance columns compare different texts; the tok/s columns are what each
  setup gives. With int8 drafter weights, the DFlash arm's text equals plain greedy decoding at the same placement for
  all four prompts (`glm5_run --plain`).
- **DFlash's second draft** was also right in 67 of 150 steps (prose), 86 of 137 (code), 54 of 158 (story) and 107 of
  134 (sql). A verify of three tokens could use them; that needs a second KDA rollback point.
- **DFlash's first round** lost story and sql time to a slow patch of the CPU's experts (75 against 68-69 ms a step);
  the second round is the cleaner one.

NextN stays GLM's drafter.

## Tried, not kept

| What | Result |
| --- | --- |
| Padding the 162,671- and 181,043-token subsets to a multiple of 8 for four rows a task | 5.53-5.56 and 5.74-5.81 ms a draft against 5.69-5.82 and 5.95-6.01: still behind the 159,402-token subset at two rows (4.84-4.92), which is what showed the four-row spill |

## Decided after the round

- **Q4_K drafter weights are the default**: 0.9 ms less a draft and 239 MiB less VRAM; end to end prose +4.1%, code
  +3.1%, story +1.0%, Chinese +2.1%, Russian -0.2%, sql -3.2%.
- **The multilingual head on every card.** The English head would save 144 MiB on an 8 GB card and make English and
  code 0.5-1.4% faster, but Chinese and Russian answers 28-31% slower.
- **`--dense q5_k --hc q8_0` for Unsloth's UD-IQ4_XS in setup** (`tools/bench_runtime.py` as in Results, today's
  drafter defaults: Q4_K weights, the multilingual head; alternated, 2 rounds of 2 runs; tok/s):

  | Dense matrices | prose | code | story | sql | Chinese | Russian |
  | --- | --- | --- | --- | --- | --- | --- |
  | the GGUF's Q8_0 (setup before) | 40.01 / 39.59 | 51.30 / 50.53 | 35.16 / 34.89 | 56.98 / 54.95 | 28.44 / 28.22 | 32.54 / 32.27 |
  | `--dense q6_k --hc q8_0` | 43.02 / 42.60 | 56.35 / 55.90 | 39.13 / 38.84 | 58.61 / 58.17 | 31.34 / 31.03 | 32.83 / 32.58 |
  | `--dense q5_k --hc q8_0` | 45.40 / 45.26 | 60.11 / 59.66 | 42.80 / 42.46 | 64.68 / 64.33 | 32.19 / 32.06 | 35.97 / 35.91 |

  Each setting's texts are the same in both rounds; the settings' texts differ from each other, since the numerics
  change (KL from the GGUF's output: 0.028 for q6_k, 0.048 for q5_k; rounds 5 and 6). A window takes 0.5-7% less time
  with q5_k than with q6_k; the rest of the difference is tokens a window on these texts. The GSQ-RCO quants keep
  their dense matrices at 3-6 bits in the GGUF, so setup writes the flags for UD-IQ4_XS only.
- **A longer GLM verify window** (a KDA rollback point a token) would let DFlash's later drafts count, and NextN's
  chained drafts too. Started after the round.

## Tools

- `dflash_kernels_test rates`: the two product kernels' GB/s at 1 to 8 columns on random rows.
- `tools/draft_vocab.py`: the subsets' commands are in its docstring.
- `dflash_check DRAFTER GGUF VOCAB PREFIX B [REF]` (round 8): acceptance and time a draft on a dumped text.
