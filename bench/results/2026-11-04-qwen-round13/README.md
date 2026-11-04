# Qwen3.8-Flash-Next, thirteenth round: the dense matrices' loss against their speed

Unsloth's UD-IQ4_XS stores every dense matrix as Q8_0; setup's `--dense q5_k` re-quantized them at load, without an
importance matrix, for fewer bytes a token. This round measures every choice of format against both its loss and its
speed, and moves setup to the best of them.

PC: 2x Xeon Gold 6130 (16 cores each, AVX-512 BW without VNNI, 2 NUMA nodes, 376 GiB RAM), GTX 1070 8 GB (sm_61,
15 SMs, 151 W cap, PCIe 3.0 x16), CUDA 12. Branch `feat/nemotron-transcribe`, the pinned kernel revisions
(guild-kernels-cpu `9b7281a`, guild-kernels-cuda `de65d60`).

## What changed

- **`--dense-imatrix <file>`** (`GUILD_DENSE_IMATRIX`): llama.cpp's GGUF importance matrix weights the
  re-quantization, as `llama-quantize --imatrix` does. Unsloth publishes the one its quants were made with:
  `imatrix_unsloth.gguf_file`, 580,038,720 bytes, SHA-256 `a5863123db1c...` at the GGUF repo's pinned revision. It
  lists every dense matrix (`<tensor>.in_sum2`, `.counts`; 45 chunks of 18,432 tokens).
- **`--dense-map kind=type,...`** (`GUILD_DENSE_MAP`): a type per matrix kind over `--dense`'s. Kinds: `attn_qkv`,
  `attn_gate`, `ssm_out`, `attn_q`, `attn_output`; types `q6_k`, `q5_k`, `q4_k`, or `q8_0` (as the GGUF has it).
- **`--prefill-mmq <off|dense|dense-hc>`**: round 12's `GUILD_PREFILL_DENSE_MMQ` as an option.
- **setup**, for UD-IQ4_XS on CUDA with an engine that has the options:
  - `--dense q4_k --dense-map ssm_out=q5_k,attn_output=q8_0 --dense-imatrix <its imatrix> --hc q8_0` (downloaded and
    checked like the DFlash drafter);
  - on Pascal cards also `--prefill-mmq dense`;
  - an older engine keeps `--dense q5_k --hc q8_0`.

## Loss

`qwen_kl` (decode's arithmetic: the verifier, 4 rows a window, teacher-forced) on round 5's two texts (2,084 and
2,173 predictions), against the GGUF's own dense matrices with `--hc q8_0`. KL mean, text 1 / text 2:

| Dense | No imatrix | Unsloth's imatrix |
| --- | --- | --- |
| q6_k | 0.031 / 0.045 | 0.027 / 0.044 |
| q5_k (setup's until now) | 0.053 / 0.077 | 0.044 / 0.069 |
| q4_k | 0.123 / 0.158 | 0.115 / 0.139 |

**The floor.** The pack's BF16 rounding of the hyper-connection matrices alone (`--hc bf16`, a 2^-9 relative change of
one kind of weight) measures 0.020 / 0.029. The model turns any small change of its numbers into about that much KL,
so nothing below ~0.02-0.03 is distinguishable from it.

**Which kinds carry the loss.** One kind at a time as Q4_K (with the imatrix), the rest the GGUF's Q8_0:

| Kind | Weights | KL text 1 / text 2 |
| --- | ---: | --- |
| attn_gate | 566 M | 0.023 / 0.036 |
| attn_q | 377 M | 0.023 / 0.036 |
| attn_qkv | 943 M | 0.038 / 0.054 |
| ssm_out | 566 M | 0.043 / 0.061 |
| attn_output | 126 M | 0.081 / 0.104 |

The QSA attention output, 5% of the bytes, costs more alone than every kind at Q5_K together.

## Loss against speed

**The speed measure.** Each setting's own cost is decode without drafts (`--decode-strategy normal`, prose and code,
256 tokens, 2 rounds alternated, one session per table block, rescaled by the C arm both sessions ran). With the
DFlash drafter on, every setting generates its own texts and those draft differently. Over six prompts x 2 runs x 2
rounds, the kept drafts a window were 2.65-2.75 for every setting, with no trend with the KL, so draft-free speed is
the comparison.

| Setting (all with the imatrix but the first) | Draft-free tok/s, round 1 / 2 | Speed | KL text 1 / text 2 |
| --- | ---: | ---: | --- |
| q5_k, no imatrix (setup's until now) | 29.71 / 29.79 | 1.000 | 0.053 / 0.077 |
| q5_k | the same | 1.000 | 0.044 / 0.069 |
| E: q4_k, attn_output q8_0 | 30.87 / 30.51 | 1.032 | 0.061 / 0.081 |
| **H: q4_k, ssm_out q5_k, attn_output q8_0 (setup's now)** | 29.88 / 30.04 | **1.014** | **0.047 / 0.066** |
| I: q5_k, attn_gate + attn_q q4_k, attn_output q6_k | 29.39 / 29.49 | 0.997 | 0.040 / 0.063 |
| C: q5_k, attn_gate + attn_q q4_k, attn_output q8_0 | 29.56 / 29.49 | 0.992 | 0.039 / 0.056 |
| A: q5_k, attn_output q8_0 | 29.16 / 29.44 | 0.985 | 0.031 / 0.053 |
| F: C with ssm_out q6_k | 28.92 / 28.91 | 0.979 | 0.034 / 0.052 |
| D: q6_k, attn_output q8_0 | 27.73 / 27.92 | 0.935 | 0.024 / 0.036 |
| q6_k | | | 0.027 / 0.044 |

(Prose and code averaged. H, I and F ran in a second session, where C measured 29.31 / 29.33; their tok/s are as
measured, the speed column scaled by C's 29.525 / 29.32.) Unsloth's imatrix
alone takes q5_k's loss down 11-16% at its speed.

**The frontier,** fastest first:
- **E** is the fastest, at about the old loss.
- **H** is faster than the old setting and has less loss than it: setup's choice.
- **C** and **A** trade 1-1.5% for 25-35% less loss than the old setting.
- **D** is within ~0.005 of the floor on text 1, at -7%.
- **F** is behind A on both counts.
- **q4_k** throughout (KL 0.115 / 0.139) drafted worse too (2.42 against 2.64 tokens a window on prose, 3.51
  against 3.88 on code) and was slower than q5_k with the drafter on.

## Setup's flags against the old ones

The new flags (`--dense q4_k --dense-map ssm_out=q5_k,attn_output=q8_0 --dense-imatrix ... --hc q8_0
--prefill-mmq dense`) against the old (`--dense q5_k --hc q8_0`), with setup's other settings (DFlash, `--prefill
auto`).

**Decode with drafts.** Six prompts, 2 runs x 256 tokens, 2 rounds alternated, tok/s (mean of the 4 runs):

| Flags | prose | code | story | sql | Chinese | Russian | Geometric mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| old | 45.1 | 59.9 | 42.7 | 64.4 | 32.1 | 35.9 | 45.2 |
| new | 49.0 | 61.5 | 42.3 | 60.1 | 35.4 | 36.7 | 46.4 |

The prompts move apart by up to 10% either way because the texts differ; the draft-free cost above is the
measure (+1.4%).

**Prompts,** no prefix cache, one run each, alternated:

| Prompt | Old | New | Decode after it, old -> new |
| --- | ---: | ---: | --- |
| 4,008 tokens | 19.70 s | 16.04 s | 46.6 -> 48.9 tok/s |
| 31,918 tokens | 171.4 s | 142.0 s | 40.1 -> 43.3 tok/s |

**The prompt path's loss** with the new matrices (`QWEN_KL_PREFIX=1024`: positions 1,024 on, against the verifier
reading the whole text):

| Prompt path | KL text 1 / text 2 | Top-1 the same |
| --- | --- | --- |
| cuBLAS | 0.0124 / 0.0138 | 97.1 / 97.0% |
| MMQ (`--prefill-mmq dense`) | 0.0147 / 0.0158 | 95.9 / 97.0% |

Both are under the floor (0.020 / 0.029).

## Files

`runs/`: the KL outputs (`*.out`), the decode reports (`b-pl*.json`, `b-six*.json`, `b-fin*.json`) and the scripts
(`base.sh`, `im.sh`, `grp.sh`, `map.sh`, `map2.sh`, `floor2.sh`, `plain.sh`, `plain2.sh`, `six*.sh`, `fin.sh`,
`klfin2.sh`, `agg.py`). The KL runs' full distributions (~0.5-1 GB each) are not kept.
