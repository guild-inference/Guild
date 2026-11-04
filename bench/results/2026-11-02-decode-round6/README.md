# Decode, sixth round: GLM-5.3-Flash's GPU half in fewer bits, and its layers kept even

Round 5 read fewer bytes on Qwen3.8-Flash-Next. This round does it on GLM-5.3-Flash: `--dense q5_k` and
`--dense q4_k` re-quantize its attention matrices and its shared experts' GPU copies at load, a new Q5_K kernel reads
them at the card's bandwidth, and two placement rules keep a layer's CPU and GPU halves even once the GPU's half is
the shorter one. Without those rules the fewer bytes saved almost nothing. Qwen's dense matrices measured too far
from today's output at Q5_K and Q4_K to pursue.

PC: 2x Xeon Gold 6130 (16 cores each, AVX-512 BW without VNNI, 2 NUMA nodes), GTX 1070 8 GB (sm_61, 15 SMs, 151 W
cap), CUDA 12. Branch `feat/nemotron-transcribe` (`82af319`), guild-kernels-cpu `9b7281a`, guild-kernels-cuda
`de65d60` (all local).

## GLM-5.3-Flash

`glm5_run --spec` from the 5-token prompt, 768 tokens (`GLM5_RUN_IGNORE_EOS=1`: every arm decodes as many), at the
server's `--ctx 262144`, arms in separate processes, alternated, two runs each. The arms' texts differ, and with them
how many drafts are accepted, so the median step is the column to compare.

| Setting | median step (ms) | tok/s | drafts accepted | KL mean (text 1 / text 2) | top-1 (text 1 / text 2) |
| --- | ---: | ---: | ---: | ---: | ---: |
| today (Q6_K attention, `Glm5Dense::Auto`) | 81.3 / 81.0 | 23.90 / 24.01 | 95% | - | - |
| `--dense q5_k` | 75.6 / 75.6 | 24.65 / 24.59 | 86% | 0.036 / 0.043 | 91.8% / 91.6% |
| `--dense q4_k` | 73.4 / 70.7 | 27.36 / 28.01 | 99% | 0.052 / 0.063 | 90.0% / 90.2% |
| noise floor (see below) | | | | 0.023 / 0.025 | 92.7% / 93.8% |

Quality is the new `tests/core/glm5_kl.cpp`: a text teacher-forced through verify windows of two tokens, as decode
runs a window whose draft is accepted, every position's next-token distribution compared with today's: KL(today ||
this), and how often the top token is the same. Text 1 is the first 7.5 KB of
`docs/architecture/EXECUTION_LIFECYCLE.md` (1,973 predictions), text 2 8 KB of `src/runtime/session.cpp` (2,009).
The noise floor is today's weights with a window's second row taking its shared expert on the GPU instead of its CPU
copy (`GUILD_GLM_SHEXP_SPLIT=0`): a change of rounding only. Perplexity of the texts: today 5.566 / 4.906, `q5_k`
5.583 / 4.882, `q4_k` 5.626 / 4.879, the noise floor 5.545 / 4.910. For scale: the GGUF's own Q8_0 attention measured
KL 0.026 and top-1 93.5% against today's Q6_K on text 1.

`--dense q5_k`'s distance from today is 1.6-1.7 times the noise floor's, as `--dense q6_k`'s was on Qwen in round 5
(0.028 against 0.0167); `--dense q4_k`'s is 2.3-2.5 times. Neither is a default: the server keeps today's Q6_K
unless `--dense` is given.

VRAM at 262,144 tokens of context: today 4,436 MiB of attention and 12,544 cells' latent rows in VRAM; `q5_k` 3,718
MiB and 107,008 cells; `q4_k` 3,042 MiB and 175,616 cells. The 43 shared experts (the NextN block's included) take
1,097 MiB as the GGUF's Q8_0, 710 as Q5_K, 581 as Q4_K.

### The changes

- **`--dense q5_k` / `q4_k` for GLM-5.3-Flash.** The attention's Q8_0 matrices (KDA q, k, v, o; DSA q_b, o; the
  NextN block's) are dequantized exactly and quantized as llama-quantize would without an importance matrix, as
  Q6_K already was. The shared experts' GPU copies take the same format (`GUILD_GLM_SHEXP=q8_0 / q6_k / q5_k / q4_k`
  picks theirs); a CPU copy of one still reads the GGUF's rows. On the shared experts it costs little: with Q4_K
  attention, text 1's KL 0.048 with them as Q8_0, 0.050 as Q5_K, 0.052 as Q4_K. The VRAM plan counts the smaller
  copies.
- **A Q5_K kernel** (`glm5k::q5k_mmvq`): native_mmvq's own Q5_K kernel reads each int with a load, 163 GB/s for KDA
  q|k|v (4096 x 24576) where Q6_K's staged kernel reaches 213. This one stages a warp's row eight blocks at a time
  with 16-byte loads, in native_mmvq's order, so every output is bitwise native_mmvq's (7 shapes at one and two
  tokens, `glm5_kernels_test`): 202-216 GB/s at one token (KDA q|k|v 387 -> 323 us against Q6_K, KDA out 133 ->
  112, DSA q_b 102 -> 85, DSA out 261 -> 216). Q4_K takes native_mmvq's staged kernel, 209 GB/s.
- **The dense layers' FFNs stay on the CPU** where they used to move to the GPU when VRAM allowed. Layers 0-2 have no
  routed experts: with their FFN on the GPU for both groups, the layer's attention and FFN all wait on the GPU and the
  CPU has nothing to do. Where the shared experts' split is the default (below) they may go to the GPU as twins
  instead: the first group's on the GPU, the second's on the CPU. Today's server configuration places them on the
  CPU either way (no room), so its output is unchanged: the same 768 ids before and after.
- **The shared experts' split follows the attention's format.** With Q8_0 or Q6_K attention a window's second group
  keeps computing its shared expert on the CPU (the GPU's half of a layer is the longer one); with Q5_K or Q4_K both
  groups compute it on the GPU, which then has the time. That also makes a verify window's rows what one-token decode
  computes, bit for bit, so `--spec` gives plain greedy decoding's text. `GUILD_GLM_SHEXP_SPLIT=0 / 1` overrides it.
- **Prompt windows** multiply Q4_K and Q5_K matrices with llama.cpp's MMQ, built for CUDA by default now
  (`GUILD_MMQ_K45`; Q6_K's was already), on activations quantized in the layout those formats take (each 32 values'
  sum beside the scale; Q8_0's and Q6_K's have none: with theirs the products came out wrong and the CPU experts
  were handed invalid expert ids). A 300-token prompt's top five tokens are the same windowed and one token at a
  time. A 1,974-token prompt (text 1) at 262,144 tokens of context: 70.2 tok/s today, 74.4 with `--dense q5_k` and
  with `q4_k` (the CPU's experts take 27.5 of its 28.1 s).

### Why the bytes did not pay at first

| Setting (step medians, ms) | runs |
| --- | ---: |
| Q5_K attention: the freed VRAM put layers 0-2's dense FFNs on the GPU | 80.9 / 82.3 |
| the same, the dense FFNs on the CPU | 79.3 / 78.3 |
| Q4_K attention, dense FFNs on the GPU | 80.8 / 78.5 |
| the same, the dense FFNs on the CPU | 76.4 / 78.4 |
| the same, both groups' shared experts on the GPU (one process: 78.4 split -> 74.5) | 74.5 |
| the same, the shared experts as Q5_K / Q4_K | 71.5 / 72.3, 70.4 / 71.6 |

A pipelined step's layers alternate the GPU's attention for one group with the CPU's experts for the other, so a
layer takes about max(GPU half, CPU half) twice, and time saved on a half that was already the shorter one is not
saved. With Q5_K or Q4_K the KDA layers stopped waiting for the GPU at all (the CPU's waits for a router 1,209 ->
102-111 ms over a run) while the dense layers' GPU FFNs made layer 3 wait (757 -> 3,414-3,529 ms). Moving the dense
FFNs back to the CPU, then the second group's shared experts to the GPU, then the shared experts to fewer bits put
the halves back in step.

### Where the time goes now

nsys, `--dense q4_k` (72.0 ms a step under the profiler, 22 steps): the GPU is busy 63.9 ms of it.

- **Q4_K attention products: 31.7 ms** (KDA q|k|v 18.7, KDA out 7.1, DSA o 4.2, DSA q_b 1.7). Today's Q6_K takes
  44.2 ms for them (nsys of today's configuration: 81.6 ms a step, the GPU busy 75.0).
- **The shared experts, both groups, as Q4_K: 6.8 ms** (11.2 as Q8_0 with Q4_K attention; today the first group's
  alone, 5.7).
- **The output head: 3.1 ms** for verify (two rows, Q6_K, 168 GB/s) and 2.4 for the draft.
- **The rest: 19.9 ms** of smaller kernels: KDA's recurrence 3.0, DSA's per-head products 2.3, the hyper-connection
  pre-mixes and post-mixes 3.5, the narrow Q8_0 rows 1.5, the routers and routing 2.2, quantization 1.1, ...
- **GPU idle 8.0 ms**: 2.1 waiting for the CPU's experts, 1.5 in layers 0-2 (their FFN on the CPU), 1.1 between
  verify and the next draft (the host's argmax and accept).

The CPU's experts take 57-61 ms of a step at 176-182 GB/s, so the two halves are again within a few ms of each other.

## Qwen3.8-Flash-Next

- **`--dense q5_k` / `q4_k`** (now accepted by the option, for both models), with `--hc q8_0`, against today's output
  (`qwen_kl`, text 1): KL 0.048 / 0.118, top-1 92.0% / 87.7%, perplexity 5.063 / 5.172 (today 5.082). `--dense
  q6_k --hc q8_0` measured 0.028 and 93.5% in round 5, against a noise floor of 0.0167. Q5_K is 2.9 times the noise
  floor, and its kernel is slower than Q6_K's at three columns (below): not pursued.
- **native_mmvq stages Q5_K rows** (guild-kernels-cuda), as Q4_K's: bitwise its own kernel (`mmvq_staged_parity`,
  739,620 outputs, 0 differ, Q5_K cases added). One column 163 -> 212 GB/s (GLM's KDA q|k|v); on Qwen's shapes
  (2560 to 6144 wide) 185-204 GB/s at one column, 151-169 at two, 123-150 at three, where Q6_K's are 173-206,
  159-192 and 122-168.

## Tried, not kept

| What | Result |
| --- | --- |
| Qwen: windows of fewer than k tokens as one row group (`GUILD_SPEC_SPLIT_MIN=k`, `--spec 4`, `--dense q6_k --hc q8_0`) | k = 2 (today) / 3 / 4: 55.5-55.8 / 55.4-55.8 / 56.0 ms a window (prose), 59.0-59.4 / 58.9-59.0 / 59.2 (code), the same texts; never splitting 61.5 / 67.8 |
| GLM: Q5_K attention with both groups' shared experts on the GPU, the shared experts as Q8_0 | 80.2 -> 79.5 ms (one process): the GPU had little time to spare |
| GLM: the dense layers' FFNs on the GPU for both groups once VRAM allows | 0.1 to 4.4 ms a step slower than on the CPU, run for run (table above) |

## What is left

- **GLM's halves are even again**: a faster step needs both the CPU's experts (57-61 ms a step at 176-182 GB/s) and
  the GPU's work to drop.
- **The draft: 5.1 ms a step**, half of it the output head's 154,880 rows as Q6_K. A draft-only Q4_K copy (357 MB of
  the VRAM `--dense q4_k` freed) or the token subset (`data/draft_vocab_glm.bin`) changes only the drafts.
- **The verify head at two rows: 3.1 ms** at 168 GB/s.
