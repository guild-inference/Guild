# GLM-5.3-Flash: its attention's formats, loss against speed, with Unsloth's imatrix

Round 13 measured Qwen's dense formats; this applies the same measurement to GLM-5.3-Flash.
- **What gets re-quantized.** GLM keeps its attention (KDA `attn_q/k/v`, DSA `attn_q_b`, every layer's
  `attn_output`) and its shared experts' GPU copies on the GPU. The load re-quantizes them from the GGUF's Q8_0:
  Q6_K by default when Q8_0 would leave less context than asked, Q5_K / Q4_K with `--dense`.
- **What is new.** `--dense-imatrix` and `--dense-map` now reach that re-quantization too
  (`guild/core/dense_quant.hpp`). Unsloth publishes GLM-5.3-Flash's importance matrix: `imatrix_unsloth.gguf_file`,
  512,687,616 bytes, SHA-256 `27d2ef6b6be0...`, repo revision `a38483c8`; 88 chunks of 9,216 tokens. It lists every
  matrix the load re-quantizes.

PC: 2x Xeon Gold 6130, GTX 1070 8 GB, CUDA 12; GLM-5.3-Flash UD-IQ4_XS (Unsloth); branch `perf/dense-quant-glm`.

## Loss

`glm5_kl` (decode's kernels and placement, verify windows of two tokens, teacher-forced) on round 6's two texts (1,973
and 2,009 predictions), against the GGUF's own Q8_0 attention. Round 6 measured ~0.023 / 0.025 for a change of
rounding alone. KL mean, text 1 / text 2:

| Attention | Imatrix | KL | VRAM of the attention |
| --- | --- | --- | ---: |
| Q6_K (the default at long contexts) | no | 0.027 / 0.029 | 4,436 MiB |
| Q6_K | yes | 0.027 / | 4,436 MiB |
| Q5_K | yes | 0.033 / 0.039 | 3,718 MiB |
| Q5_K, attn_output Q6_K | yes | 0.031 / 0.037 | 3,964 MiB |
| Q4_K | yes | 0.054 / 0.060 | 3,042 MiB |
| Q4_K, attn_output Q5_K | yes | 0.043 / | 3,274 MiB |
| Q4_K, attn_output Q6_K | yes | 0.040 / 0.048 | 3,520 MiB |
| Q4_K, attn_output Q8_0 | yes | 0.040 / 0.048 | 3,970 MiB |

At Q6_K the model is at its floor, so the imatrix cannot show there. As in Qwen, the attention output carries an
outsized share of the loss: keeping it at Q6_K takes Q4_K's 0.054 / 0.060 to 0.040 / 0.048, and Q8_0 adds nothing
more.

## Speed

`glm5_run --spec --prompts` (the four chat prompts of round 9, 256 tokens each, NextN drafts, a 262,144-token
context), separate loads, 2 rounds alternated. Each format generates its own texts, so the step median is the
comparison:

| Attention | Step median, round 1 / 2 | tok/s (mean of the 4 prompts) | Context cells in VRAM |
| --- | --- | --- | ---: |
| Q6_K | 79.6 / 79.5 ms | 22.2 / 22.3 | 11,776 |
| Q5_K | 73.4 / 73.2 | 24.1 / 24.2 | 74,752 |
| Q5_K, attn_output Q6_K | 76.0 / 75.8 | 23.4 / 23.5 | 53,760 |
| Q4_K | 67.9 / 68.7 | 26.0 / 25.7 | 143,360 |
| Q4_K, attn_output Q6_K | 71.8 / 71.5 | 25.3 / 25.4 | 102,656 |
| Q4_K, attn_output Q8_0 | 75.0 / 75.3 | 23.8 / 23.7 | 64,256 |

Fewer attention bytes are fewer to read a step, and they leave room for more of the context's cells in VRAM.

**The frontier,** fastest first:
- **`--dense q4_k`** (with the imatrix): −14%, at about twice the floor.
- **`--dense q4_k --dense-map attn_output=q6_k`:** −10%, at 0.040 / 0.048.
- **`--dense q5_k`:** −8%, at 0.033 / 0.039.
- **Q6_K, the default:** at the floor.

Q5_K with a Q6_K output and Q4_K with a Q8_0 output are behind those on both counts. GLM's server config is the
user's, and setup does not install GLM, so no default changes here. The README lists the options.

## Files

`runs/`: the scripts (`gkl.sh`, `gsp.sh`), the KL outputs (`*_1.out`, `*_2.out`, `base*.out`) and the progress files.
`GUILD_GLM_KEEP_CACHE=1` (new) kept the routed experts' pages in the page cache between the loads.
