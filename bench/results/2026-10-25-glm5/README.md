# GLM-5.3-Flash on the dual Xeon + GTX 1070

Measured 2026-10-25 on this PC: Ubuntu, two Xeon Gold 6130 (2 NUMA nodes, 32 cores), 376 GiB RAM, GTX 1070 8 GiB
(sm_61), driver 580.178.04, CUDA 12 Release build of `glm5_run` and `guild-generate` on branch `feat/glm5` at
4dda201 (`-DGUILD_EXPERIMENTAL_SM60=ON`; `guild-kernels-cuda` at fbd2304 on `perf/sm61-decode`, `guild-kernels-cpu`
at 629817c). The model is [unsloth/GLM-5.3-Flash-GGUF](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF), UD-IQ4_XS
(5 shards, 147 GiB). The reference is llama.cpp at 86ebfef (the glm5next branch), on the CPU.

## What GLM-5.3-Flash is, and how Guild runs it

45 layers: 34 KDA (gated linear attention with a 128 x 128 state per head) and 11 DSA (MLA attention with the latent
cached, a lightning indexer keeping 2048 of the past positions, in pools of 4, once a query sees more), four residual
streams mixed by mHC, a NextN layer for drafting. Layers 0-2 have a dense FFN (12288 wide); the rest have 288 routed
experts (8 used) and a shared expert. Hidden size 4096, vocabulary 154,880.

| Part | Where it runs | Format |
| --- | --- | --- |
| attention, mHC, routers, head | GPU, `src/core/glm5.cu` and the dense products | Q8_0 (as the GGUF), or Q6_K re-quantized at load; head Q6_K |
| routed experts | CPU, half of every expert's rows in each NUMA node's memory | gate/up IQ3_S repacked as NL4, down IQ4_XS (layer 44: Q6_K) |
| shared experts, dense FFNs | GPU when the VRAM holds them, else CPU | Q8_0 |

`GUILD_GLM_DENSE=auto` (the default) re-quantizes the attention's largest Q8_0 matrices to Q6_K when Q8_0 would
leave less context than asked for (up to 16k tokens). On this card that is the difference between 6400 and 17664
tokens of context, and 0 or 43 of the 46 FFNs on the GPU. The re-quantization adds about 16 s to the load. Loading
takes about 65 s from the page cache (the experts are copied into 77.4 + 77.4 GiB of node-local memory).

## Checks against llama.cpp

The last prompt token's logits, Guild against llama.cpp on the same GGUF (KL in nats, of llama.cpp's distribution
to Guild's):

| Prompt | Tokens | Q8_0 | Q6_K |
| --- | ---: | --- | --- |
| "The capital of France is" | 5 | 0.013 | 0.020 |
| a chat turn | 23 | 0.101, top-1 differs | 0.021 |
| a passage | 640 | 0.037 | 0.116 |
| a long passage | 2967 | 0.083 | 0.024 |

Top-1 agrees on every row but the Q8_0 chat turn, where the two leading tokens (`</think>` and a word) are 0.49
apart in llama.cpp's own logits. These numbers move by 0.02 to 0.07 from one build to the next on rounding-level
changes (Guild's MMQ against its MMVQ path alone, same weights: 0.079), so a single row says little about which
build is closer.

Decode steps, which a single prompt row does not cover: Guild's greedy output (`glm5_run --dump`, the one-token
path) replayed through llama.cpp in one batch, KL at every step:

| Prompt | Steps | Q8_0 KL mean / median, top-1 | Q6_K KL mean / median, top-1 |
| --- | ---: | --- | --- |
| "The capital of France is" | 64 | 0.018 / 0.004, 92.2% | 0.012 / 0.001, 95.3% |
| a chat turn (the reply ends) | 10 / 12 | 0.001 / 0.000, 100% | 0.001 / 0.000, 100% |
| a passage of 640 tokens | 64 | 0.026 / 0.016, 93.8% | 0.027 / 0.017, 95.3% |

The Q6_K re-quantization costs nothing these numbers can see.

On a tiny random GGUF of the same architecture (`tools/glm5_tiny_gguf.py`, all Q8_0) Guild's last-token logits
are within KL 0.001 to 0.01 of llama.cpp's, and speculative decoding writes the same tokens as plain greedy
decoding.

## Speed

`glm5_run`, the 5-token prompt, 256 new tokens, greedy. A step is one token, or with `--spec` a draft from the
NextN layer and a verify window of two:

| Dense matrices | Context | FFNs on the GPU | Plain | Per step (CPU experts) | `--spec` | Drafts accepted | Per step |
| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| Q8_0 (`--ctx 6400 --q8`) | 6400 | 0 of 46 | 11.70 tok/s | 84.8 ms (40.9) | 15.51 tok/s | 91% | 122.1 ms |
| Q6_K (`--ctx 32768`, auto) | 17664 | 43 of 46 | 13.76 tok/s | 72.6 ms (34.3) | 16.93 tok/s | 87% | 109.0 ms |

Through the server (`serve/server.py` with `guild-generate`, Q6_K, context 16384), short chat replies decoded at
17.3 to 17.4 tok/s with 93% to 96% of the drafts accepted.

Prompts (`glm5_run`, the same four prompts as above; windows of 128 tokens):

| Prompt | Tokens | Q8_0 | Q6_K |
| --- | ---: | ---: | ---: |
| "The capital of France is" | 5 | 21.0 tok/s | 23.3 tok/s |
| a chat turn | 23 | 32.0 tok/s | 35.9 tok/s |
| a passage | 640 | 41.4 tok/s | 45.2 tok/s |
| a long passage | 2967 | 36.2 tok/s | 39.8 tok/s |

A conversation reads its history once: the snapshot for the next turn is taken after the prompt's last
`<|assistant|>` (the recurrent states cannot be rewound). Three chat turns over a 1192-token system prompt, through
the server: the first prompt took 28.9 s, the second and third reused 1191 and 1236 tokens and took 1.4 s and 0.9 s.

## Where the time goes

A Q6_K decode step, 72.6 ms: the GPU's part about 38 ms, the CPU experts' 34 ms, one after the other (each layer's
attention needs the last layer's FFN). The shared experts run on the GPU while the CPU computes the routed ones.

The CPU experts, per token (`GUILD_GLM_TIMING=1`): gate/up 19.0 ms for 3.27 GB of rows (172 GB/s), down 12.3 ms for
1.66 GB (135 GB/s), the activations 1.1 ms. The machine reads 226 GB/s with 16 threads a node (node-local), so the
phases reach 60% to 76% of it; with every core busy the cores run at 2.5 to 2.7 GHz (3.7 GHz on one core).

The GPU's products, one token, GB/s of weights (`glm5_kernels_test bench`); before is native_mmvq, but for attn_k_b,
whose earlier kernel was GLM's own with a warp a row:

| Matrix | Before | Now |
| --- | ---: | ---: |
| Q6_K 4096 x 8192 (KDA q/k/v) | 146 | 208 |
| Q6_K 8192 x 4096 (KDA output) | 146 | 209 |
| Q6_K 16384 x 4096 (DSA output) | 143 | 213 |
| Q6_K 4096 x 154880 (head) | 152 | 219 |
| Q8_0 4096 x 8192 | 181 | 191 |
| Q8_0 16384 x 4096 | 188 | 210 |
| Q8_0 128 x 8192 (KDA gates' second half) | 48 | 166 |
| Q8_0 per head, 64 x [512][256] (attn_k_b) | 46 | 167 |

The staged Q6_K / Q8_0 kernels give bitwise native_mmvq's results; they load a warp's row through shared memory
with 16-byte loads, since both formats' blocks are only 2-byte aligned.

## Over the day

Decode with the Q8_0 matrices and a 6400-token context went from 9.0 tok/s (NL4 experts and node-local arenas,
before the GPU kernels above) to 11.7; with the Q6_K choice, 13.8; with the NextN drafts, 16.9. The 640-token prompt
went from 24.9 tok/s (Q6_K, before Q6_K MMQ and the pinned calling thread) to 45.2.

## How to reproduce

```bash
cmake -S . -B build-cuda12 -DGUILD_ENABLE_CUDA=ON -DGUILD_EXPERIMENTAL_SM60=ON -DCMAKE_CUDA_ARCHITECTURES=61
cmake --build build-cuda12 -j --target glm5_run glm5_kernels_test guild-generate
M=GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf
build-cuda12/glm5_run $M 785,6722,315,9621,374 256 --ctx 32768 [--spec] [--q8 --ctx 6400]
build-cuda12/glm5_kernels_test && build-cuda12/glm5_kernels_test bench
```

`glm5_run` prints the placement, the per-step percentiles and the generated ids. `--logits <file>` writes the last
prompt token's logits, `--dump <file>` every plain decode step's.
