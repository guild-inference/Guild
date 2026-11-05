# Qwen3.8-Flash-Next, fourteenth round: short prompts through the verifier

PC: 2x Xeon Gold 6130, GTX 1070 8 GB (PCIe 3.0 x16), CUDA 12. Setup's settings for Unsloth UD-IQ4_XS (round 13: the
DFlash 2 drafter, `--dense q4_k --dense-map ssm_out=q5_k,attn_output=q8_0 --dense-imatrix ... --hc q8_0
--prefill-mmq dense`), `--context 32768`, FP16 KV in host memory, no prefix cache.

## Short prompts

A prompt tail goes either through the prompt path (batched, the experts streamed to the GPU) or through the
verifier's windows (decode's path, the experts on the CPU). The bound was 64 tokens, set when the two crossed near 110
on this PC (bench/results/2026-10-24-split-window) with a GPU expert cache.

Without a cache, as an 8 GB card runs this model, every chunk of the prompt path streams nearly all of the experts
(a 100-token chunk already routes to most of each layer's 512). Prompts of the first n tokens of the bench's 4K
prompt, plus its last 8, one run each, prompt seconds:

| Tokens | Prompt path | Verifier (`GUILD_PREFILL_VERIFY_MAX=100000`) | Now (default) |
| ---: | ---: | ---: | ---: |
| 100 | 3.17 | 0.95 | 0.95 |
| 200 | 4.20 | 1.88 | 1.89 |
| 400 | 5.51 | 3.78 | 3.79 |
| 800 | 6.81 | 7.63 | 6.79 |
| 1,600 | 9.02 | 15.51 | 8.99 |

- **The two paths.** The verifier costs ~9.5 ms a token (windows of 8 with the DFlash drafter). The prompt path costs
  3-4 s of streaming plus ~3.3 ms a token.
- **The crossing** is near 680 tokens.
- **The bound** is now 512 when the GPU holds no experts, and 64 as before with a cache. `GUILD_PREFILL_VERIFY_MAX`
  still sets it.

An agentic turn's new message or tool result of a few hundred tokens after a cached prefix waits 1-2 s less. The
verifier's arithmetic is decode's, the reference of rounds 12-13's KL measurements.

## Long prompts: where the time goes

**A 31,917-token prompt** with `GUILD_PREFILL_TIMING=1` (`runs/p32t.timing`) took 138 s:
- the experts' gate/up products 25.5 s, their down products 10.6 s;
- waiting for expert copies 23.9 s;
- QSA attention 17.0 s, the hyper-connection read 15.2 s, the GDN input projections 11.4 s;
- the gather of expert blobs into MMQ's group buffers 5.2 s.

**The expert time depends on the chunk.** The 4K prompt at chunks of 1,024 / 2,048 / 4,096 tokens (`runs/chk.progress`),
i.e. about 20 / 40 / 78 tokens an expert:

| Chunk | Gate/up | Down | Gather | Waiting for copies | Prompt |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,024 | 6.36 s | 2.19 s | 2.06 s | 12.0 s | 30.8 s |
| 2,048 | 4.56 s | 1.64 s | 1.16 s | 5.8 s | 21.2 s |
| 4,096 | 3.09 s | 1.30 s | 0.62 s | 2.9 s | 16.0 s |

Each doubling of an expert's tokens takes its products to ~0.68x, and the copies and the gather are paid per chunk.
Every 4,096-token chunk of a long prompt streams all of the experts again (~50 GB at the link's 8 GB/s).
- **Layer-major (not built).** Each layer's experts streamed once for the whole prompt, ~620 tokens an expert at
  32K, estimated at ~30% off a 32K prompt. It needs a restructuring of the prompt path's chunk loop.

## Larger chunks, the same tokens

The chunk was held at 4,096 tokens by ~0.40 MiB of buffers a token. Three changes cut that:
- **Group-local MoE buffers** (MMQ on every layer, not fused, one GPU; `GUILD_PREFILL_GROUP_LOCAL=0` keeps the old
  ones). The MoE half's gate/up rows, SwiGLU output and q8_1 activations hold one MMQ group's rows, at most 2T (a
  larger group runs in parts), not all T x 10 slots'. The same products of the same rows land in the same rows of
  `Dm`, which keeps every slot (the combine's fixed `fmaf` order needs them all).
- **`gated` in the shared region.** The hyper-connection read's up product lives in the region the half's scratch
  takes after the mix has read it.
- **Dense MMQ in slices.** The dense MMQ path runs in slices of 1,024 rows: each row's product is its own.

**Chunk sizing.** The chunk now grows to the largest, up to 8,192, whose `Prefill::bytes_needed` (the prompt path's
own count of its buffers) leaves 768 MiB, and 48 more for the dense slices. Before, it was 168 B x n_embd a token,
up to 4,096. On Pascal the rows of `--prefill-rows 512` keep a chunk's results those of its 512-token parts, so a
larger chunk gives the same tokens.

| | Before | After |
| --- | --- | --- |
| 4K prompt's VRAM after the run, at chunks of 4,096 | 7,331 MiB | 6,965 MiB |
| 4K prompt, 4,007 tokens, one chunk either way | 15.97 s | 16.07 s |
| 31,917 tokens, `--context 32768` (chunks of 8,192) | 141.99 s | 116.55 s |
| 31,917 tokens, `--context 262144` (2,692 MiB free: chunks of 6,656) | 137.06 s | 121.73 s |

The tokens were the same in every row. At 32K the experts' gate/up products took 25.5 -> 19.8 s, down 10.6 -> 8.7,
the gather 5.2 -> 2.7 and the wait for copies 23.9 -> 12.5. Decode after the 32K prompt was 43.3 -> 45.0 tok/s; about
440-465 MiB of VRAM is left free after these runs.
