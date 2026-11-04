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
- **The fix: layer-major.** Each layer's experts streamed once for the whole prompt, ~620 tokens an expert at 32K.
  Estimated at ~30% off a 32K prompt; not built here (a restructuring of the prompt path's chunk loop).
