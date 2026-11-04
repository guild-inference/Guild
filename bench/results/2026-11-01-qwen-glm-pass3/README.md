# Qwen's prompts a third faster with the same bits, and GLM-5.3-Flash's CPU experts

A third pass on Qwen3.8-Flash-Next and GLM-5.3-Flash, outputs unchanged throughout:
- **Qwen:** a 4,007-token prompt takes 35.6 s instead of 53.8 s with `--prefill 1024 --prefill-rows 512`, and gives
  the same 256 tokens after it.
- **GLM:** its CPU expert rows got a little faster: a 2,051-token prompt ~33.2 -> ~30.9 s over the two changes
  below (each measured against the one before it), decode ~1%.

The GLM long-context work of the same day is in [../2026-11-01-glm-256k](../2026-11-01-glm-256k/README.md).

PC: 2x Xeon Gold 6130 (16 cores each, AVX-512 BW without VNNI, 2 NUMA nodes), GTX 1070 8 GB on PCIe 3.0 x16,
CUDA 12. Branch `feat/nemotron-transcribe`, built against guild-kernels-cpu `perf/q8-avx512` (`9b7281a`, local) and
guild-kernels-cuda `perf/prefill-mmvq` (`f55b62d`, local). Qwen runs use `tools/bench_runtime.py`: the UD-IQ4_XS
pack, MTP, `--spec 4`, 256 tokens, and the prefix cache off. The prompt is `qwen4k.json`: the first 4,000 ids of
the decode-roofline lanes' 32K text and its last 8.

## Qwen3.8-Flash-Next: where a prompt's time goes

`GUILD_PREFILL_TIMING`, the 4,007-token prompt, 512-token chunks (`runs/qwen-c512.log`):
- **Time:** 52.1 s of GPU timeline. 22.7 s of it is the GPU waiting for expert copies.
- **GPU work:** the expert products 12.5 s (MMQ gate/up 9.4, down 3.1) plus 3.6 s in the phase the timer calls
  "dequant" (the MMQ path's gathers of each expert into its group), the GDN
  layers 6.1 s, the hyper-connections 1.9 s, and the QSA projections and attention 3.4 s.
- **Why the copies:** an 8 GB card holds none of the 24,576 experts (55.4 GiB). Each chunk streams every expert its
  tokens route to, and at 512 tokens that is nearly all of them: about 476 GB over PCIe for eight chunks.

Tried, and not kept:

| Change | Prompt | Decode | Text |
| --- | ---: | ---: | --- |
| `GUILD_PIN_EXPERTS=1` (the RAM copy page-locked: DMA straight from it) | 52.0 s against 52.2 | 34.0 tok/s against 36.8 (the arena loses its NUMA placement) | the same |
| `--prefill 1024` (the stream-everything ring) | out of memory at load | | |
| `--prefill 1024`, the routed-expert ring (`GUILD_PREFILL_STREAM_MIN=100000`) | 34.7 s | 41.8 tok/s | differs from token 15 |
| the same with 768-token chunks | 42.4 s | 39.2 tok/s | differs from token 15 |

The larger chunk halves the copy wait (22.7 -> 11.7 s), but the text moves.

**Why it moved.** The prefill's final residuals (`GUILD_PREFILL_DUMP_R`) already differ at position 0, so the cause
lies inside a chunk. A throwaway per-step dump of position 0 at layer 0 showed every attention-half step identical
(the hyper-connection read, the GDN projections, gates, recurrence and output). The router's logits and the shared
expert's output differed.

On Pascal the prompt path's BF16 products run as `cublasSgemm`, and the native ones as an FP16 `cublasGemmEx` after
a dequantization. cuBLAS picks its kernel by the row count, and a 1024-token chunk is one call of 1024 rows.

Slicing every product into calls of at most 512 rows, in a throwaway build, made the 1024-token chunks' residuals
(tokens 0 and 300 after every layer of the first chunk) and their 256 tokens equal to the 512-token chunks'. So
nothing else in this path depends on the chunk size on this card:
- llama.cpp's MMQ for the experts has no stream-k on Pascal, so an output's sum does not depend on how many rows its
  expert has;
- the GDN and attention kernels are per token.

**The change: `--prefill-rows N`** (`src/prefill/prefill.cpp`, `src/program/model_options.cpp`). Each prompt chunk is
computed as chunks of N tokens:
- every product runs in slices of at most N rows; the native product dequantizes its weight once and multiplies
  the later slices from that scratch;
- the expert ring is the one an N-token chunk gets, so the same expert path.

The experts are still streamed once a chunk. The flag is off by default, so every existing setting computes what it
did. `qwen_rows.sh` runs two of each, one process a setting:

| Context | `--prefill 512` | `--prefill 1024 --prefill-rows 512` | Its copy wait | Decode | Text |
| --- | ---: | ---: | ---: | --- | --- |
| 32,768 | 53.7 / 53.8 s | 35.6 / 35.6 s | 24.4 -> 12.3 s | 37.0 / 36.9 -> 36.5 / 36.6 tok/s | the same 256 tokens |
| 262,144, `--kv-host-only` (the server's) | 54.5 / 54.4 s | 36.2 / 36.2 s | 25.0 -> 13.0 s | 36.5 / 36.5 -> 36.7 / 36.6 tok/s | the same 256 tokens |

Both arms of both contexts give the same ids, which are also those of the earlier `--prefill 512` runs
(`runs/qwen-rows-*.json`, `cmp_ids.py`).

Where MMQ uses stream-k (llama.cpp turns it on from Volta), its expert sums depend on the work split. On those cards
a 1024-token chunk computed as two of 512 is not guaranteed to equal two 512-token chunks: it is another rounding of
the same products, and it was not measured here.

## GLM-5.3-Flash: the CPU experts

A 2,051-token prompt (`runs/glm-prompt2051-phases.log`, `GUILD_GLM_TIMING`) takes 32.8 s, of which the CPU experts
take 32.1 s. Their phases:
- gate/up 17.4 s at ~21 GMAC/s a thread;
- down 14.2 s at ~13 GMAC/s a thread;
- the threads 98% busy, reading ~60 GB/s of rows.

So the prompt is the CPU kernels' arithmetic, not the memory's bandwidth.

**The IQ4_XS rows on AVX-512 BW** (guild-kernels-cpu `9b7281a`; Guild `e50b4a3`).
- **Gap:** GLM's down rows are IQ4_XS, which the kernel repository's AVX-512 i-quant rows did not cover. Those rows
  run only where `cpu_avx512_ok()`, which wants VNNI and VBMI too, so this Xeon always took the AVX-2 rows.
- **Kernel:** `Fmt<23>` adds IQ4_XS, with each token's row ggml-cpu's dot to the bit (`iq_ggml_order_test` with
  `IQ512_TEST_BW=1`: 0 of 72 token-sets differ, 1-8 tokens).
- **Guild:** `cpu_moe` runs IQ4_XS rows of several tokens through it on any AVX-512 BW CPU (`GUILD_IQ4XS_512=0`:
  the AVX-2 rows).

`iq4xs_rows_bench.cpp`, one thread, 2,048-value rows in 16-row items:

| Tokens | 1 | 2 | 4 | 8 |
| --- | ---: | ---: | ---: | ---: |
| AVX-2, GMAC/s | 14.5 | 20.4 | 24.9 | 26.1 |
| AVX-512 BW, GMAC/s | 16.9 | 27.5 | 33.9 | 36.7 |

In the engine (`glm_iq4xs_prompt.sh`, one process alternated three times): the 2,051-token prompt 32.7 / 33.2 /
33.6 -> 31.2 / 31.5 / 31.4 s, the same prompt logits and ids.

Run in decode too, the same kernel for one- and two-token groups did not pay (`glm_iq4xs_decode.sh`, four
alternations: the step 84.6 / 86.9 / 88.0 / 89.0 ms against 87.7 / 87.6 / 88.9 / 89.4; the rows are the memory's
there). That build was not kept.

**The down phase in items of 64 rows of every expert** (`34118b5`).
- **Measured:** in a decode step the gate/up rows streamed 179 GB/s and the down rows 147 (`GUILD_GLM_TIMING`, which
  now prints both).
- **The change:** a down item took 16 rows of each expert; GLM now takes 64 (`cpu_moe::Experts::down_rows`). The
  row partition touches no sum.
- **GLM** (`glm_down_items.sh`, one process alternated six times, 64 first): CPU experts 71.8-73.1 -> 68.1-70.7 ms
  a step, the step 85.7-86.5 -> 84.5-85.6 ms. The 2,051-token prompt 31.15-31.24 -> 30.85-30.99 s
  (`runs/glm-down-items-prompt.out`). The same ids and prompt logits.
- **Nemotron 3.5 Transcribe** (`down_items_nemotron_glm.sh`, two alternated rounds): its Q8_0 down rows were ~1%
  slower with 64 (the 29.6 s clip greedy 1967 -> 1989 ms decode), the same transcripts. It keeps 16.

Both scripts ran with a throwaway switch (`GUILD_MOE_DRB`) that was not kept.

**What GLM's CPU side is held by.**
- `glm5_nl4_test --bench`, one thread: the NL4 gate/up rows run 7.1 GB/s of rows at four tokens (~54 GMAC/s at
  single-core turbo), against ~21 GMAC/s a thread with all 32 cores in the prompt.
- **Drift:** within one process the CPU experts' time drifts up by ~5 ms a step over two to three minutes of decode,
  in every arm alike, with the machine otherwise idle (core temperatures to 75 C). That is why every comparison here
  alternates its arms in one process.

## Files

`qwen_chunks.sh`, `qwen_small_ring.sh`, `qwen_rows.sh` (the chunk sizes, the small ring, `--prefill-rows`);
`glm_iq4xs_prompt.sh`, `glm_iq4xs_decode.sh`, `glm_down_items.sh`, `down_items_nemotron_glm.sh`;
`iq4xs_rows_bench.cpp` (built against guild-kernels-cpu's library); `runs/` holds their logs and JSON.
