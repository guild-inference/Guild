# Several requests at once: batch slots on the runtime, with drafts

PC: 2x Xeon Gold 6130 (16 cores each, AVX-512 BW without VNNI, 2 NUMA nodes, 376 GiB RAM), GTX 1070 8 GB (sm_61,
15 SMs, 151 W cap, PCIe 3.0 x16), CUDA 12. Branch `perf/parallel` (from `perf/next` 6066936), the kernel revisions
pinned then (guild-kernels-cpu `9b7281a`, guild-kernels-cuda `de65d60`). The CUDA pin is now `11936b5`
([below](#the-cuda-kernels-at-11936b5)).

Model: Qwen3.8-Flash-Next UD-IQ4_XS with setup's flags (`--dflash` with `data/dflash_vocab.bin`, `--spec 8`,
`--dense q4_k --dense-map ssm_out=q5_k,attn_output=q8_0 --dense-imatrix`, `--hc q8_0`, `--prefill-mmq dense`,
`--prefill auto`), `--max-context 32768 --kv fp16 --kv-host-only`. The 8 GB card holds no GPU expert cache for this
model: every routed expert runs on the CPU. Ornith-1.5-35B Q4_K_M with its MTP layer (`--spec 4`) for the second
model.

## What was broken

The server's `"parallel": N` (the engine's `--batch N`, #465) did nothing. `guild-generate` became a thin wrapper
around the runtime on 2026-10-06 (4530824) and dropped the old engine's batch slots: it ignored `--batch`, never said
`INFO batch_slots`, and the server served one request at a time whatever the config said. The verifier's batch
windows (`init_slots`, `run_slot_rows`) were still there, unused.

## What the engine does now

- **Slots.** `--batch N` carves N slot sessions at load, like the main session (same context, same K/V mode), before
  the drafter, the expert cache and the prompt chunk take their share of the VRAM: as many as fit, at most 8.
- **Admission.** A request (`BGEN`) reads its prompt on the usual path (prompt cache, prompt engine) up to its first
  token. Its state is then copied into the slot buffer to buffer (`conversation_kv_copy`, `conversation_state_copy`):
  1.7-2.6 ms for a short conversation, 16.5 ms for 4,890 tokens.
- **Batch windows.** Every active slot gets rows in one verify window: the dense weights, the shared expert and the
  head are read once for all of them. A slot keeps its conversation when it ends, so its next turn (or the same
  request going back to the solo path) continues from it.
- **Long prompts.** A long prompt is read one chunk a run, at the boundaries one run has, with the slots decoding
  between the chunks. `BYIELD` parks the part read in a slot at a chunk boundary.
- **Drafts.** While at most 3 slots decode, a window's 8 rows are shared out among them. Each slot drafts from its
  own DFlash context (40 MiB, kept current every window, copied in at admission and back with the conversation).
  Each row computes what row t of that slot's own speculative window computes, and a commit keeps each slot's
  accepted rows.
- **Exactness.** With greedy decoding, every slot produces exactly its solo tokens. This held in
  `tools/batch_test.py` (which sets `GUILD_IQ_MT_MIN=1`) at 2, 3 and 7 slots, with and without drafts, with the
  default kernels too (`--mt-min ""`, 3 slots with drafts, `runs/dflt.out`), and in all eight comparisons of
  `tools/batch_interleave_test.py`: a 11,713-token prompt read beside two decoding slots, a next turn continued from
  its slot, a prompt that gave way and went on, a request moved back to the solo path mid-answer, and a history sent
  back without the reply's thinking. The solo path's tokens are unchanged (Qwen bench ids
  `[38747, 38747, 41916, 41916]`, Ornith `[30936, 30936, 65886, 65886]`).

## Through the server

`tools/parallel_bench.py`: the HTTP server with the run config, C different chat requests sent at once (streamed,
greedy, thinking off, 200 tokens each), two rounds per C (the rounds' questions differ). Total tok/s is every
completion token over the round's wall time; the first-token times are the median request's and the last one's.

| At once | `"parallel"` off: total tok/s | first token median / last | `"parallel": 7`: total tok/s | first token median / last |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 51.4 / 75.3 | 0.31-0.37 s | 51.0 / 74.7 | 0.32-0.36 s |
| 2 | 63.6 / 46.8 | 1.41 / 2.76 s | 62.1 / 52.5 | 0.60 / 0.85 s |
| 3 | - | - | 63.4 / 57.8 | 0.86 / 1.30 s |
| 4 | 52.9 / 48.8 | 5.57 / 11.33 s | 61.0 / 61.0 | 1.09 / 1.76 s |
| 7 | 51.1 / 50.3 | 11.15 / 23.63 s | 72.6 / 72.9 | 1.72 / 3.00 s |

A request alone is not slower: the server keeps it on the solo path with its drafts. From four requests at once the
slots decode 15-45% more in total, and the last request starts within 1.8-3.0 s instead of 11-24 s. Before the drafts
in the slots, two requests at once decoded 41.3 tok/s in total (`runs/http-p7.out`); with them, as much as one after
the other.

## The batch windows (engine level)

`tools/batch_test.py`: the same prompts alone (the solo path, with its drafts), then together in the slots, 96 tokens
each. "Together" is the tokens over the time from the first batch token; "alone" is the tokens over the prompts'
decode times summed.

| Slots | Rows a window | Tokens a window | ms a window | Together tok/s | Alone tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3, one row a slot, before the changes below | 2.94 | 2.94 | 63.2 | 43.3 | 53.5 |
| 3, one row a slot | 2.94 | 2.94 | 51.1 | 52.6 | 53.5 |
| 7, one row a slot | 6.58 | 6.58 | 79.7 | 69.1 | 48.5 |
| 2, with drafts (4 + 4 rows) | 8.00 | 4.75 | 91.3 | 50.3 | 53.1 |
| 3, with drafts (3 + 3 + 2) | 8.00 | 6.20 | 96.6 | 58.3 | 53.3 |
| Ornith, 5, one row a slot | 4.80 | 4.80 | 33.5 | 126.8 | 79.6 |

What took a 3-row window from 63.2 to 51.1 ms:
- **Two groups.** The window runs in two groups on one GPU, as a speculative window does: the CPU's experts of one
  group overlap the GPU's layer work of the other. It was one group, with 30.4 ms waiting for the GPU and 28.5 ms of
  CPU experts back to back. `GUILD_BATCH_SPLIT=0` keeps one group.
- **Host combine.** The host sums the CPU's expert rows, so one row per token crosses PCIe instead of ten.
- **No commit.** The window advances each slot's GDN recurrence and conv history as it reads them (the commit form of
  the same kernels: the same bits), so it needs no commit. It used to have a 4.6 ms commit the host waited for.
  `GUILD_BATCH_INPLACE=0` restores it.
- **One graph per row count.** The window's graph is updated in place (`cudaGraphExecUpdate`) when the set of slots
  changes. Before, there was one graph per set of slots, 2^N - 1 of them at ~36 MiB of VRAM each.
  `GUILD_BATCH_GRAPH_KEEP=1` keeps one graph per set.

Where the time goes (`runs/prof*.engine.txt`, one-group windows under `GUILD_VERIFY_PROFILE=1`):
- **GPU.** Busy ~25 ms for one row plus ~6.5 ms per extra row: the dense products' extra columns, a GDN recurrence per
  row, and attention per row.
- **CPU.** The experts run at ~126 GB/s with one row a slot and ~106 GB/s with drafts. A 7-row window spends 64 ms of
  its 79.7 ms on CPU experts, and a drafted 8-row window 57-65 ms of ~91-97.
- **The bottleneck.** The CPU expert pool limits the larger windows: the machine reads ~220 GB/s.
- **Drafts.** They pay at 2 slots (4.75 tokens a window against 2) and break even at 3. From 4 slots they would add
  rows the CPU has no time for, so `GUILD_BATCH_DRAFT_SLOTS` is 3.

## What a slot costs

- **VRAM.** 274 MiB per slot for Qwen at 32K, plus 40 MiB for its DFlash context. The streamed K/V stages two window
  rows' selections (1,028 pages, the resolver's floor of 1,024), not a speculative window's eight: it was 564 MiB.
  Ornith's slot is 787 MiB, because its dense attention stages the whole 32K window. Each batch window graph takes
  34-38 MiB, made when its shape is first used. If VRAM runs out for one, the other batch graphs are given back.
- **Pinned RAM.** Each slot holds its whole K/V in pinned RAM with `--kv-host-only`: 768 MiB at 32K in fp16.
- **Prompt speed.** The prompt chunk grows into the VRAM the slots leave (`runs/pc.out`, a 4,000-token prompt, no
  prefix cache):

| Slots | Prompt chunk | 4,000 tokens read in |
| ---: | ---: | ---: |
| 0 | 8,192 | 15.3 s |
| 3 | 4,608 | 15.3 s |
| 6 (7 asked; the drafter contexts' reserve leaves 6) | 512 | 37.9 s |

On this card, 3 slots keep prompts of up to ~4.6K tokens at full speed. Six or seven slots give the most to several
users at once, but a long prompt then reads 2.5 times slower.

## The CUDA kernels at 11936b5

After the measurements above, `backends.lock.json`'s guild-kernels-cuda revision moved from `de65d60` to `11936b5`
(round 18's `kcuda-onechunk`, pushed as `perf/onechunk-mmvq`). Q4_K rows of 8-16 blocks and Q6_K rows of exactly 8
blocks now run as one chunk, and 4-column Q4_K calls on rows of 9-12 blocks run as two 2-column calls. The arithmetic
and its order are the same. The tables above were measured at `de65d60`.

A build from the pins alone (no kernel source overrides) against the development build at `de65d60`, same Guild
sources, one request at a time (`qwen-bench-r18` / `ornith-bench-r18`: setup's flags, 2 runs x 256 tokens, prose and
code). Two pairs, run alternately:

| Model | prose tok/s, `de65d60` -> `11936b5` | code tok/s, `de65d60` -> `11936b5` | ids, both |
| --- | --- | --- | --- |
| Qwen3.8 | 49.78 / 49.40 -> 51.71 / 51.38 (+3.9%) | 62.61 / 62.39 -> 64.81 / 64.64 (+3.6%) | `[38747, 38747, 41916, 41916]` |
| Ornith-1.5 | 80.03 / 80.09 -> 83.10 / 82.85 (+3.6%) | 92.75 / 92.94 -> 97.24 / 96.81 (+4.5%) | `[30936, 30936, 65886, 65886]` |

With the pinned build:
- **Parity tests.** The 23 GPU parity tests pass: `mmvq`, `gr_parity`, `route_window`, `router_fast`, `host_combine`,
  `native_expert`, `dflash`, `gdn_step` and `glm5_kernels`. `mmvq_staged_parity_one_wide_off` covers the old path.
- **Exact slots.** Every slot still gives its solo tokens: `batch_test.py` at 3 slots with drafts, at 6 slots, and
  Ornith at 5 (`runs/pin-*.out`), plus all eight comparisons of `batch_interleave_test.py --long 12000`.
- **Smaller batch gain.** The batch windows gain less, because the CPU's experts limit them. 3 slots with drafts gave
  58.9 tok/s (58.3 above) and Ornith at 5 slots 130.4 (126.8), one run each.
- **Drama 3.** Its WAVs keep their md5s: seed 11 `6e04c9b1`, seed 12 `f84288c9`.

## Limits

- **Models.** Qwen3.8-Flash-Next (and its variants) and Ornith-1.5-35B, the models on the verify-window engine.
  GLM-5.3-Flash's engine and the text-to-speech model run one request at a time and say so.
- **Ornith.** Its MTP layer drafts only on the solo path; its slots carry one row each.
- **Sampling.** Penalties are not applied in batch windows. Sampled slots draw row by row with the solo window's
  Philox(seed, position).
- **DFlash blocks.** The slots draft whole blocks, as the solo path does, and take the rows they need. A block
  shorter than the drafter's (8) used to fault: `topk_lse` kept its counts behind the parts, at an offset that moved
  with the rows, so they were not zero. It is fixed (the counts come first), and `GUILD_BATCH_DRAFT_BLOCK=rows` drafts
  blocks of a slot's rows. That saved 0.5-1.5 ms of drafting a window and was within noise (2 slots 52.0 -> 53.0, 3
  slots 61.9 -> 61.0 tok/s, `runs/blk-*.out`), so whole blocks stay the default.
