# GLM-5.3-Flash at a 262,144-token context on an 8 GB card, and its long-context attention

GLM-5.3-Flash now holds the server's 262,144-token context on the GTX 1070 (8 GB), where it held 32,768: the
latent rows of the cells past what the VRAM holds live in pinned host memory, and each window gathers the rows its
indexer selected.  The output does not change: a 32K prompt at 262,144 cells gives the 32,768-cell run's prompt
logits and decoded ids bit for bit.  The attention kernel the long context leans on is 1.85x faster with the same
bits, and the gather reads a prompt window's shared rows once.

PC: 2x Xeon Gold 6130 (16 cores each, AVX-512, 2 NUMA nodes, 376 GB), GTX 1070 8 GB on PCIe 3.0 x16, CUDA 12;
`GLM-5.3-Flash-UD-IQ4_XS` (experts in RAM, split over both nodes).  Branch `feat/nemotron-transcribe` (on
`feat/glm5`), built against guild-kernels-cpu `c12ae99` and guild-kernels-cuda `f55b62d` (both local).

## What a token of context costs

A DSA layer keeps, per cell, its FP16 latent row (512 values, 1,024 B), a quarter of a pooled indexer key (64 B)
and, for a prompt window, a row of indexer scores; over the 11 DSA layers and the NextN block that is 13,184 B of
VRAM a cell.  The 8 GB card had room for 32,768 cells next to the weights (with four shared experts moved to their CPU
copies).  262,144 cells would need 3.2 GiB.

Of the 1,088 bytes, the latent is 1,024, and attention reads only the rows the indexer selects: 2,048 cells a query
(its top 512 pools of 4) and the cells of its own incomplete pool, at any context length.  The pooled keys are read
whole by every query (the indexer scores all of them), so they stay in VRAM.

## The change

- **Placement** (`src/core/glm5.cpp`, `load`): the VRAM budget is placed as before (32K cells first, then the
  FFNs, then more context while it fits).  When the context asked for is longer, the budget of the cells placed so
  far is split again: every cell's pooled key and score row (896 B), a 16 MiB stage, and as many cells' latent rows
  as are left (13,184 B each).  On this card at 262,144: the first 14,848 cells keep their latents in VRAM, the rest
  (247,296 cells, 2,898 MiB) in pinned, mapped host memory, and the FFNs stay where they are at 32K.  The log says
  `context 262144 tokens 14848 x 13184 B + 247296 x 896 B + a 16 MiB stage`.  `GUILD_GLM_KV_HOST=0` turns it off
  (the context is then what the VRAM holds); `GUILD_GLM_KV_NEAR=<cells>` caps the VRAM cells (tests).
- **Writing a cell**: `kv_store` writes cells below the VRAM part's end to VRAM and the rest through the mapping.
- **Reading**: a window whose cells all lie in VRAM runs as before.  Otherwise its selected rows are copied into
  the stage, eight queries at a time, and attention runs over the stage with the same cells in the same order, and
  with the whole window's split of each query's cells (`mla_attend`'s new `splits_of`), so every sum is the one the
  all-VRAM window computes.  A one-token window copies its 2,048 rows (`kv_gather`); a window of several tokens
  copies the union of its queries' rows once each (`kv_gather_union`: mark the cells in a bitmap, count them, list
  them, copy, renumber each query's selection).
- **The attention kernel** (`mla_attend2_k`, `src/core/glm5.cu`): the same arithmetic as `mla_attend_k` with fewer
  instructions.  A chunk's 16 rows go to shared memory as floats, each lane's 16 values side by side (swizzled, so
  a quarter warp's 16-byte reads meet no bank twice), read with four 16-byte loads instead of sixteen 2-byte loads
  and conversions, the rows fetched from global memory 16 bytes a thread; 32 KiB and 80 registers, three blocks an
  SM.  `GUILD_GLM_ATTEND_LEGACY=1` runs the old kernel (read each window, for A/Bs in one process).

## Exactness

- `glm5_kernels_test` (all pass): the new attention kernel against the old, bitwise, causal and selected, windows
  of 1, 2 and 20 tokens; attention over rows gathered from VRAM and mapped host memory, per query and as a union,
  eight queries at a time with the window's split, against the whole window over one VRAM cache, bitwise, windows
  of 1, 2, 3, 20 and 37 tokens.
- In the engine, `glm5_run --spec`, the same binary (before the new kernel):
  - a 15,999-token prompt, 48 tokens, all cells in VRAM against `GUILD_GLM_KV_NEAR=4096` (cells from 4,096 on in
    host memory): the same prompt logits (`--logits`, `cmp`) and the same ids;
  - a 31,991-token prompt, 64 tokens, `--ctx 32768` (all in VRAM) against `--ctx 262144` (cells from 15,104 on in
    host memory): the same prompt logits and the same ids.

## Speed

`glm5_run --spec`, 64 tokens; the prompt is the decode-roofline lanes' 31,991-token text.  The first three rows
are separate processes of the binary before the new kernel (`run.sh`); the last two are one process
(`ab.sh`: the prompt once, then the decode from a snapshot three times with each kernel, alternated).

| Context, attention | Prompt | Decode step, median | Verify | Decode |
| --- | ---: | ---: | ---: | ---: |
| 32,768 cells, all in VRAM, old kernel | 612.8 s (52.2 tok/s) | 101.8 ms | 93.8 ms | 17.20 tok/s |
| 262,144 cells, each query's rows gathered, old kernel | 678.6 s (47.1 tok/s) | 103.4 ms | 97.2 ms | 16.60 tok/s |
| 262,144 cells, the union gathered, new kernel | 562.1 s (56.9 tok/s) | | | |
| the same process, decoding with the old kernel | | 103.7 / 103.8 / 103.9 ms | 95.5-95.9 ms | 16.82-16.89 tok/s |
| the same process, decoding with the new kernel | | 97.4 / 97.5 / 97.6 ms | 89.8 ms | 17.62-17.96 tok/s |

So at 32K tokens the long context costs ~2 ms a step for its gathers, and the new kernel gives back ~6 ms; the
prompt reads its rows from host memory and is still 8% faster than the all-VRAM run with the old kernel was.  A
16K prompt with `GUILD_GLM_KV_NEAR=4096` (three quarters of its cells in host memory, the old kernel, each query's
rows): prompt 302.6 -> 352.7 s, decode step 101.4 -> 103.4 ms.

The kernels alone (`glm5_kernels_test attend` and `indexer`, 2,048 selected cells of a 65,536-cell cache, 64 heads;
the gathers from mapped host memory, 1,792 of each query's cells shared by the window's queries):

| Queries | Attention, old kernel | New kernel | Rows from host, each query's | The union's |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 681 us | 368 us | 295 us (7.1 GB/s) | 297 us |
| 2 | 1,302 us | 712 us | 562 us | 332 us |
| 8 | 4,449 us | 2,585 us | 2,415 us | 533 us |
| 128 | 64.6 ms | 39.2 ms | 38.1 ms | 8.5 ms |

The layouts tried for the new kernel, one query: padded rows (40 KiB, 112 registers, two blocks an SM) 413 us,
swizzled (32 KiB) 416 us, swizzled with three blocks an SM (80 registers) 368 us, kept.

The indexer at 262,144 cells, a query: scores 263 us, selection 97 us (at 32,768: 41 and 19 us).

At a short context nothing moves (`short.sh`: the 5-token prompt, 256 tokens, one process a context, the old and the
new kernel alternated three times each): `--ctx 262144` gives `--ctx 32768`'s 256 ids, and the steps are the CPU's
there (CPU experts 70-75 ms of 85-89 ms), the same within the runs' drift in both processes and with either kernel:
21.76 / 21.26 / 20.98 tok/s (old) and 21.35 / 21.01 / 20.92 (new) at 32,768; 21.54 / 21.00 / 20.77 and 21.27 /
20.81 / 20.76 at 262,144.

## What is left

- **The decode's gathers**: a one-token window copies its 2,048 rows from host memory every step (~0.3 ms a layer,
  ~2 ms a step at 32K, more as the context fills), though consecutive tokens select nearly the same cells.  A VRAM
  cache of the host rows the last steps read would leave only the new ones to copy.
- **The prompt** at 262,144 tokens is the CPU experts' (about 57 tok/s at 32K: 1.3 hours for a full context).
- **The attention kernel**, 368 us a query, is held by its occupancy (three blocks of eight warps an SM, each
  walking its share of the cells 16 at a time between two barriers); more blocks an SM need fewer than 80
  registers a thread.

## Files

`run.sh` (the four exactness runs: `vram16`, `host16`, `host256`, `vram32`), `attend.sh` (the kernel bench and
tests), `ab.sh` (the 262,144-cell prompt and the attention A/B), `short.sh` (a short context at both settings, and
the 2,051-token prompt's CPU phases); their logs in `runs/`.
