# GLM-5.3-Flash at a 262,144-token context on an 8 GB card, and its long-context attention

GLM-5.3-Flash now holds the server's 262,144-token context on the GTX 1070 (8 GB), where it held 32,768. The latent
rows of the cells past what the VRAM holds live in pinned host memory; each window copies in the rows its indexer
selected, and a decoded token copies only those its layer's last token did not select.

The output does not change: a 32K prompt at 262,144 cells gives the 32,768-cell run's prompt logits and decoded ids
bit for bit.

The attention kernel the long context leans on is twice as fast with the same bits. After the 32K prompt a decode
step takes 95.4 ms against 103.8 with the original kernel; after a 128K prompt GLM decodes at 18 tok/s.

PC: 2x Xeon Gold 6130 (16 cores each, AVX-512, 2 NUMA nodes, 376 GB), GTX 1070 8 GB on PCIe 3.0 x16, CUDA 12;
`GLM-5.3-Flash-UD-IQ4_XS` (experts in RAM, split over both nodes). Branch `feat/nemotron-transcribe` (on
`feat/glm5`), built against guild-kernels-cpu `c12ae99`/`9b7281a` and guild-kernels-cuda `f55b62d` (all local).
Commits `e7fc761` (host latents, the gathers, the first new kernel) and `79855f1` (FP16 rows in the kernel, the row
cache).

## What a token of context costs

A DSA layer keeps, per cell:
- its FP16 latent row (512 values, 1,024 B);
- a quarter of a pooled indexer key (64 B);
- for a prompt window, a row of indexer scores.

Over the 11 DSA layers and the NextN block that is 13,184 B of VRAM a cell. The 8 GB card had room for 32,768 cells
next to the weights (with four shared experts moved to their CPU copies); 262,144 cells would need 3.2 GiB.

Of the 1,088 bytes, the latent is 1,024. Attention reads only the rows the indexer selects: 2,048 cells a query (its
top 512 pools of 4) and the cells of its own incomplete pool, at any context length. The pooled keys are read whole by
every query (the indexer scores all of them), so they stay in VRAM.

## The change

- **Placement** (`src/core/glm5.cpp`, `load`). The VRAM budget is placed as before: 32K cells first, then the FFNs,
  then more context while it fits. When the context asked for is longer, the budget of the cells placed so far is
  split again:
  - every cell's pooled key and score row (896 B);
  - a stage and the row caches (52 MiB);
  - as many cells' latent rows as are left (13,184 B each).

  On this card at 262,144: the first 11,776 cells keep their latents in VRAM, the rest (250,368 cells, 2,934 MiB)
  sit in pinned, mapped host memory, and the FFNs stay where they are at 32K.
  - `GUILD_GLM_KV_HOST=0` turns it off (the context is then what the VRAM holds).
  - `GUILD_GLM_KV_NEAR=<cells>` caps the VRAM cells (tests).
- **Writing a cell.** `kv_store` writes cells below the VRAM part's end to VRAM and the rest through the mapping.
- **Reading.** A window whose cells all lie in VRAM runs as before. Otherwise its rows are copied to the GPU, and
  attention runs over the copy with the same cells in the same order. Each query's cells are split as the whole
  window splits them (`mla_attend`'s new `splits_of`), so every sum is the one the all-VRAM window computes.
  - **A window of several tokens** (a prompt's): the union of eight queries' rows at a time into a stage, each row
    once (`kv_gather_union`: mark the cells in a bitmap, count them, list them, copy, renumber each query's
    selection).
  - **A decoded token**: each DSA layer keeps the rows its last token read, in 2,052 slots (`kv_cache_rows`). A
    cached cell keeps its slot; a new cell takes a slot no selected cell holds and is copied in; attention reads
    through the slots in the selection's order.
  - Every write of a cell forgets its slot, so a rolled-back or new conversation's cell is copied again.
  - `GUILD_GLM_ROW_CACHE=0` copies each token's 2,048 rows whole (read each window; at load it also frees the VRAM).
- **The attention kernel** (`mla_attend2_k`, `src/core/glm5.cu`): `mla_attend_k`'s arithmetic, every product, sum
  and softmax step in its order, with fewer instructions.
  - **Rows:** a chunk's 16 rows go to shared memory as FP16, each lane's 16 values side by side in two swizzled
    16-byte pieces, read with two 16-byte loads (where the old kernel did sixteen 2-byte loads) and converted to
    float as read. Global memory is read 16 bytes a thread.
  - **Occupancy:** 16 KiB a block, three blocks an SM for a decode step's 128 blocks, four for longer launches.
  - `GUILD_GLM_ATTEND_LEGACY=1` runs the old kernel (read each window, for A/Bs in one process).

## Exactness

- **`glm5_kernels_test` (all pass), bitwise:**
  - the new attention kernel against the old: causal and selected, windows of 1, 2 and 20 tokens;
  - attention over rows gathered from VRAM and mapped host memory, per query and as a union, eight queries at a
    time with the window's split, against the whole window over one VRAM cache: windows of 1, 2, 3, 20 and 37
    tokens;
  - the row cache over eight one-token steps whose selections overlap (~10% new cells a step; at step 8, 214 of
    2,048 copied), with cells rewritten and forgotten between steps, against the whole cache.
- **In the engine** (`glm5_run --spec`), prompt logits (`--logits`, `cmp`) and ids:
  - a 15,999-token prompt, 48 tokens, all cells in VRAM against `GUILD_GLM_KV_NEAR=4096` (cells from 4,096 on in
    host memory): the same;
  - a 31,991-token prompt, 64 tokens, `--ctx 32768` (all in VRAM) against `--ctx 262144`: the same with each
    query's rows gathered, the old kernel; again with the row cache (both settings in one process); and again with
    the final kernel (both kernels in one process);
  - after a 127,997-token prompt at 262,144 cells: the old and the float-row kernel's decodes give the same ids.

## Speed

`glm5_run --spec`, 64 tokens after the decode-roofline lanes' 31,991-token text. "One process" rows decode from a
snapshot after the prompt, the arms alternated three times; the others are separate processes.

| Setting | Prompt | Decode step, median | Decode |
| --- | ---: | ---: | ---: |
| 32,768 cells, all in VRAM, old kernel (`run.sh`) | 612.8 s (52.2 tok/s) | 101.8 ms | 17.20 tok/s |
| 262,144, each query's rows copied, old kernel (`run.sh`) | 678.6 s (47.1 tok/s) | 103.4 ms | 16.60 tok/s |
| 262,144, the union copied, float-row kernel (`ab.sh`, one process) | 562.1 s (56.9 tok/s) | old kernel 103.7 / 103.8 / 103.9 ms; new 97.4 / 97.5 / 97.6 | 16.8-16.9 / 17.6-18.0 tok/s |
| 262,144 with the row caches' 36 MB (11,776 cells in VRAM), float-row kernel (`rowcache.sh`, one process) | 544.7 s | rows copied whole 97.8 / 98.2 / 98.1 ms; row cache 96.5 / 96.6 / 96.7 | 17.8 / 18.0 tok/s |
| the same with the final kernel (`final.sh`, one process) | 540.6 s (59.2 tok/s) | old kernel 109.3 / 104.5 / 103.4 ms; final 101.9 / 95.3 / 95.5 | 16.4 / 17.5 tok/s |

`final.sh`'s first round had CPU noise in both arms (CPU experts ~80 ms a step against ~71 in the other rounds).
In the steady rounds the final kernel and the row cache take 8-9 ms off a step at 32K.

Also measured:
- **A 16K prompt with `GUILD_GLM_KV_NEAR=4096`** (three quarters of its cells in host memory, the old kernel, each
  query's rows): prompt 302.6 -> 352.7 s, decode step 101.4 -> 103.4 ms.
- **127,997 tokens** (`long128k.sh`, a corpus text made by `make_prompt.py 128000`, at 262,144 cells, before the row
  cache): the prompt 2,350 s (54.5 tok/s; the CPU experts 1,939 s of it). Decode from a snapshot, alternated twice:
  the old kernel 108.1 / 107.8 ms a step (16.5-16.6 tok/s), the float-row kernel 101.7 / 101.9 ms (18.0-18.2 tok/s),
  88% of the drafts accepted, the same ids.

**The kernels alone** (`glm5_kernels_test attend` and `indexer`). 2,048 selected cells of a 65,536-cell cache, 64
heads; for the copies from mapped host memory, 1,792 of each query's cells are shared by the window's queries:

| Queries | Attention, old kernel | Float rows | FP16 rows (final) | Rows from host, each query's | The union's |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 684 us | 369 us | 334 us (three blocks an SM) | 295 us (7.1 GB/s) | 297 us |
| 2 | 1,306 us | 715 us | 621 us (four) | 562 us | 332 us |
| 8 | 4,437 us | 2,576 us | 2,154 us (four) | 2,415 us | 533 us |
| 128 | 64.7 ms | 39.4 ms | 32.3 ms (four) | 38.1 ms | 8.5 ms |

Layouts tried, one query:
- float rows padded (40 KiB, 112 registers, two blocks an SM): 413 us;
- float rows swizzled (32 KiB): 416 us; the same at three blocks an SM: 368 us;
- FP16 rows at four blocks an SM (64 registers): 357 us, the best from 256 blocks a launch;
- FP16 rows at three blocks an SM: 331-334 us, the best for a decode step.

The indexer at 262,144 cells, a query: scores 263 us, selection 97 us (at 32,768: 41 and 19 us).

**Short context** (`short.sh`): nothing moves. The 5-token prompt, 256 tokens, one process a context, the old and the
new kernel alternated three times each:
- `--ctx 262144` gives `--ctx 32768`'s 256 ids;
- the steps are the CPU's there (CPU experts 70-75 ms of 85-89 ms), the same within the runs' drift in both
  processes and with either kernel: 21.76 / 21.26 / 20.98 tok/s (old) and 21.35 / 21.01 / 20.92 (new) at 32,768;
  21.54 / 21.00 / 20.77 and 21.27 / 20.81 / 20.76 at 262,144.

## What is left

- **The prompt** at long context is the CPU experts' (54-59 tok/s from 2K to 128K tokens: about 1.3 hours for a full
  262,144-token context). See [../2026-11-01-qwen-glm-pass3](../2026-11-01-qwen-glm-pass3/README.md) for what moved
  there.
- **The row caches cost 36 MB of VRAM**, 3,072 cells that now come from host memory: a context between 11,776 and
  14,848 tokens reads some rows over PCIe it did not before (through the cache, a few a step).
- **Long-context decode**: at 128K a step is ~6 ms longer than at 32K. The indexer's scores grow with the context
  (~140 us a query at 128K, 23-25 queries a step).

## Files

- `run.sh`: the four exactness runs (`vram16`, `host16`, `host256`, `vram32`).
- `attend.sh`: the kernel tests and benches.
- `ab.sh`: the 262,144-cell prompt and the float-row kernel's A/B.
- `rowcache.sh`: the row cache A/B.
- `final.sh`: the final kernel's tests, bench and A/B.
- `long128k.sh` with `make_prompt.py`: the 127,997-token run.
- `short.sh`: a short context at both settings, and the 2,051-token prompt's CPU phases.

Their logs are in `runs/`.
