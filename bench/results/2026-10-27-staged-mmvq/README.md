# GTX 1070 decode: staged MMVQ rows, batched routing on CUDA

Measured 2026-10-27 on this PC: Ubuntu, two Xeon Gold 6130 (2 NUMA nodes), 376 GiB RAM, GTX 1070 8 GiB (sm_61),
driver 580.178.04, CUDA 12.0 Release build of `guild-generate` (`-DGUILD_EXPERIMENTAL_SM60=ON`) on branch
`feat/glm5`, with `GUILD_CUDA_SOURCE_DIR` on a `guild-kernels-cuda` checkout of branch `perf/sm61-decode`:
`fbd2304` before, `1c04475`, `9cff816`, `25495c3` and `429b632` after. That branch is local, not pushed, and
`backends.lock.json` still pins `fe4c080`.

Every kernel change keeps its outputs bit for bit. Decode rows are medians of three 256-token greedy generations from
`tools/bench_runtime.py` on its prose and code prompts, with the settings of
[2026-10-24-numa-iq3s](../2026-10-24-numa-iq3s/README.md) (Ornith with its MTP drafter and the English draft
vocabulary, Qwen with its MTP draft layer, `--spec 4`). The arms of each A/B are the same binary with an environment
switch, run alternately, two rounds each, and their texts are compared.

## Decode

| Model | Before | After | Change |
| --- | ---: | ---: | --- |
| Ornith-1.5-35B Q4_K_M (prose / code) | 67.6 / 78.9 tok/s | 72.8 / 85.2 tok/s | staged MMVQ, then batched routing |
| Qwen3.8-Flash-Next UD-IQ4_XS (prose / code) | 28.6 / 36.6 tok/s | 29.7 / 38.1 tok/s | staged MMVQ with records |
| Fish Audio Drama 3 Q4_K_M, the 290-character passage, seeds 11 and 12 | 37.3 frames/s | 38.1 frames/s | staged MMVQ |
| GLM-5.3-Flash UD-IQ4_XS, `glm5_run --spec --ctx 32768`, 128 tokens | 21.2 tok/s | 21.9 tok/s | within noise |

- Ornith: `GUILD_MMVQ_STAGED=0` against the default, 67.6 -> 71.9 (prose) and 78.9 -> 83.8 (code); then
  `GUILD_ROUTE_PER_TOKEN=1` against the default, 71.8 -> 72.8 and 83.6 -> 85.2. A verify window took 29.1 -> 28.6 ms
  (code) with the routing change alone.
- Qwen: `GUILD_MMVQ_STAGED=0` against the default on the final binary; a verify window took 76.4 -> 73.9 ms (prose)
  and 82.4 -> 79.1 ms (code), the draft 5.6 -> 5.3 ms of it. With the staged rows alone (before the records) the same
  A/B gave 27.5 -> 28.4 and 36.3 -> 37.5.
- The router change of section 4, the binary before against the one after: Ornith 73.0 -> 73.5 (prose) and
  85.5 -> 85.7 tok/s (code), verify windows 27.6 -> 27.4 ms.
- Drama 3: the tower's time per run 4.19 -> 4.10 s and 3.75 -> 3.67 s; the WAV files are byte-identical.
- GLM-5.3-Flash keeps its own staged kernels for its hot products (section "Tried and not kept"), so only its smaller
  ones changed. Its CPU experts' time moved by 2.8 ms a step between the two runs, more than the verify step did
  (82.0 -> 78.9 ms); a first pair of runs gave 18.7 against 21.7 tok/s with one run's draft at 8.6 ms instead of
  about 4.3, for no reason found. All four runs wrote the same 128 tokens.

## 1. Staged rows (`1c04475`)

`native_mmvq` read a row of Q6_K or Q8_0 with 2-byte loads (their 210- and 34-byte blocks are only 2-byte aligned)
and Q4_K's with 4-byte loads, 128 threads a row. Now, on CUDA, a warp owns a row at a time and copies it into shared
memory 8 blocks (Q4_K, Q6_K) or 32 blocks (Q8_0) at a time with aligned 16-byte loads, the next chunk arriving in
registers while it computes on the last. The grid is as many blocks as the GPU holds at once and every warp takes
every (blocks x 8)-th row, so a warp's next row loads while it finishes the last and a block copies the activations
of all its columns into shared memory once for all its rows. Lane `l` stands for the threads `l`, `32 + l`,
`64 + l` and `96 + l` of the one-column kernel: the same blocks in the same order into four sums, added in the same
order and then by the same xor tree, so every output is bitwise the one-column kernel's at every column count.

It runs for 1-4 columns of Q8_0, 1-3 of Q4_K and 1-8 of Q6_K, and only for matrices of at least 100 KiB per SM
(1.5 MB here); `GUILD_MMVQ_STAGED=0` turns it off. GB/s with the weights read from DRAM (`glm5_kernels_test shapes`
cycles through 64 MB of copies of each matrix; the GTX 1070's L2 holds 2 MB):

| Matrix | 1 column | 2 columns | 3 columns | 4 columns |
| --- | ---: | ---: | ---: | ---: |
| Q8_0 2560 -> 12288 (Qwen's q) | 196 -> 210 | 188 -> 207 | 177 -> 204 | 172 -> 192 |
| Q8_0 2560 -> 10240 (Qwen's qkv) | 195 -> 209 | 187 -> 205 | 175 -> 202 | 170 -> 190 |
| Q8_0 6144 -> 2560 (Qwen's ssm_out) | 198 -> 206 | 189 -> 201 | 178 -> 197 | 175 -> 182 |
| Q4_K 2048 -> 8192 (Ornith) | 161 -> 194 | 137 -> 165 | 104 -> 129 | 101 (kept) |
| Q4_K 2048 -> 4096 (Ornith) | 154 -> 182 | 131 -> 145 | 100 -> 118 | 97 (kept) |
| Q6_K 2048 -> 8192 (Ornith) | 155 -> 197 | 132 -> 180 | 120 -> 157 | 107 -> 135 |
| Q6_K 2048 -> 248320 (Ornith's head) | 160 -> 213 | 135 -> 208 | 122 -> 175 | 108 -> 145 |
| Q6_K 2560 -> 248320 (Qwen's head) | 160 -> 210 | 132 -> 172 | 117 -> 149 | 106 -> 113 |

Matrices of 512 or 640 rows stay on the old kernels: a warp a row leaves the GPU short of rows there (Q8_0 2560 ->
512 at 135 against 147 GB/s, Q4_K 2048 -> 512 at 96 against 108). Q8_0 past four columns and Q4_K at four lost to
the multi-column kernels and stay on them.

`mmvq_staged_parity` checks the outputs against the kernels before (one run with `GUILD_MMVQ_STAGED=0` dumps, the
next compares): 13 shapes at 1-8 columns, 341,928 outputs, rows of one chunk and of a partial last chunk, rows
starting inside a 16-byte piece (Q6_K and Q8_0 rows of 2100 and 340 bytes), matrices ending inside one, and fewer
rows than a block's warps. All equal. `mmvq_multi_parity` still passes.

## 2. Activation records (`9cff816`, `25495c3`)

At two or more columns the staged kernel was bound by shared-memory loads, most of them the activations': three to
six 4-byte loads per column and block step. The columns are now copied in as records laid out the way the format
reads them (Q6_K: an int and its block's scale, 8 bytes; Q4_K: two ints and the scale, 16 bytes, padded so a
quarter-warp's two blocks fall in different banks), so a lane takes them in one load. The scale is the float the
old load computed, so the products are unchanged. The records take 1.8x (Q6_K) or 2.7x (Q4_K) the shared memory, so
they are used only while a block stays within 32 KiB (three blocks an SM), and not at one column or for Q8_0, where
they measured no faster. The table's Q4_K and Q6_K figures at two to four columns include them (Ornith's head at
2 / 3 / 4 columns was 192 / 159 / 126 GB/s with the staged rows alone).

End to end this was inside the noise: Ornith 73.7 -> 74.0 (prose) and 86.2 -> 86.3 tok/s (code), the binary before
against the one after. Its multi-column head is about 2.7 ms of a 28 ms verify window.

## 3. Batched routing on CUDA (Guild `353f6a0`)

A verify group routed each of its tokens on its own (a BF16 router projection and a top-k launch each) unless the
router was Qwen's native 512 x 10 one. `moe_route_window` (one multi-column projection reading the weight once, one
n-token top-k, bitwise the per-token calls) existed but was compiled for gfx906 only. It now runs on CUDA too, for
Ornith's 256 x 8 router (and Drama 3's prompt windows, not measured). Alone, the top-8 of three tokens took 12.7 us
in one launch against 33.6 us in three, and the projection 11.1 against 18.1 us. `route_window_parity` now also
checks Ornith's 2048 -> 256, top-8 shape.

## 4. The fast router's sum bracket (`429b632`)

Every thread of `router_top10_fast_kernel` (Ornith's top-8, the CUDA default since `1d18a1b`) summed the warps'
exponential sums and took the bracket's two double divisions, the same values in every thread, and then all but
the first warp returned. With the GTX 1070's FP64 at 1/32 rate, those 512 divisions queued ahead of the first
warp's. Thread 0 now does it once. A single-token call of 256 experts / k 8 took 13.7-14.2 -> 11.4-11.6 us, launch
included. `router_fast_parity` (new: `tests/hip/router_fast.cpp` on the CUDA runtime) checks the fast kernel
against the block kernel bit for bit on 65,536 rows of 512 / k 10 and 8,192 of 256 / k 8, with ties, near-ties
and NaN rows, on both of its paths.

## Qwen's expert cache and its text

The GPU expert cache is sized from the VRAM left after loading, and CUDA modules load eagerly (see
[PHASE2_PAIRED_REGRESSION](../../../docs/performance/PHASE2_PAIRED_REGRESSION.md)), so every kernel in the binary
takes VRAM. The staged kernels left Qwen's cache at 114 slots (2.3 MB each); the record kernels added another
0.29 MB of device code (31.66 against 31.38 MB) and it went to 113. An expert that moves from the GPU cache to the
CPU is computed with other rounding, so Qwen's text with the new binary differs from the earlier one from token 136
(prose) and 233 (code), the same in every run of either binary. With equal slot counts (the A/Bs above) the texts
are identical.

## Tried and not kept

- One group per verify window (`GUILD_SPEC_SPLIT=0`) now that the products are faster: Ornith 65.9 / 74.7 against
  73.3 / 86.2 tok/s with the split, Qwen 28.5 / 34.5 against 29.6 / 38.2. The CPU's experts overlapping the GPU's
  attention is worth more than reading the dense weights once per window.
- Leaving Qwen's shared expert (two 2560 x 640 Q8_0 products and one 640 x 2560, 1.74 MB each) on the old kernels,
  so the router beside it gets more of the GPU: 29.6 / 38.0 against 30.0 / 38.5 tok/s with the staged ones.
- The hyper-connection read (20% of Qwen's GPU time at ~145 GB/s), all bitwise equal, all slower: the AMD up kernel
  ported to CUDA (8 lanes a row; 89.5 -> 109.8 us a read of two tokens, its `lo` reads conflict in shared memory
  here), every row's loads issued up front in the up kernel (-> 124.6 us, 79 registers), the up kernel's weights only
  (-> 118.6 us, 69 registers), the down kernel's next tiles prefetched into L2 (-> 92.0 us), the down kernel's next
  activation tile loaded into registers before the tile in use (-> 101.5 us).
- GLM-5.3-Flash's own staged Q6_K / Q8_0 kernels against these at its shapes: within 3% either way (GLM's 2% faster at
  one column, these slightly faster at two), so GLM keeps its own.

## Later

The hyper-connection read, the router, the verify head and the rest of Qwen's decode window were taken further on
2026-10-27/28 and 2026-10-30, against the memory roofline:
[2026-10-30-decode-roofline](../2026-10-30-decode-roofline/README.md).

## How to reproduce

```bash
cmake -S . -B build-cuda12 -DGUILD_ENABLE_CUDA=ON -DGUILD_EXPERIMENTAL_SM60=ON -DCMAKE_CUDA_ARCHITECTURES=61 \
      -DGUILD_CUDA_SOURCE_DIR=<guild-kernels-cuda checkout on perf/sm61-decode>
cmake --build build-cuda12 -j --target guild-generate glm5_kernels_test mmvq_staged_parity route_window_parity \
      router_fast_parity
(cd build-cuda12 && ctest -R "mmvq|route_window|router_fast")
build-cuda12/glm5_kernels_test shapes                         # GB/s at Qwen's and Ornith's shapes
GUILD_MMVQ_STAGED=0 build-cuda12/glm5_kernels_test shapes     # the kernels before
```
