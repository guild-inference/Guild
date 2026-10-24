# Decode on the dual Xeon: the NUMA row split, repacked IQ3_S rows, and drafting for Ornith

Measured 2026-10-24 on this PC: Ubuntu, two Xeon Gold 6130 (32 physical cores, 2 NUMA nodes, AVX-512 without
VNNI or VBMI), 376 GiB RAM, GTX 1070 8 GiB, driver 580.178.04, CUDA 12 / sm_61 Release build of
`guild-generate` at `968c6b5` plus this change (`-DGUILD_EXPERIMENTAL_SM60=ON`), with `guild-kernels-cpu` at
`629817c` (`GUILD_CPU_SOURCE_DIR`).

Decode rows are medians of three fresh 256-token greedy generations after a 16-token warmup, from
`tools/bench_runtime.py` on its prose and code prompts, as in
[2026-10-24-split-window](../2026-10-24-split-window/README.md): Qwen3.8-Flash-Next UD-IQ4_XS with the MTP draft,
`--spec 4`, 32768-token context, 56 GiB expert RAM budget; Ornith-1.5-35B Q4_K_M with 4096-token context, 24 GiB
budget, no drafter. Both: FP16 host-only KV, 31 pool workers + host. Each A/B is one binary with an environment
switch, and the two arms gave the same text in every pair listed.

## Where the CPU time went

A 32-thread read of 8 GiB, one thread per physical core (`micro/numa_bandwidth.c`):

| Pages | Read bandwidth |
| --- | ---: |
| interleaved over both nodes (the expert arena until now) | 125 GB/s |
| each thread's own node | 214 GB/s |

The same with 2 MiB pages: 117 and 216 GB/s.

One thread, 48 random experts from DRAM (`micro/iq3s_kernels.cpp`, `micro/iq4nl_kernels.cpp`):

| Rows | 1 token | 2 tokens | 4 tokens |
| --- | ---: | ---: | ---: |
| IQ3_S gate/up, AVX2 kernel (iq256) | 2.25 GB/s | 1.80 GB/s | 1.51 GB/s |
| IQ3_S gate/up, AVX-512 kernel (iq512) | 2.19 GB/s | 2.01 GB/s | 1.59 GB/s |
| IQ4_NL down, AVX2 kernel | 5.94 GB/s | 3.97 GB/s | 2.45 GB/s |

In weights, IQ3_S runs at about 5 G/s per core and IQ4_NL at about 11 G/s: the IQ3_S codebook lookups are the
cost: 32 cores at 2.25 GB/s is about 72 GB/s, well under what the memory gives. `GUILD_CPU_PROFILE` (which
reported zeros for the fused pipeline the decode uses; fixed here) gave Qwen's IQ3_S/IQ4_NL layers 97.9 s of
gate/up thread time against 30.7 s of down (which includes waiting for gate/up) over one 128-token run of each
prompt. The AVX-512 kernel is -3% to +12% against the AVX2 one here and rounds differently, so it was not switched
on: it is compiled with VNNI and VBMI flags and enabled only on CPUs that have them.

## The NUMA row split

The resident arena now gives each node half of every expert's rows - the first half of the gate, up and down rows
on node 0, the rest on node 1, placed by first touch from threads on each node - and the pool publishes its row
tasks in two queues, one per node; a thread takes its node's tasks, then the other node's remainder. Every row is
computed by the same kernel call as before, so outputs are bitwise unchanged. 98-100% of sampled pages were on
the intended node. It is the default on a two-node machine whose process memory policy is the default;
`GUILD_EXPERT_NUMA=interleave` (or `process`) is the old placement.

| Configuration | Prose | Code |
| --- | ---: | ---: |
| Ornith, interleaved | 50.24 tok/s | 50.07 tok/s |
| Ornith, split | **54.60 tok/s** | **54.61 tok/s** |
| Qwen, interleaved | 25.74 tok/s | 34.29 tok/s |
| Qwen, split | 25.63 tok/s | 33.69 tok/s |

Ornith's CPU dispatch fell from 6.65 to 4.96 ms per token. Qwen did not change: its rows are bound by the lookups,
not by memory. `ornith_prefill_bench` with `GUILD_NO_KV_CACHE=1`, three runs each:

| Ornith | Interleaved | Split |
| --- | ---: | ---: |
| decode after a 64-token prompt | 50.06-50.08 tok/s | **54.37-54.50 tok/s** |
| 4096-token prompt | 225.8-227.7 tok/s | 221.8-224.2 tok/s |
| decode after it | 47.99-48.03 tok/s | **51.42-51.95 tok/s** |

The prompt path's whole-expert tasks cannot be node-local under the split (211-219 tok/s), so with the split the
pool gives prompt batches row tasks instead; that is the -2% above.

## IQ3_S rows repacked for the CPU

At load, every IQ3_S expert's gate/up rows are copied once into a second, CPU-only form (`guild-kernels-cpu`
`629817c`): the grid lookup done ahead of time, one 4-bit code per value, decoded with two `pshufb` per 32 values.
The integer sums and float steps are those of the kernels the IQ3_S rows take (ggml-cpu's dot for one token, iq256
from `GUILD_IQ_MT_MIN` tokens), so the pool's outputs are bitwise those of the IQ3_S rows. The GPU cache, PCIe
offload and the prompt path keep the IQ3_S bytes. On Qwen: 47 layers, 38.44 GiB more RAM, 5.8 s more load time.
It runs only with the bundled CPU kernels, on an AVX2 CPU without the AVX-512 set the AVX-512 kernels need, when
`--resident-budget-gib` holds every expert (a smaller budget is a request to use less RAM) and 16 GiB of RAM remain
after it; the copy is locked in RAM like the arena. `GUILD_IQ3S_REPACK=0` turns it off.

One thread, 48 random experts (`micro/iq3s_repacked.cpp`): 5.40 -> 19.96 G weights/s for one token, 4.56 -> 13.27
for two, 3.72 -> 8.47 for four.

| Qwen | Prose | Code |
| --- | ---: | ---: |
| IQ3_S rows (`GUILD_IQ3S_REPACK=0`) | 25.17 tok/s | 32.68 tok/s |
| repacked | **26.21 tok/s** | **34.45 tok/s** |
| repacked, interleaved arena | 25.90 tok/s | 34.21 tok/s |
| repacked, split, second run | 25.98 tok/s | 34.20 tok/s |
| repacked, one verify group (`GUILD_SPEC_SPLIT=0`) | 25.79 tok/s | 33.33 tok/s |

CPU dispatch fell from 48-50 to 29 ms per verify window, but the window only from 82-83 to 81.3-81.5 ms: the GPU is
now the limit. A one-group window profile (`GUILD_VERIFY_PROFILE=1 GUILD_SPEC_SPLIT=0`) spent about 51 of 82 ms in
GPU kernels and the rest waiting for the CPU.

## Drafting from the context by default without an MTP drafter

`auto`, the default strategy of `guild-generate` and `guild serve`, meant MTP when a drafter is loaded and no
drafting otherwise, so Ornith never used the lookup and retrieval drafters. Without a drafter (and with drafts
allowed, so not with `--spec 0`) `auto` now runs as `adaptive`. Ornith, the bench prompts through
`guild-generate --decode-strategy` (new; also `decode_strategy=` per GEN line), three runs each:

| Strategy | Prose | Code |
| --- | ---: | ---: |
| normal | 54.86 tok/s | 54.72 tok/s |
| adaptive | 54.68 tok/s | 55.00 tok/s |
| lookup | 54.20 tok/s | 54.84 tok/s |

All nine pairs of texts were identical. `ornith_bench --prompt-len 512` (its prompt repeats one sentence):
54.02 tok/s normal, 108.74 lookup, 109.24 adaptive. Through the pipe, greedy and two sampled configurations with
fixed seeds gave the same 160 tokens with normal, adaptive and lookup, on the code prompt and on a 512-token
repeating prompt. Qwen keeps MTP for `auto`.

## An MTP drafter for Ornith

Ornith-1.5-35B's GGUF ships an MTP layer as its last block (`blk.40.*`, `blk.40.nextn.*`, `nextn_predict_layers =
1`), which the loader skips. `tools/ornith_mtp_rt.py` writes it in the drafter's runtime files - eh_proj split into
its embedding and hidden halves, the projections as Q8_0, the experts in their GGUF form with the down rows
re-quantized from Q6_K to Q8_0 for the grouped GPU expert kernels, and `data/draft_vocab.bin` as the draft head's
token subset (Ornith's tokenizer is Qwen's, byte for byte). `MtpDrafter` runs that single-stream layer when the
files' `mtp.txt` says so, and the batched prefill hands it each prompt chunk's final residuals. On this PC:
`/mnt/models-ssd/Strata-data/mtp/ornith-rt`, 587 MB of experts and 42 MB of dense weights, +794 MiB of VRAM.

`guild-generate --mtp <dir>`, three runs each:

| Ornith | Prose | Code |
| --- | ---: | ---: |
| no drafter (normal) | 54.86 tok/s | 54.72 tok/s |
| MTP, whole head | 63.97 tok/s | 65.92 tok/s |
| MTP, draft head over 106299 tokens (default) | **64.73 tok/s** | **74.71 tok/s** |
| the same, `--spec 3` | 65.94 tok/s | 74.76 tok/s |
| the same, `--spec-min-p 0.3` | 59.81 tok/s | 72.26 tok/s |
| the same, `--spec-min-p 0.7` | 64.86 tok/s | 72.96 tok/s |
| the same, `--decode-strategy adaptive` | 65.20 tok/s | 70.11 tok/s |

(The whole-head row is one 128-token run.) With the default subset 129 of 202 prose drafts and 151 of 192 code
drafts were accepted, 2.44 tokens per window on code, and drafting took 5.0 ms per window (8.3-8.5 with the whole
head). Every row's text matched the normal run. Through the pipe (`mtp_check.py`), greedy and sampled requests on
the code prompt and on a 3000-token prompt gave the same 128 tokens with MTP as with `decode_strategy=normal`,
with requests alternating between the two and with the prefix cache on and off; 128 tokens after the 3000-token
prompt took 1.9 s with the drafter against 2.5 s without.

The session set the prefill engine's chunk callback only for requests with drafts, and the callback held that
request's token list: a request without drafts after one with them would have run the drafter's prompt pass over
freed memory. The callbacks are now set or cleared for every request, and cleared after it.

## The engine's exit with its pipe open

The serve loop's stdin thread (added in `b9cff45` today) read `std::cin`; `exit()` takes every stdio stream's lock to
flush it, and the blocked reader held stdin's, so an engine sent `QUIT` by a client that kept the pipe open never
exited (Ornith: still running after 300 s; 2.2 s with the pipe closed). It now reads file descriptor 0 directly and
exits in 2.2 s either way.

## Checked

- `pool_numa_test` (new): random batches of 1-40 experts and 1-4 tokens through the fused and two-phase pool
  paths, with and without the repacked rows, split on and off, bitwise equal to the row kernels called directly.
- `iq3s_nl4_exact` (new, `guild-kernels-cpu`): the repacked rows against ggml-cpu's dot and iq256 for 1-8 tokens,
  also with `GUILD_FORCE_AVX2=1`.
- The full CTest suite: 102 of 102 passed with `ple_parity` excluded (its Q2_0 fixtures are absent) and four
  hardware skips as before; the Ornith prefill, Qwen parity and pool tests again after the prompt-path change; and
  103 of 103 (with `pool_numa_test`) after the decode-strategy change and again with the Ornith drafter; Qwen's
  decode after that change: 25.88 / 34.07 tok/s with the same texts and draft counts as before it.
- `expert_source.cpp` and `pool.cpp` compile against the pinned `guild-kernels-cpu` revision, without the repack
  (`GUILD_CPU_HAS_IQ3S_NL4`).

## Looked at and not done

| Idea | Measurement | Verdict |
| --- | --- | --- |
| Refreshing the GPU expert cache at run time | Qwen's cache holds 58 of 24576 experts; Ornith's formats have no grouped GPU kernel | Nothing to gain on this card. |
| Expert budgets in the verify window (MoE-Spec, AcceptMoE) or expert deferral (KTransformers) | - | They change the target's arithmetic, so a draft would no longer be accepted exactly when greedy decode produces it. Not done. |

## Reproduce

From `/home/ubuntu/Guild` with the API server stopped, the commands of the split-window report, with
`GUILD_EXPERT_NUMA=interleave` or `GUILD_IQ3S_REPACK=0` for the controls. Ornith's drafter:

```bash
.venv/bin/python tools/ornith_mtp_rt.py --gguf /home/ubuntu/models/ornith-1.5-35b/Ornith-1.5-35B-Q4_K_M.gguf \
  --out /mnt/models-ssd/Strata-data/mtp/ornith-rt
.venv/bin/python tools/bench_runtime.py --exe <build>/guild-generate \
  --model /home/ubuntu/models/ornith-1.5-35b/Ornith-1.5-35B-Q4_K_M.gguf \
  --pack /mnt/models-ssd/Strata-data/packs/ornith-1.5-35b --context 4096 --resident-budget-gib 24 \
  --mtp /mnt/models-ssd/Strata-data/mtp/ornith-rt --runs 3 --max-new 256 --out ornith-mtp.json
```

Raw reports are in `raw/`.
