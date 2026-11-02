# Decode, seventh round: GLM-5.3-Flash's draft, and what limits both models now

Two bit-exact GLM-5.3-Flash changes shorten the draft that starts every step (5.1 -> 3.6 ms with `--dense q4_k`)
and spread the CPU pool's reads (median step 74.0 -> 71.6 ms with the draft head's copy). The rest of the round
measured where the time goes: GLM's CPU and GPU halves are even again, the CPU's gate/up rows stream at the
memory's rate and its down rows at the IQ4_XS kernel's compute rate, and the levers tried on top of that (fewer
bytes, other item shapes, cached reuse, other kernels) did not pay.

PC: 2x Xeon Gold 6130 (16 cores each, AVX-512 BW without VNNI, 2 NUMA nodes), GTX 1070 8 GB (sm_61, 15 SMs, 151 W
cap), CUDA 12. Branch `feat/nemotron-transcribe` (`42fd6cf`), guild-kernels-cpu `9b7281a`, guild-kernels-cuda
`de65d60` (all local).

## GLM-5.3-Flash

`glm5_run --spec` from the 5-token prompt at the server's `--ctx 262144`, arms in one process (`glm5_run --ab`, from
a snapshot after the prompt), the ids compared in every run.

| Change | Commit | Result |
| --- | --- | --- |
| The draft's copy of the output head as Q4_K (340 MiB; only where the drafts cannot change the output); the CPU experts' rows added by the post-mix kernel itself; the pool's items spread | `cdf95cf` | `--q4k`, 512 tokens, 4 runs a side: median step 74.0 -> 71.6 ms, 26.84 -> 27.91 tok/s; the draft 5.1 -> 4.4 ms |
| A draft's NextN pass computes only the row it reads | `42fd6cf` | `--q4k`, 768 tokens, 4 rounds: the draft 4.4 -> 3.6 ms, median step 71.7 -> 71.1 ms (27.85 -> 27.92 tok/s); Q6_K attention, 2 rounds: the draft 5.2 -> 4.5 ms, 82.1 -> 81.3 ms |

The same ids in every run, and the same accepted drafts; today's configuration (Q6_K attention) gives the same 768
ids as before the round.

- **The pool's items spread.** A node's threads take items in queue order, so neighbours in the queue run at once.
  Group by group, sixteen threads read sixteen adjacent row blocks of one expert, which DRAM serves slower: plain
  reads by 32 threads of 137 KiB items ran at 193 GB/s side by side and 214-216 spread, 70 KiB items at 175 against
  208. Gate/up items now take the experts in turn, an expert's blocks from its two halves alternately; down items
  start at their own expert and spread their row blocks, and their sums stay in expert order.
- **A draft's NextN pass.** When a draft is accepted, the next draft runs the NextN block on two tokens: the kept
  draft (its catch-up) and the new token. Only the last row's FFN is read, by the draft head; a catch-up row is
  there for the keys and values its attention stores. The CPU now computes the routed experts of the last row's
  token alone, in the order both tokens give the experts, so that row is the 2-token pass's to the bit, and a
  catch-up pass (after a prompt, or a draft not followed by another) skips the FFN. `GUILD_GLM_MTP_FFN_ALL=1`
  computes every row.
- **`glm5_run --ab` alternates the arms' order** by round (a b, b a, ...). In one process a decode's verify time
  drifted by up to 3 ms over a run whatever the arm, which a fixed order charged to the second arm; separate
  processes of one configuration differ by 2 ms a step (68.6 against 70.6).

### Where the CPU's time goes

The pool with its item times split by kind (`--q4k`, 768 tokens, prompt and decode): gate/up items read 2,575 GB at
202 GB/s while busy, down items 1,356 GB at 162 GB/s (3.5% of their time waiting for an expert's hidden rows), and
the threads are idle 6% of the phases. The phases themselves cost nothing: plain reads in the pool's order (the
gate/up items spread over 8 experts, then down items over all of them, ~490 us phases between barriers) run at
214-215 GB/s, as one stream a thread does.

One token against each kernel, 32 threads streaming node-local rows (warm huge pages), GB/s of the rows' own bytes:

| Rows | 32 threads | 1 thread |
| --- | ---: | ---: |
| NL4 gate/up (IQ3_S repacked; 4,096 values, 2,144 B) | 213-216 | 12.4 |
| IQ4_XS down (2,048 values, 1,088 B), `iq4xs_rows512` | 168-208 | 7.6-8.5 |
| IQ4_XS down, ggml-cpu's dot | 150-151 | 6.0-6.5 |
| Q8_0 (4,096 values), `q8_0_rows512` | 218 | 10.1-12.5 |
| Q4_K, ggml-cpu's dot | 172-192 | 7.4-7.9 |
| Q4_K, guild-kernels-cpu's `kq256_rows` | 139-140 | 4.9-5.7 |
| IQ3_S (1,760 B), guild-kernels-cpu's AVX-512 rows | 22-31 | 1.1-1.5 |

The gate/up rows stream at the memory's rate. The down rows are compute-bound at the all-core AVX-512 clock: 5.3-6.5
GB/s a thread where 214 GB/s needs 6.7; the other layout tried (below) did worse.

### Where the GPU's idle goes

nsys of round 6's `--q4k` build (27 steps of 71.3 ms): 7.98 ms a step without a kernel. 2.77 ms of it before the adds
of CPU rows (the dense layers 0-2, and 1.4-1.7 ms in the draft waiting for the NextN layer's experts, the wait
`42fd6cf` shortens), 1.61 ms before post-mixes (an expert phase late), 1.07 ms between the verify head and the next
draft (the host copies two rows of logits, 1.24 MB, takes their argmax and accepts).

### Tried, not kept

| What | Result |
| --- | --- |
| The dense layers' FFNs (on the CPU) as Q4_K with `--dense q4_k` | KL (text 1 / text 2) 0.052 -> 0.053 and 0.063 -> 0.066, top-1 90.0 -> 89.9% and 90.2 -> 89.5%; speed within the run-to-run spread (median step 71.0 / 68.0 ms against 70.6 / 68.6, separate processes) |
| Down items of one expert's rows, a block's sums by its last item (128 / 256 / 64 rows) | median step 74.9 / 74.0 / 79.2 ms against 75.7 (one process, 2 runs each), the same ids |
| Group B reading first the experts it shares with group A, from the caches (a probe of the pool's phases, plain reads) | group B's phases 1.4-3.6% faster with 3 of 8 experts shared, 4-7% with all 8: the caches keep little of a 47 MB phase a node |
| IQ3_S gate/up rows read as they are, 18% fewer bytes than NL4 | 1.1-1.5 GB/s a thread (table above): the grid lookups are gathers, and this CPU runs the Downfall microcode mitigation (`gather_data_sampling: Mitigation: Microcode`) |
| An IQ4_XS kernel on activations permuted per 32-value sub-block (fewer shuffles a byte) | bitwise ggml's, but 4.4 GB/s a thread against 8.3: the per-block scale decode and lane folds cost more than the shuffles saved |
| 3-token windows (a second chained draft) | the second draft is accepted 85% of the time (the first 95%), but the window's verify took 85-92 ms against 63 for two tokens: its groups are one token and two |
| The route kernel over sorted candidates; the router and the route as one kernel | 10.7 us against 8.9; 0.5 us a call saved |
| The verify head (Q6_K, two rows) in native_mmvq | 161 GB/s against the current kernel's 169 |

## Qwen3.8-Flash-Next

Measured, unchanged this round: two row groups, prose 55.57 ms a window (40.06 tok/s), code 59.06 ms (50.40 tok/s);
one row group 61.70 / 68.15 ms (36.08 / 43.68 tok/s).

- **The GPU's busy time** (nsys, prose and code requests): a kernel runs 82.6% of the decode span, `wait_flag` spins
  included. One gap of ~9 ms a window is nsys's own (the verify graph's launch under the profiler); the rest is
  ~2.7 ms a window, plus 2.2 ms of `wait_flag` spinning on the CPU's experts.
- **The hyper-connection reads** (~13 ms a window of kernel time) overlap the dense products for about half their
  time (602 of 1,263 ms of `gr_down_q8`), so their slower in-engine times are partly the bandwidth they share.
- **One row group** stays slower: its CPU phase is compute-bound (546 us a layer for 24 experts, the threads 90%
  busy).
- **The draft head** is already a Q4_K copy (146 MiB); a draft pass is ~1.5 ms of kernels, the head 0.74 ms of a
  pass that drafts.

## What is left

- **GLM's halves are even and near their limits:** the CPU's experts 57-61 ms a step (gate/up rows at the memory's
  rate, down rows at the IQ4_XS kernel's), the GPU's large products at 175-209 GB/s. Outside them a step has the
  draft (3.6 ms), the verify head (3.1 ms) and the turnaround (~1 ms).
- **A larger step needs the computation to change** (fewer experts a token, or experts deferred to the next layer
  as Qwen's `--defer-experts` does), not measured on GLM, **or gathers without the microcode mitigation**, which
  would let the CPU read IQ3_S rows: 1.1 GB fewer bytes a step.
- **The turnaround**: an argmax on the GPU would save the 1.24 MB copy and the host's argmax for greedy requests
  without penalties.
