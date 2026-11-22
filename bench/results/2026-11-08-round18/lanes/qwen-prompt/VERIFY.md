# qwen-prompt (round 18): independent verification

Verifier pass, 2026-11-08. Lane branch `lane18/qwen-prompt` at HEAD `1f7ed75` (8 commits on perf/next `a10d867`);
the kernel worktree `gkc/` has no commits past `de65d60` and no changes, so there is no lock bump.

**Verdict: A, C, D accepted. B accepted as opt-in code only (off by default, as the lane left it). E rejected.**
With its default on, E changes the output of every hc = 1 model that runs the verifier's head. Measured: Ornith's
token ids change, and Drama 3's audio changes and never reaches end of speech. A, C and D are exact and as fast as
claimed on Qwen. E must be reverted or limited to hc > 1 before this branch merges.

| Item | Verdict | Exactness | My A/B on the GTX 1070 |
| --- | --- | --- | --- |
| A. Head-less ingest windows | accept | hashes = base, every prompt | p100 / p200 / p400: -9.8 / -9.2 / -9.1 %, about 7.1 ms per 8-token window |
| B. Shared expert beside the host grouping | accept as opt-in | default-off path = perf/next's GPU order | not re-run (not on by default) |
| C. 32-slot streamed walk under --prefill-rows, HC buffers in the region | accept | hashes = base at 7.6K / 4K / 32K | 7.6K: 27.26 -> 25.05 s (-8.1 %) |
| D. Block scores per 8 queries from 16K | accept | hash = base at 32K | 32K: -0.80 s (-0.75 %), 2 pairs, same sign in both |
| E. Head's fused HC read on by default | **reject** | **Ornith ids and Drama 3 WAV change** | n/a |

## Setup

- Lane build: `/home/ubuntu/lanes/r18/qwen-prompt/build`. The rebuild under build-run had nothing to do (the
  sources have not changed since build6). `guild-generate` sha256 `1d6fed66...` = the lane's `ab/gg-g` (its final
  binary). Base: `/home/ubuntu/lanes/r18/base/build/guild-generate`, sha256 `db36d21d...`.
- Every GPU command ran inside one `gpu-run bash verify/batch.sh` hold, from 13:17:26 to 13:53:13 UTC (36 min).
  `verify/qb`, `verify/ob` and `verify/db` are qwen-bench-r18, ornith-bench-r18 and drama3-run-r18 with only
  `gpu-run` taken out (checked with diff). The outputs are in `verify/out/`, the timeline in `verify/batch.log` and the
  tests in `verify/ctest.log`.
- One slip: after the batch I re-ran `ctest -R '^ple_parity$' -V` once outside the lock to read its error. It
  stopped at the missing fixture after 0.01 s.

## E: rejected (it breaks hc = 1 models)

**The bug, from reading the code.** `src/core/verify.cpp` (lane line 1558) sets
`fuse_head_gr = fuse_head_gr_env && le_ == n_layers && head_ loaded && !cvec covers` and never checks `g.hc`.
When it is true, `post()` at the last layer (line ~1705) skips the whole `if (!fuse_head_gr) {...}` block. For
hc = 1 that block is the residual add `add_inplace(Rt(t), bo_ + t*N)`. The head's hc = 1 branch (line ~1789) then
RMS-norms `Rt(t)` as it is: the fused read (`fused_gr_read_multi`) only runs under `if (g.hc > 1)`. So for
hc = 1, the head reads a residual that is missing the last layer's MoE output. Every model goes through this path:
the native head loads whenever the model has shards (model.cpp ~757). perf/next had the same latent bug, but there
it was opt-in (`GUILD_FUSE_HEAD_GR=1`).

**Measured** (same lane binary, default vs `GUILD_FUSE_HEAD_GR=0`):

| Model | Lane default | `GUILD_FUSE_HEAD_GR=0` | perf/next (BRIEF) |
| --- | --- | --- | --- |
| Ornith-1.5-35B, ornith-bench (2 runs x prose + code x 256) | ids [79308, 79308, 40889, 40889], 77.9 / 92.5 tok/s | ids [30936, 30936, 65886, 65886], 80.4 / 93.3 tok/s | [30936, 30936, 65886, 65886] |
| Drama 3, seed 11 | md5 1948153f, 2000 frames with no end of speech (92.9 s of audio, hit --max-new) | md5 6e04c9b1, 479 frames (eos), 46.38 frames/s | 6e04c9b1 |

Labels: `vo_L1` / `vo_E0`, `vd_L11` / `vd_E011`. The Ornith text is still fluent but different. StartLux-Decision
was not run (to keep the lock time down). It is hc = 1 and goes through the same verifier head (`session.score`
-> `ver->run(1, ...)` -> `copy_logits`), so its answer probabilities should change the same way.

On Qwen (hc 4), E is exact and does speed up the prelude:
- My decode run `vQ_1` (default bench, E on) gives ids [38747, 38747, 41916, 41916] at 49.4 / 61.9 tok/s.
- `tools/prelude.py` on the lane's `ab/nE1.sqlite` vs the scout's `runs/N.sqlite`, median prelude in us for
  T = 1..8: 79 81 91 99 138 153 166 186 vs 140 239 356 485 596 713 821 943. This matches the lane's table.

**The fix belongs to the lane or the user.** One way is to add `g.hc > 1 &&` to the verify.cpp condition. The
mtp.cpp half of 7548495 only touches `record_forward`, the hc > 1 drafter: hc = 1 drafters use
`record_forward_single`, as `native_ == (g.hc == 1)` shows. After that, re-run Ornith and Drama 3 to show the bits
are the same. The lane's note "affects only hc > 1 models" is not what the code does, and the lane ran no other
model.

## A: accepted

Code review (verify.hpp/.cpp, session.cpp):
- **What the head-less window skips.** `capture(T, err, true)` records the same `record_window` with `no_head_`
  forced on and puts it back afterwards. The only `no_head_` branch inside `record_window` is after the head
  prelude (line 1804), and nothing that changes state comes after it except the head product, the finite check and
  the sampler. So R_ (`final_R_all`, read by the MTP prefill), `head_mixed_`, the K/V, the GDN/conv state, the PLE
  history and the DFlash features are all written as before.
- **Per-T state written while recording.** `groups_[T]`, `host_combine_[T]` and `defer_[T]` are recomputed from
  the same inputs, so they get the same values.
- **Every place the window graphs are dropped also drops the head-less ones.** That covers `reset_graphs`, the
  destructor, `configure_expert_cache` and `set_feature_layers`.
- **The flag lasts one run.** `ingest_` is cleared at the top of `run` and only applies at the stage that owns the
  head. A layer split's first stage passes it to `next_` (not run here: one GPU).
- **What reads the picks.** `session.cpp` discards `sink_out`; `commit`, `mtp->prefill` and `dflash->add_context`
  read state the layers write. `score()` runs its final 1-token window with the full graph.
- **Window sizes.** `set_ingest(T == S)`: only the 8-token head-less graph is captured. The log shows one
  "head-less ingest window" capture.

My pair (`vA1_1` on, `vA0_1` `GUILD_INGEST_NO_HEAD=0`; short.json, GUILD_NO_KV_CACHE=1, --max-new 32, --warmup 1):

| prompt | on (ms) | off (ms) | change | hash (both = base_sh) |
| --- | --- | --- | --- | --- |
| p100 | 869 | 963 | -9.8 % | 10706 |
| p200 | 1716 | 1890 | -9.2 % | 74277 |
| p400 | 3454 | 3800 | -9.1 % (346 ms over 49 full windows: 7.1 ms each) | 60685 |
| p800 | 6803 | 6802 | 0 (batched path) | 12302 |
| p1600 | 9029 | 9056 | -0.3 % (batched path) | 73032 |

gpu_after is 7585 vs 7559 MiB (+26 MiB, one graph). The stop rule (>= 5 ms per window) passes.

Caveats, none of which change bits:
1. Head-less windows skip `validate_logits`, so a non-finite state during the ingest is reported at the first
   decode window, with that window's position.
2. Once set, `nohead_off_` stays set: `reset_graphs` does not clear it.
3. The fallback when a head-less capture fails was never run.
4. These numbers were taken with E on. On Qwen, E changes only the prelude, which the head-less window keeps.

## B: accepted as opt-in only

With `GUILD_PREFILL_SHEXP_EARLY` unset (the default), the shared expert is the same lambda body, and it is
enqueued before any other GPU work after `route()`. The only code that now runs before it computes host-side bools
(`fused_l`, `use_mmq`, `shexp_early`) and enqueues nothing, so the GPU stream order is perf/next's. The
`ids_ready` event is created but not used. Opt-in mode: the event comes after the ids' copy, and the shared
expert's buffers (sgate, sup, sh_h, shared, sg) are their own takes in the MoE set. The lane's se1/se0 hashes were
identical. I did not measure it: its gain (-0.6 to -0.9 %) failed the lane's own 1 % rule, and it is off by default.

## C: accepted

Code review (prefill.cpp):
- **The ring rule.** `rows_stream()` is true only for cc < 7, `g_rows` in (0, stream_all_min) and not
  fused_ring. Then the ring has 32 slots from T >= 4096 and STAGE below that. `stream_all` and `bind_stage_helper`
  use the same threshold (`stream_chunk_min`). The routed walk always stages with `% STAGE` (line ~3748), so a
  smaller last chunk in a 32-slot layout runs as before. The model.cpp chunk scan calls `set_rows(512)` before
  `bytes_needed`, so the scan counts the same ring that carve takes. `GUILD_PREFILL_STREAM_ROWS=0` makes
  `rows_stream()` false, which restores the old `ring_slots`, `stream_all` and helper tests exactly.
- **The carve.** `bytes_needed` and `carve` take buffers in the same order outside the region: emb, xn16, lo and
  lo16 are skipped in both. Both size the region with `T*D*4 + hc_set_bytes(T)`. Inside the region, the four buffers
  start at offset T*D*4, after `gated`, and do not overlap it or each other. The ring slots, `ple_emb`, R, bo, inj,
  grs and mixed stay outside the region.
- **Liveness.** I traced it myself:
  - xn16, lo and lo16 are live only within one HC read, apart from F-2's xn16.
  - F-2 (`gr_write_norm_rs`) writes xn16 after the combine on `m.cs`. It is off before layer 1's PLE block when PLE
    is on, at the last half of a stage and under a control vector.
  - `draft_kv` runs inside `on_chunk`, after a stream sync, on `m.cs`.
  - emb is written by the chunk's embedding and read only by `gr_broadcast`.
  - The copy stream writes only ring slots.
- **Not run here.** The layer-split, peer, BF16X2 and GR_UNFUSED paths were checked by reading only. BF16X2's
  `*_lo` buffers and GR_UNFUSED's `xn` stay outside the region, and F-2 is off under GR_UNFUSED.

Measured (GUILD_NO_KV_CACHE=1, --max-new 32, --warmup 1, w16 hash 86768 everywhere):

| run | prompt | lane (ms) | base (ms) | change | hash |
| --- | --- | --- | --- | --- | --- |
| vC1_1 / vCb_1 (pD) | 7,600 | 25054 | 27259 | -8.1 % | 77730 both |
| pB, lane mean of 2 with D on / base vDb_1 | 31,918 | 105674 | 116564 | -9.3 % (C+D) | 3242 all |
| pB, lane mean of 2 with D off / base | 31,918 | 106476 | 116564 | -8.7 % (C alone) | 3242 all |
| pB | 4,008 | 16482-16572 | 16539 | none (below 4096) | 59650 all |

Every arm logs "chunks of 8192 tokens". gpu_after is 7585 MiB (lane) vs 7727 MiB (base).

Not measured by me or the lane: one-chunk prompts of 4,096-7,599 tokens, and pC (1-2K). The lane's open G6 point
(+1.5 % at 1K) was not reproduced on its final binary and I did not re-test it.

## D: accepted

I read `gkc/src/attention/qsa_select.cu`. For each (block, query) `block_scores_multi_kernel` does what
`block_scores_kernel` does: the same float4 key (dead for b == n_bid), the same per-head dot expression and
xor-shuffle sum, relu and sum, and the same +1e9 tail rule, written to `out[q*max_blocks + b]` for
b <= n_bid < max_blocks. The caller passes `active = last query's n_bid + 1`, which is at least every query's
n_bid + 1, so both kernels write the same set of cells. The top-k is called with the same `active` in both arms.
The per-8 offsets are right: `q_idx + (t0+q0)*512`, `steps0 + q0*kStepCount` and `sel_scores + q0*max_blocks`.
With `active_blocks = 0` and nq <= 8 the call goes to the multi kernel. `GUILD_SCORES_MULTI=0` would send it to the
warp kernel over max_blocks, which gives the same values. `GUILD_PREFILL_SEL_MULTI=0` restores the old call.

Two alternated pairs on pB (`vD1_1` / `vD0_1`, `vD0_2` / `vD1_2`; `GUILD_PREFILL_SEL_MULTI=0` for off):

| pair | 32K on (ms) | 32K off (ms) | change |
| --- | --- | --- | --- |
| 1 | 105230 | 106263 | -1.03 s |
| 2 | 106118 | 106688 | -0.57 s |

The mean is -0.80 s (-0.75 %), with hash 3242 in all four. At 4K D is not active (16482 / 16560 on,
16572 / 16563 off). This agrees with the lane's -0.64 s and its phase check (qsa select 4805 -> 4186 ms).

## Tests (lane build, under the lock)

I ran `ctest -R 'qsa|dflash|ple|host_combine|shared_expert|gr_parity|sampler|decode_cluster|route_window|coupled_draft|^parity_test|tokenizer_parity|finite_output|runtime_inference_failure|prefill|native_expert_parity|mmq|expert_parity|verify'`.
That selects 37 tests:
- **34 passed:** qsa_parity, qsa_topk_parity, qsa_topk_active_parity, decode_cluster_parity, gr_parity,
  gr_parity_staged, gr_parity_staged_pf, route_window_parity, dflash_kernels_test, coupled_draft_test, parity_test,
  finite_output_test, runtime_inference_failure_test, prefill_failure_test, ornith_prefill_test, host_combine_parity,
  shared_expert_parity, sampler_parity*, native_expert_parity_* and the others.
- **2 skipped:** prefill_fused_moe_test and prefill_fused_iq_test, which do not apply to this card.
- **1 failed: ple_parity**, because its fixture `../../Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf`
  is not on this machine. This is the environment, as the lane found.

None of these tests runs Ornith or Drama 3 through the verifier's head, which is why E's defect got past them.

## Other models whose paths the lane touched

- **prefill.cpp (B, C, D)** runs only for the Qwen3.8-Flash-Next geometry: `Prefill::init` declines when
  `n_embd`, `hc` or `hc_lr` differ, and the model then has no `m_impl->prefill`.
- **verify.cpp and session.cpp (A, E)** run for every model on the runtime. Ornith and Drama 3 were run above. With
  E off, both match perf/next bit for bit, and neither took a head-less window: Ornith reads its prompts through the
  batched prefill, and Drama 3's verifier is `no_head_`.
- **Not touched:** GLM (`glm5.cpp`) and Nemotron (`nemotron_h`) use neither file.
