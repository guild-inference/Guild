# qwen-prompt lane (round 18) - notes

Lane: /home/ubuntu/lanes/r18/qwen-prompt (guild worktree on lane18/qwen-prompt from perf/next a10d867).
Base arm: /home/ubuntu/lanes/r18/base/build/guild-generate.
Assignment: judge.json lane "qwen-prompt", items A (head-less ingest windows), B (shared expert beside host grouping),
C (streamed walk 32-slot ring for big --prefill-rows chunks), D (optional, indexer block scores), E (fused head mix
default on, decode).

## Log

- 06:39 UTC: lane set up, initial build started.
- A implemented (verify.hpp/verify.cpp/session.cpp): exec_nohead_[T] captured lazily with no_head_ forced for the
  capture; Verifier::set_ingest(true) arms the next run only (cleared at entry); a layer split's first stage hands it
  to the next stage's run.  session.cpp's ingest loop sets it before each window.  GUILD_INGEST_NO_HEAD=0 = old.
- B implemented (prefill.cpp): shared expert in a lambda; on the host-grouping path (not fused, not
  GUILD_PREFILL_TIMING) record m.ids_ready after the ids' copy, enqueue the shared expert, cudaEventSynchronize.
  GUILD_PREFILL_SHEXP_EARLY=0 = old order.
- Tests (lane build at a2a9b4c, A+B), under gpu-run: ctest -R 'verify|route_window|dflash|gr_parity|qwen|mmq|
  expert_parity|prefill|ple|parity_test|finite_output|runtime_inference_failure|host_combine': 32/33 pass, 2 skipped
  (prefill_fused_*: not this card), ple_parity fails in 0.01 s: its fixture bench/micro/ple_in.bin does not exist in
  any tree here (environment, not code).
- Commits: 1ab5f95 (A), a2a9b4c (B).
- A/B chain for A and B (ab/chainAB2.sh) on a snapshot of that binary (ab/gg-ab, sha256 6309d550...), so the lane
  build can move on.  First A run nh1_1: p100 871 ms, p200 1761, p400 3493, p800 6721, p1600 8862 (hash per prompt
  10706 74277 60685 12302 73032).
- C implemented (prefill.cpp): rows_stream() (GUILD_PREFILL_STREAM_ROWS, Pascal only, g_rows in (0, stream_all_min),
  !fused_ring): ring_slots from the chunk's own T (>= 4096: 32 slots, GUILD_PREFILL_RING overrides; else STAGE);
  stream_chunk_min() for run_impl's stream_all and bind_stage_helper.  Carve: xn16, emb, lo, lo16 in the region at
  offset T*D*4 (hc_set_bytes), GUILD_PREFILL_HC_REGION=0 = own buffers; bytes_needed matches (RING_BYTES=0 keeps its
  over-count).
- E implemented: GUILD_FUSE_HEAD_GR default on in verify.cpp and mtp.cpp.
- A pair 1 (gg-ab, short.json, GUILD_NO_KV_CACHE=1, --max-new 32): nh1 vs nh0 (GUILD_INGEST_NO_HEAD=0):
  p100 871 / 957 ms, p200 1761 / 1887, p400 3493 / 3793 (-300 ms over 50 windows = 6.0 ms a window), p800 6721 /
  6695, p1600 8862 / 8822 (prompt path: unchanged); hashes identical per prompt (10706 74277 60685 12302 73032).
- D implemented (prefill.cpp): from position 16384 on, block scores per 8 queries with no active count (the multi
  kernel); GUILD_PREFILL_SEL_MULTI=0 = the warp kernel's grid.  Chains CE/N stopped while waiting, to be restarted on
  a binary with D (C/E runs not started yet).
- A done, 2 pairs (gg-ab): prompt_ms nh1 (on) / nh0 (GUILD_INGEST_NO_HEAD=0):
  p100 871, 879 / 957, 959 (-8.7%); p200 1761, 1767 / 1887, 1906 (-7.0%); p400 3493, 3487 / 3793, 3839 (-8.5%,
  6.5 ms a window of 8); p800 6721, 6729 / 6695, 6629 and p1600 8862, 8945 / 8822, 8793 (prompt path, A not in it:
  +1%, the arm order).  Hashes identical in all four runs per prompt.  Stop rule (>= 5 ms a window): passes. KEEP.
- B done, 2 pairs (gg-ab, pC.json): se1 (on) / se0 (GUILD_PREFILL_SHEXP_EARLY=0): 1K 7339, 7297 / 7425, 7305
  (-0.6%); 2K 10167, 10068 / 10279, 10149 (-0.9%); 4K 16026, 15912 / 16127, 16080 (-0.84%).  Hashes identical
  (86557 81685 59650, = the scout's base runs b-C0/b-C).  Stop rule (4K >= 1%): FAILS -> B left off by default
  (GUILD_PREFILL_SHEXP_EARLY=1 opts in).  Expected 0.4 s at 4K; measured 0.135 s: the shared expert at 4K is
  ~3 ms a MoE layer, which is all the host wait can hide.
- base lane on short.json (base_sh): p100 963, p200 1902, p400 3825, p800 6800, p1600 9073 ms; hashes 10706 74277
  60685 12302 73032 = the lane's in every run.
- The lock waits grew to 20-35 min a run, so the rest runs in two groups under one gpu-run each (tools/qb = the
  helper without its own lock): G1 = B pair 2, base_sh, C on pD (lane with B on vs base, 2 pairs), ctest, E ids
  pair, nsys; G4 = C/D on pB (lane B off vs base, 2 pairs), pD pair with B off, timing runs, pC, ring rule off.
- C on one 7,600-token chunk (G1; lane gg-c with B still on vs base, 2 pairs, GUILD_NO_KV_CACHE=1): lane 25053,
  24983 ms / base 27172, 27235 -> -8.0%; hash 77730 in all four (= the scout's).  'chunks of 8192 tokens' in both;
  gpu_after 7613 MiB (lane) vs 7727 MiB (base).
- ctest (lane build A-E+D, t2.log): the same 32 of 33 pass, ple_parity fails on its missing fixture, 2 skipped.
- E (gg-c, default bench, 2 runs x prose + code x 256): fh1 (on) prose 49.2 / code 61.8 tok/s, ms/window prose
  57.42, 56.91, code 63.56, 63.96; fh0 (GUILD_FUSE_HEAD_GR=0) 48.8 / 59.9, 58.05, 57.34 / 65.61, 65.91.  ids
  [38747, 38747, 41916, 41916] in both (= perf/next).  The tok/s pair cannot resolve E (the code windows' CPU dispatch
  moved 38-40 -> 41-42 ms between them).
- E nsys (gg-c, 1 run, 64 tokens; the tokens are the bench's first 64): head prelude (end of the last combine to the
  head's quantize), median by window size T=1..8: 79 81 91 99 138 153 166 186 us; the base capture (scout N.sqlite,
  the per-row path) 140 239 356 485 596 713 821 943 us (tools/prelude.py on both).  Confirms the predicted drop. KEEP.
- 10:13 UTC (taking over after the usage-limit stop): G4 still waiting for the lock (queued 09:35).  gg-c = the
  binary with A-E+D (B on by default, so G4 forces GUILD_PREFILL_SHEXP_EARLY=0 = HEAD's default).  Reviewed C's carve
  liveness: F-2's fused norm (the only xn16 write outside the HC read) is off before layer 1's PLE block and at a
  stage's last half, so xn16 is not live across the PLE sub-batches or draft_kv (both carve the region from 0);
  gr_write_norm_rs reads bo/inj (outside the region); emb dies at the chunk's broadcast; lo/lo16 live only norm..mix;
  the copy stream writes ring slots (not the region), the peer / CPU helper join m.cs before the combine; GR_UNFUSED's
  xn and BF16X2's *_lo buffers stay outside.  Format check passes.  Queued G6 (ab/G6.sh): ctest on the final build
  (t3.log; t2 ran on a build with B still on), E pair 2 on gg-f (= build/guild-generate at HEAD adddaf4, sha256
  fcebadc5...), order fh0_2 then fh1_2.
- G4 done (10:16-10:47; gg-c, lane arms with GUILD_PREFILL_SHEXP_EARLY=0 = HEAD's default, GUILD_NO_KV_CACHE=1):
  pB (4K + 32K) lane cB1 / base cBb, 2 pairs: 4,008 tokens 16459, 16589 / 16531, 16545 (C not in it: below 4096);
  31,918 tokens 104856, 105677 / 116369, 116245 ms (-9.5%, C + D).  pD 7,600 with B off: 25164 / 27288 (-7.8%).
  Hashes: 86768 (w16), 59650 (4K), 3242 (32K), 77730 (7.6K) in every run, lane and base.  'chunks of 8192 tokens'
  in both.  Timing runs at 32K: D on (cB1_t) 105952 ms, qsa select 4186 ms, wait copy 2722; D off (cBd_t,
  GUILD_PREFILL_SEL_MULTI=0) 106978, qsa select 4805, wait copy 2735; base (scout b-A) 115811, qsa select 4792, wait
  copy 12503.  So D: -0.62 s on its phase (predicted ~0.8 s); C: wait copy -9.8 s.  Ring rule off (cD0_1,
  GUILD_PREFILL_STREAM_ROWS=0, carve on) 7.6K 27271 ms = base: the carve is speed-neutral, the ring rule is all of
  C's gain.  pC with C (cC1_1) 7438 / 10344 / 16234 vs se0 (other session) 7425,7305 / 10279,10149 / 16127,16080:
  one run, not alternated -> G6 adds 2 alternated lane/base pairs on pC and D's own 2 pairs at 32K.
- D's exactness read in gkc: block_scores_multi_kernel computes per (block, query) the same float4 dot per head, the
  same xor-shuffle sum, relu, the +1e9 tail rule, and writes out[q * max_blocks + b] for b <= n_bid, as
  block_scores_kernel does; the caller's offsets (q_idx + (t0+q0)*512, steps0 + q0*kStepCount, sel_scores +
  q0*max_blocks) put each query's scores in the cells the one-grid call writes; the top-k reads b <= n_bid only.
- 11:10 A's VRAM: gpu_after 7805 MiB with A (nh1_1, nh1_2) vs 7727 without (nh0_1, nh0_2, base_sh), the same binary:
  +78 MiB for the 3 head-less graphs short.json captured (T = 3, 7, 8: tails of 99/199/399 ingested tokens), ~26 MiB
  a graph.  A server meeting every tail size would hold 8 more (~208 MiB) next to decode's 8, out of the load's
  384 MiB reserve for the window graphs (model.cpp: 600 MiB free already failed instantiating them), and a failed
  capture failed the request.  Refinement (uncommitted until G7 checks it): session.cpp arms set_ingest only for
  full windows (T == S), the tail runs the full graph decode captures anyway (one head a prompt, ~3-6 ms); run()
  falls back to the full window, once and for all (nohead_off_), when the head-less capture fails.  Build6, then G7
  (ab/G7.sh on gg-g): engine tests t4.log, A pairs nh1_3/nh0_3/nh0_4/nh1_4 on short.json, decode ids fg1_1.
- G6 (11:04-~11:40, old build = HEAD adddaf4): ctest t3.log 36/37 pass (with qsa_topk_active_parity,
  decode_cluster_parity, qsa_parity, qsa_topk_parity), ple_parity fails: its fixture needs
  ../../Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf (not on this machine).  E pair 2 (gg-f, fh0_2 then
  fh1_2): fh0 49.18 / 61.75 tok/s, ms/window prose 57.29 57.12 code 63.81 63.75; fh1 49.23 / 61.85, 57.32 56.98 /
  63.74 63.61; ids [38747, 38747, 41916, 41916] both.  pC lane gg-f / base, 2 alternated pairs (cCf_1, cCb_1, cCb_2,
  cCf_2): 1K 7587, 7516 / 7454, 7425 (+1.5%); 2K 10443, 10375 / 10352, 10307 (+0.8%); 4K 16222, 16207 / 16174, 16171
  (+0.3%); hashes 86557 81685 59650 all.  Not explained by reading: the routed walk stages with % STAGE (8) on any
  layout, the streamed issue is gated on stream_all, no GPU expert cache in either arm (the lane's log adds 'a GPU
  cache of 45 slots ... none is kept'), no MTP (DFlash only).  G7 extended to attribute it (xL default / xC C off /
  xA all off on pC).  D pair 1 (gg-f): dB1_1 105236 / dB0_1 105962 ms at 32K, hash 3242.
- D pairs done (gg-f, pB, alternated dB1_1 dB0_1 dB0_2 dB1_2): 32K on 105236, 105552 / off 105962, 106107 ms
  (-0.64 s, -0.6%), 4K 16547, 16560 / 16574, 16555; hashes 3242 / 59650 / 86768 all.  D KEPT (exact, phase -0.62 s).
- E pair 2 done: tok/s unresolvable as predicted, ids identical; with the nsys prelude drop E is KEPT.
- 11:34 build6 ok (A refinement), committed 1f7ed75; gg-g = its guild-generate (sha256 1d6fed66...).  G7 trimmed
  to xL_1 xC_1 xC_2 xL_2 on pC (lane default vs C off) after the A pairs and the decode bench.  The harness refused a
  REPORT.md from this agent (report files go back as text): the report's content goes into the lane's returned summary.
- G7 done (12:24-12:42, gg-g = HEAD 1f7ed75): t4.log 12/12 pass (runtime_inference_failure, prefill_failure,
  finite_output, parity_test, coupled_draft, dflash, route_window, gr_parity*, host_combine).  A refined, 2 pairs
  short.json (nh1_3 nh0_3 nh0_4 nh1_4): p100 870, 877 / 957, 956 (-8.7%); p200 1729, 1732 / 1900, 1888 (-8.6%); p400
  3467, 3484 / 3810, 3832 (-9.0%); p800 6868, 6874 / 6821, 6856 (+0.5%); p1600 9125, 9121 / 9091, 9101 (+0.3%);
  hashes 10706 74277 60685 12302 73032 all; only the 8-token head-less window captured; gpu_after 7585 / 7559 MiB
  (+26, was +78).  Decode fg1_1: 49.5 / 62.3 tok/s, ids [38747, 38747, 41916, 41916].  pC lane default xL_1, xL_2 vs C
  off (GUILD_PREFILL_HC_REGION=0 GUILD_PREFILL_STREAM_ROWS=0) xC_1, xC_2: 1K 7423, 7462 / 7497, 7499 (-0.7%); 2K
  10304, 10352 / 10368, 10352 (-0.3%); 4K 16152, 16176 / 16207, 16184 (-0.2%); hashes 86768 86557 81685 59650;
  gpu_after 7585 / 7751 MiB.  G6's +1.5% at 1K (gg-f vs base) does not repeat: the final lane's 1K mean 7442 =
  G6's base 7440.  Lane finished: A, C, D, E kept on by default, B opt-in.  Format check passes; tree clean.
