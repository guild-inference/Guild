# nem-decode (round 18) notes

Lane: /home/ubuntu/lanes/r18/nem-decode (guild/ on lane18/nem-decode from perf/next a10d867; gkc/ unchanged).
Scripts: ab.sh (alternated same-binary A/B on the 2 perf clips, --repeat 6), absum.py (medians of runs 1..5),
gatecheck.sh (60 gate clips, ids against /home/ubuntu/work/nemotron/gate/guild5-greedy.jsonl).

## Item 1: coalesced scan state (GUILD_NEM_SCAN_LAYOUT=0 = old)

- 06:58 implemented (template<bool CO> ssm_scan_k; state + snapshots at h*8192 + i*256 + tid). Built.
- 07:20 A/B (bin/tr-item1 md5 fb55e8a4, `gpu-run ab.sh scan 2 GUILD_NEM_SCAN_LAYOUT=0 --mtp 3`), medians of runs 1-5,
  2 pairs (per pair: abpair.py scan):
  | arm | clip | total | prefill | decode | draft | verify | steps | ids |
  | on (coalesced) | 5 s | 605.8 | 350.2 | 211.3 | 42.5 | 168.6 | 5 | 25430 |
  | off (old) | 5 s | 612.5 | 350.1 | 216.7 | 42.5 | 173.9 | 5 | 25430 |
  | on | 30 s | 2208.4 | 915.8 | 1123.2 | 229.0 | 890.8 | 26 | 35646 |
  | off | 30 s | 2247.6 | 913.2 | 1159.1 | 229.1 | 926.7 | 26 | 35646 |
  30 s: -39 ms total (-1.7%), verify -1.4 ms a step; prefill +2.6 ms (+0.3%, the long-window scan, as the scout's
  microbenchmark had: +1.5-2.7% of the scan). 5 s: -6.7 ms.
- item 2 (device argmax) written meanwhile and built (build/transcribe_run), not yet measured.
- 07:45 greedy A/B (`gpu-run ab.sh scang 2 GUILD_NEM_SCAN_LAYOUT=0 --no-mtp`, bin/tr-item1):
  | arm | clip | total | prefill | decode | steps | ids |
  | on | 5 s | 753.3 | 333.0 | 376.4 | 20 | 25430 |
  | off | 5 s | 781.5 | 335.2 | 401.6 | 20 | 25430 |
  | on | 30 s | 2861.5 | 859.9 | 1827.8 | 95 | 35646 |
  | off | 30 s | 2976.3 | 861.1 | 1942.1 | 95 | 35646 |
  greedy decode -5.9% (-1.2 ms a token), total -3.9% (30 s), -3.6% (5 s).
- item 3 written (draft head subset, data/nemotron_draft_vocab.bin = scout's file, md5 3685b6e5; tools/nemotron_draft_vocab.py
  = the scout's builder with arguments; no Python here has pyarrow, so not re-run: the .bin equals the scout's
  sub9995.json, 44,602 ascending ids, 0-1255 and eos 11 in).
- 07:24 gate (bin/tr-item1, `TAG=scan MODES='greedy mtp1 mtp2 mtp3' gpu-run gatecheck.sh`): greedy, mtp1, mtp2, mtp3 each
  60 of 60 clips' ids equal to guild5-greedy.jsonl, PASS (gate/scan-gate-*.cmp). Drafted/accepted over the 60 clips
  (no subset): mtp1 834/792, mtp2 1148/1052, mtp3 1338/1180.
- item 1 kept.

## Item 2: device argmax (GUILD_NEM_DEVICE_ARGMAX=0 = logits to the host + std::max_element)

- nhk::argmax_rows (a block of 1024 per row, float4 loads, strict > per thread, ties to the lower index in the
  reductions; INT_MAX sentinel so all -inf gives 0): std::max_element's first maximum for any non-NaN values.
  verify(..., top) and step(..., top); transcribe_run uses it for verify rows and greedy steps. TRANSCRIBE_CHECK_SPEC
  copies the rows too and prints any row where the GPU's argmax differs from the host's.
- binary bin/tr-item2 (md5 cfe5461d; items 1+2). Queued: run_groupA.sh (A/Bs of items 2 and 3).

## Item 3: draft head subset (GUILD_NEM_DRAFT_VOCAB=0 = whole head)

- binary bin/tr-item3 (md5 f94ed6f9; items 1+2+3). Loads "draft head over 44602 of 131072 tokens (127 of 364 MiB)".
  GUILD_NEM_DRAFT_CHECK=1 compares every draft's subset logits with the whole head's (bitwise) and counts drafts
  whose whole-head argmax is outside the subset. Queued: gate (TAG=dvoc), run_groupC.sh (checks).

## Item 4: prefix snapshot (GUILD_NEM_PREFIX=0 = every prompt whole)

- save_prefix / restore_prefix / prefix_exact in the engine; transcribe_run reads prompt.json's 43 prefix ids once.
  A prompt takes the prefix path only when neither reading ends in a window of 1-8 rows (prefix_exact: such a
  window takes MMVQ and the one-token router, so its rows differ from the same rows in a wider window; e.g. a
  259-row prompt is 256 + 3 read whole, but 216 read from the prefix). TRANSCRIBE_PREFIX_CHECK=1 compares the
  last-row logits of both readings bit for bit per clip.

### 07:36 A/Bs of items 2 and 3 (run_groupA.sh, one lock hold; medians of runs 1-5 of 2 files each)
Item 2 (`EXE=bin/tr-item2 ab.sh argm 2 GUILD_NEM_DEVICE_ARGMAX=0 --mtp 3`, and `argmg ... --no-mtp`):
  | arm | clip | total | prefill | decode | draft | verify | steps | ids |
  | on | 5 s | 596.1 | 347.6 | 203.8 | 42.3 | 161.2 | 5 | 25430 |
  | off | 5 s | 603.8 | 349.6 | 210.8 | 42.3 | 168.1 | 5 | 25430 |
  | on | 30 s | 2168.2 | 911.1 | 1085.4 | 228.3 | 854.5 | 26 | 35646 |
  | off | 30 s | 2207.7 | 914.3 | 1122.2 | 228.5 | 890.9 | 26 | 35646 |
  | greedy on | 5 s | 751.7 | 334.9 | 371.9 | | | 20 | 25430 |
  | greedy off | 5 s | 756.9 | 334.9 | 377.5 | | | 20 | 25430 |
  | greedy on | 30 s | 2842.9 | 864.8 | 1805.3 | | | 95 | 35646 |
  | greedy off | 30 s | 2869.1 | 861.4 | 1834.7 | | | 95 | 35646 |
  30 s: -39.5 ms (-1.8%), verify -1.4 ms a step; 5 s -7.7 ms (-1.3%); greedy decode -0.3 ms a token (-1.6%).
Item 3 (`EXE=bin/tr-item3 ab.sh dvoc 2 GUILD_NEM_DRAFT_VOCAB=0 --mtp 3`):
  | on | 5 s | 577.7 | 349.0 | 185.6 | 24.6 | 160.7 | 5 (15/15) | 25430 |
  | off | 5 s | 598.4 | 350.0 | 203.9 | 42.5 | 161.1 | 5 (15/15) | 25430 |
  | on | 30 s | 2088.2 | 918.0 | 996.1 | 136.7 | 856.5 | 26 (78/69) | 35646 |
  | off | 30 s | 2177.9 | 916.0 | 1090.2 | 228.9 | 858.4 | 26 (78/69) | 35646 |
  30 s: -89.7 ms (-4.1%: draft -3.5 ms a step, the same 26 steps and 69 accepted); 5 s -20.7 ms (-3.5%).
  VRAM: LM 2383 MiB with the subset (2261 without), audio model still 4221 MiB with its float32 matrices kept.
- 07:48 gate with bin/tr-item3 (TAG=dvoc, items 1-3): greedy, mtp1, mtp2, mtp3 each 60 of 60 ids equal, PASS.
  Drafted/accepted/steps over the 60 clips, without the subset (scan gate, tr-item1) -> with it (dvoc gate):
  mtp1 834/792/834 -> 835/791/835; mtp2 1148/1052/574 -> 1150/1051/575; mtp3 1338/1180/446 -> 1344/1178/448.
  Only 2300-131720-0031 (all modes, one accepted draft fewer) and 2094-142345-0054 (mtp3) change; per clip:
  gate/dacc-subset.tsv (dacc.py scan dvoc).
- item 2 and item 3 kept.
- 07:40 final cleanup: the old scan order and GUILD_NEM_SCAN_LAYOUT deleted (7a8cea4), bin/tr-final md5 b04ccefd.
- 08:00 exactness checks (run_groupC.sh, bin/tr-item3): GUILD_NEM_DRAFT_CHECK=1 on the 2 perf clips: 93 draft passes,
  0 of 44,602 subset logits differ from the whole head's in every pass, the whole head's argmax in the subset in all
  93. TRANSCRIBE_CHECK_SPEC=1 --mtp 3 on the 60 gate clips: the GPU's argmax equals std::max_element on every verify
  row (1,792 rows over 448 windows, 0 "window row" lines), every verified row's logits equal greedy's (0 "check: step" lines), the prompt's
  logits again identical (60 of 60), ids 60 of 60 equal to guild5-greedy.
- 08:06 prefix check (bin/tr-item4, TRANSCRIBE_PREFIX_CHECK=1 --mtp 3, 60 gate clips): FAILED, every clip read from the
  prefix differs in nearly every logit (max |d| 1.6-15). Cause: save_prefix allocates its buffers with dalloc, whose
  cudaMemset runs on the legacy default stream, which does not order with the engine's non-blocking stream: the
  zeroing could land after the snapshot copies. Fixed with a cudaDeviceSynchronize after the allocations (only the
  first save allocates). 3 clips were read whole (302, 303 and 259 rows: 259 = 256 + 3, 302 - 43 = 259). Re-check
  queued with bin/tr-final2 (run_after4.sh: check, then if every logit matches, the final gate and the A/Bs).
- 09:42 re-check with the fix (bin/tr-final2): the same differences to the digit (59 of 59 clips read from the prefix
  differ; deterministic), so the race was not the cause. Cause (from the code): cpu_moe Experts::run sums each
  token's expert outputs in group order, the order in which the window's tokens first name the experts ("the group
  order follows the window's other tokens", cpu_moe.cpp); a window that starts at token 43 instead of 0 orders the
  groups differently, the sums round differently, and the router's near-ties (selection bias ~41, sigmoids ~0.01
  apart) turn that into other experts. Stop rule hit: item 4 stopped and reverted (1e59836); the patch is kept at
  item4-prefix-snapshot.patch. An exact version needs the base's window boundaries (the first window after the
  prefix ends at 256) and the prefix tokens' routing ids per MoE layer (and the MTP block's) handed to
  Experts::run as the rows before t_first, which keeps the whole window's group order; not attempted (time, lock).
- 09:55 final binary = items 1-3 (build7 -> bin/tr-final3). Queued run_final.sh: final gate, ctest
  transcribe_gate_test, lane vs base A/B (mtp3 and greedy).
- (takeover after the usage-limit stop) run_final.sh (pid 3863657) still waiting for the machine lock behind GLM,
  Ornith, Qwen-prompt and StartLux jobs of other lanes; bin/tr-final3 md5 4b28e9be = build/transcribe_run (built
  09:54 from 1e59836, items 1-3). format_code.py --check: all 546 files formatted. Reviewed the diff a10d867..HEAD:
  h_arg is 64 B pinned (16 ids >= kMaxVerify 4), top[8] >= 4; argmax_rows ties/-inf/+-0 match std::max_element.
- 10:52-11:07 run_final.sh (bin/tr-final3 md5 4b28e9be = items 1-3; base /home/ubuntu/lanes/r18/base/build/transcribe_run
  md5 111cc2a7):
  - final gate (TAG=final): greedy, mtp1, mtp2, mtp3 each 60 of 60 ids equal to guild5-greedy, WER 10.34%, PASS.
  - ctest transcribe_gate_test: FAILED (rc 8), as on base: its greedy-vs-reference_fp32 bound (57 of 60 identical,
    from the bf16 run) is not met by the Q8_0 GGUF (50 of 60, WER 10.34 vs 10.27); guild5-greedy.jsonl (the committed
    engine) gives the same 50 of 60 and rc 1, and the lane's greedy ids equal it. Its drafted-vs-greedy part
    (--same-as, mtp2) passed (rc 0). Only the Q8_0 GGUF is on this machine.
  - lane vs base (`gpu-run ab2.sh lane 2 bin/tr-final3 base --mtp 3`, `laneg ... --no-mtp`), medians runs 1-5 x 2:
    | arm | clip | total | prefill | decode | draft | verify | steps | ids |
    | lane mtp3 | 5 s | 578.6 | 350.2 | 185.8 | 24.6 | 160.9 | 5 | 25430 |
    | base mtp3 | 5 s | 605.9 | 349.4 | 214.1 | 42.4 | 171.4 | 5 | 25430 |
    | lane mtp3 | 30 s | 2089.2 | 916.2 | 998.5 | 136.4 | 858.9 | 26 | 35646 |
    | base mtp3 | 30 s | 2252.6 | 920.1 | 1156.8 | 228.9 | 924.8 | 26 | 35646 |
    | lane greedy | 5 s | 752.1 | 337.8 | 371.1 | | | 20 | 25430 |
    | base greedy | 5 s | 779.8 | 335.5 | 401.7 | | | 20 | 25430 |
    | lane greedy | 30 s | 2850.1 | 866.2 | 1812.1 | | | 95 | 35646 |
    | base greedy | 30 s | 2980.6 | 866.0 | 1945.0 | | | 95 | 35646 |
    mtp3: 30 s -163.4 ms (-7.3%, decode -13.7%), 5 s -27.3 ms (-4.5%); greedy: 30 s -130.5 ms (-4.4%, decode -6.8%,
    -1.4 ms a token), 5 s -27.7 ms (-3.6%). Lane done; REPORT.md could not be written from this session (the harness
    refuses report files from a subagent), the report went to the orchestrator's structured summary.
