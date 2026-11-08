# kcuda-onechunk (round 18) notes

Lane: one-chunk staged MMVQ for Q4_K rows of 8-16 blocks and Q6_K rows of 8 blocks, plus 4-column Q4_K as two
2-column calls.  gkc worktree `gkc/` (lane18/kcuda-onechunk from de65d60), guild worktree `guild/` (tests only).

## Log

- 06:40 UTC setup, base build of the lane done (unchanged sources).
- Implemented in gkc/src/math/native_mmvq.cu: Staged<Q4KTraits>::ROWC = 16 + dot(); staged_one<F>(nb, ncols) rule
  (Q4_K nb 8..16, not >10 at >=3 cols; Q6_K nb >= 8); GUILD_MMVQ_ONE_WIDE=0 restores the old rule (and no split);
  4-col Q4_K split into two 2-col staged calls (GUILD_MMVQ_SPLIT4=0 turns off only the split).
- guild: mmvq_staged_parity gets Q4_K 2560x10240, 2048x4096, 2048x8195, 2304x2001, 3072x1501, Q6_K 2048x8192,
  2048x4099, and an in-process check that every column of a T-column call == the 1-column call; ctest
  mmvq_staged_parity_one_wide_off (GUILD_MMVQ_ONE_WIDE=0) added.
- 07:00 trap: the first rebuild did not recompile native_mmvq.cu (its .o finished after the edit's mtime); touched it.
- 07:10 ctest -R mmvq (lane build, under gpu-run): mmvq_multi_parity, mmvq_staged_reference, mmvq_staged_parity,
  mmvq_staged_parity_one_wide_off all pass; 2,119,284 outputs, 0 differ bitwise, and the per-column check passes.
  nsys of mmvq_staged_parity: default runs Q4K C=16 kernels at 1/2/3 cols (2-col n=18 = 6 shapes x (T=2 + T=4 split
  x 2)), Q4K C=8 only at 3 cols (12/16-block rows); ONE_WIDE=0 runs only Q4K C=8 (as base). Q6K C=12 1-col 36 -> 54.
- commits: gkc b7f7ab9, guild 5080bbd.
- 07:20 ctest -R 'mmvq|gr_parity|ornith|route_window|glm5|native_expert|decode_cluster|shared_expert|iq_multi|host_combine': 27/27 pass (ctest1.log).
- Qwen A/B pair 1 (ab/qwen_ab.sh; qwen-bench-r18 lane guild-generate, new = default, old = GUILD_MMVQ_ONE_WIDE=0):
  new1 prose 54.93/54.91 ms/win, code 61.38/61.21, 51.22 / 64.25 tok/s, ids [38747, 38747, 41916, 41916]
  old1 prose 57.23/57.12 ms/win, code 63.61/63.65, 49.20 / 61.89 tok/s, ids identical.  (+4.1% / +3.8%)
  old2 prose 57.06/57.08 ms/win, code 63.67/63.57, 49.30 / 61.91 tok/s, ids identical
  new2 prose 54.93/54.95 ms/win, code 61.15/61.08, 51.21 / 64.44 tok/s, ids identical
  => means: prose 57.12 -> 54.93 ms/win (-2.19), 49.25 -> 51.21 tok/s (+4.0%); code 63.63 -> 61.21 ms/win (-2.42),
     61.90 -> 64.34 tok/s (+3.9%).  Stop rule (+1.5%) cleared.  Base lane's own numbers (runs/q_base.sum): 49.34 / 62.71.
- next (queued, ab/chain1.sh): shapes harness (3 arms x 2), Ornith 2 pairs, split 2 pairs, Drama 3 2 rounds x 2 seeds;
  ab/chain2.sh: nsys new/old Qwen (QWEN_RUNS=1), GLM lane + base.
- 09:10 harness (ab/shapes.sh: glm5_kernels_test shapes, lane binary, arms new / GUILD_MMVQ_ONE_WIDE=0 / SPLIT4=0,
  x2 alternated, min of 2; us): Q4_K one-chunk wins as expected at T1-T3 (qwen qkv q4 T2 95.8 -> 75.3, T3 129.0 ->
  92.8; ornith qkv T2 59.4 -> 50.3, T3 75.9 -> 62.0; ornith out (16 bl) T2 33.4 -> 27.6, T3 unchanged (chunked)).
  Q6_K 8-block: ornith qkv q6 T3..T8 -14..-28%, ornith head T2..T8 -5..-22%.
  SPLIT at T4: 10 blocks faster (qwen qkv 171.9 -> 156.1, gate 100.3 -> 95.4, q 205.6 -> 191.7) but 8 and 16 blocks
  SLOWER (ornith qkv 98.3 -> 104.6, gate 51.1 -> 59.0, out 44.6 -> 55.2).  => restrict the split; measuring 9/11/12/14.
  Unaffected paths move up to +-8% at T2 between arms (noise of this harness: qwen qkv q6 T2 106.7/114.5/106.9).
- 10:15 (takeover after a usage-limit stop) state: orn_ab.sh running (onew1 done: prose 83.64 / code 97.47 tok/s, ids
  [30936, 30936, 65886, 65886]; oold1.. waiting for the lock), probe.sh (q4w shapes 8/9/10/11/12/14/16 blocks, 3 arms
  x 3) waiting for the lock, chain3.sh waiting for orn_ab.done (then d3_r1, glm lane, split pair 1, d3_r2, split pair
  2, nsys new/old), all on build/ (binary 1 = b7f7ab9, split for every 8-16-block row).  Stopping chain3 was not
  allowed, so build/ stays binary 1 until chain3 ends; the restricted split goes into a second build dir, build-fin/.
  Committed the harness shapes (guild 98ad15c).
- Why chain3's numbers stand for the restricted split too: the split only acts at 4 columns.  Qwen's Q4_K dense rows are
  all 2560 wide (10 blocks; setup's map makes ssm_out Q5_K and attn_output Q8_0; the drafter's 2560-wide rows are 10
  blocks, its down is 30 blocks); Ornith (--spec 4: T <= 5, groups ceil(T/2) <= 3 columns), Drama 3 (1 column) and GLM
  (1 draft, 2 columns) never call Q4_K at 4 columns.  So keeping the split at 10 blocks runs the same kernels for all
  four; it only spares rows of 8 and 16 blocks (e.g. a 4096-wide Q4_K attn_output without setup's map, Ornith at
  --spec >= 7) the slower split.
- 10:52 probe done (ab/probe_{new,old,sp0}_{1,2,3}.txt, GLM5_SHAPES_ONLY=q4w, Q4_K x 8192 rows, min of 3, us).  At 4
  columns, sp0 (no split, launch_multi) -> new (split): 8 bl 101.1 -> 108.0 (+6.8%), 9: 134.0 -> 123.9 (-7.5%),
  10: 137.0 -> 126.6 (-7.6%), 11: 150.6 -> 141.8 (-5.8%), 12: 152.3 -> 145.9 (-4.2%), 14: 165.8 -> 168.1 (+1.4%),
  16: 179.4 -> 190.5 (+6.2%).  T1-T3 one-chunk vs old: 9 bl T2 78.6 -> 59.4, T3 101.5 -> 74.4; 10 bl T2 79.3 -> 61.4,
  T3 102.5 -> 74.8; 11-16 bl T2 -11..-21%, T3 chunked in both (11 bl chunked T3 116 us vs 10 bl one-chunk 75: one
  chunk at 3 columns for 11-12 blocks is unmeasured, an open point).  T5/T8 (the same upstream kernels in every arm)
  read up to +5% in the new arm: harness noise / power state, not a code difference.
  => split only for rows of 9 to 12 blocks: gkc 11936b5 (q4k_split4), guild 94a91d4 (test comment).
- 10:58 ab/chainF.sh (background): configure build-fin/ (cfg_fin.sh), full build, ctest -R
  'mmvq|gr_parity|qwen|ornith|drama3|route_window' (11 tests) -> ctest_fin.log, one Qwen run fin1 (ids) on build-fin.
- 11:04 Ornith pair 1 (orn_ab.sh, binary 1): onew1 prose 24.36/24.23 ms/win, code 25.03/25.00, 83.64 / 97.47 tok/s;
  oold1 (GUILD_MMVQ_ONE_WIDE=0) prose 25.44/25.39, code 26.40/26.44, 79.93 / 92.27 tok/s; ids [30936, 30936, 65886,
  65886] both.  (+4.6% / +5.6%)
- 11:20 the lock gives ~15-30 min a hold to each job (5-6 lanes waiting), so chain3's 15 holds will not end in this
  session; replaced my own chainF (stopped while it still waited for the configure) with ab/chainG.sh: configure +
  build build-fin/, then ONE gpu-run of ab/fin_batch.sh: ctest -R 'mmvq|gr_parity|qwen|ornith|drama3|route_window' ->
  ctest_fin.log, Qwen fin1, Drama 3 dfin11 / dfin12 (ab/qwen-nolock.sh, ab/drama3-nolock.sh = the lane helpers with
  their gpu-run removed, same flags) -> ab/fin_batch.log.
- 11:36 build-fin built (gkc 11936b5, guild 94a91d4).  tools/format_code.py --check: all 546 files formatted.
- 11:46 Ornith oold2: prose 25.51/25.48 ms/win, code 26.42/26.35, 79.69 / 92.40 tok/s, ids identical.  onew2 queued.
- 12:28 the final batch (ab/fin_batch.sh: ctests, Qwen fin1, Drama 3 dfin11/dfin12, GLM glm_fin.out, all on
  build-fin/) re-queued detached as ab/finq.sh (-> ab/fin_batch.log, ab/fin_batch.done) so it survives the hand-off;
  my chainG (session task) was stopped first.  REPORT.md could not be written from this session (the tool refuses report
  files from a subagent); the report went to the orchestrator in the structured result.
- 12:45 Ornith onew2: prose 24.39/24.44 ms/win, code 25.00/25.03, 83.22 / 97.47 tok/s, ids identical.
  => Ornith 2 pairs (binary 1): prose 25.46 -> 24.36 ms/win (-1.10), 79.81 -> 83.43 tok/s (+4.5%); code 26.40 ->
     25.02 ms/win (-1.39), 92.33 -> 97.47 tok/s (+5.6%); ids [30936, 30936, 65886, 65886] in all 4 arms.
  chain3 now starts (d3_r1, glm, split pair 1, d3_r2, split pair 2, nsys) on build/; ab/finq.sh still queued.
- 13:02 hand-off (3 h budget used).  Still queued for the lock, results land on disk:
  * chain3.sh (binary 1, build/): ab/d3_ab.log (dn/do 11/12 x 2 rounds: frames/s + md5, expect 6e04c9b1 / f84288c9),
    ab/glm.out + glm_lane.txt (ids line vs ab/glm_base_ref.out's last line), ab/split_ab.log (sp1 vs
    GUILD_MMVQ_SPLIT4=0, 2 pairs, code ms/win), ns/qn*, ns/qo* (nsys new/old Qwen, QWEN_RUNS=1); chain3.done.
  * ab/finq.sh -> ab/fin_batch.sh (final binary build-fin/, gkc 11936b5): ctest_fin.log (11 tests), ab/fin1.out /
    b-fin1.json (Qwen ids, expect [38747, 38747, 41916, 41916]), ab/dfin11.wav / dfin12.wav (md5), ab/glm_fin.out;
    ab/fin_batch.done.
  Next: read those, check ids / md5s, add the split and Drama 3 tables, ask the user for the lock bump.
