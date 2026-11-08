#!/bin/sh
# Round 18 integration check: perf/next with the accepted lanes (int/build) against perf/next a10d867 (base/build),
# every model, alternated, outputs compared.  Each helper takes the machine lock itself.
I=/home/ubuntu/lanes/r18/int/build; B=/home/ubuntu/lanes/r18/base/build; O=/home/ubuntu/lanes/r18/int/runs
P=/home/ubuntu/lanes/r18/int/progress; : > $P
log() { echo "$(date -u +%T) $*" >> $P; }
H=/home/ubuntu/lanes/bin
# 1. tests
(cd $I && $H/gpu-run ctest -R 'mmvq|gr_parity|route_window|router_fast|host_combine|native_expert|expert_parity|pool|glm5|verify|dflash|qwen|prefill|mmq|iq4xs|cpu_moe|nemotron' --timeout 1200 > $O/ctest.log 2>&1); log "ctest $(grep -E 'tests passed|tests failed' $O/ctest.log)"
$H/gpu-run $I/drama3_tower_test /mnt/models-ssd/Strata-data/drama3/tower /tmp/claude-1000/-home-ubuntu-Guild/fc412f01-5572-5649-9611-4ca7c5573b54/scratchpad/drama3/ref.npz.d > $O/tower_test.log 2>&1; log "drama3_tower_test rc $? $(tail -1 $O/tower_test.log)"
# 2. Qwen decode, 2 pairs
for r in 1 2; do for a in i b; do E=$I/guild-generate; [ $a = b ] && E=$B/guild-generate
  $H/qwen-bench-r18 $E $O q${a}$r > $O/q${a}$r.sum 2>&1; log "qwen $a$r $(grep SUMMARY $O/q${a}$r.sum) $(grep ids $O/q${a}$r.sum)"; done; done
# 3. Qwen prompts: short (verifier ingest), 7.6K chunk, 32K
for a in i b; do E=$I/guild-generate; [ $a = b ] && E=$B/guild-generate
  QWEN_RUNS=1 QWEN_BENCH_EXTRA="--prompts /home/ubuntu/work/nemotron/opt/r18/short.json --max-new 32 --warmup 1" $H/qwen-bench-r18 $E $O ps${a} GUILD_NO_KV_CACHE=1 > $O/ps${a}.sum 2>&1
  QWEN_RUNS=1 QWEN_BENCH_EXTRA="--prompts /home/ubuntu/lanes/r18/scout/qwen-prompt/pD.json --timeout 3000 --max-new 32 --warmup 1" $H/qwen-bench-r18 $E $O pd${a} GUILD_NO_KV_CACHE=1 > $O/pd${a}.sum 2>&1
  QWEN_RUNS=1 QWEN_BENCH_EXTRA="--prompts /home/ubuntu/lanes/r18/scout/qwen-prompt/pB.json --timeout 3600 --max-new 32" $H/qwen-bench-r18 $E $O pb${a} GUILD_NO_KV_CACHE=1 > $O/pb${a}.sum 2>&1
  log "qwen prompts $a done"; done
# 4. Ornith, 2 pairs
for r in 1 2; do for a in i b; do E=$I/guild-generate; [ $a = b ] && E=$B/guild-generate
  $H/ornith-bench-r18 $E $O o${a}$r > $O/o${a}$r.sum 2>&1; log "ornith $a$r $(grep SUMMARY $O/o${a}$r.sum) $(grep ids $O/o${a}$r.sum)"; done; done
# 5. Drama 3, 2 rounds x 2 seeds
for r in 1 2; do for s in 11 12; do for a in i b; do E=$I/guild-generate; [ $a = b ] && E=$B/guild-generate
  log "drama3 $($H/drama3-run-r18 $E $O d${a}_${s}_$r $s)"; done; done; done
# 6. GLM, 3 pairs (files kept in the page cache)
for r in 1 2 3; do for a in i b; do E=$I/glm5_run; [ $a = b ] && E=$B/glm5_run
  $H/glm-run $E $O/g${a}$r.out -- GUILD_GLM_KEEP_CACHE=1 > /dev/null 2>&1
  log "glm $a$r $(grep -o 'per step: [0-9.]* / [0-9.]* / [0-9.]* ms' $O/g${a}$r.out | head -1) $(grep -o 'decode 256 tokens.*' $O/g${a}$r.out) ids $(tail -1 $O/g${a}$r.out | md5sum | cut -c1-8)"; done; done
# 7. Nemotron (mtp3), 1 pair each order
M=/home/ubuntu/models/nemotron-3.5-transcribe-guild
for a in i b b i; do E=$I/transcribe_run; [ $a = b ] && E=$B/transcribe_run
  $H/gpu-run $E --model $M/Nemotron-3.5-Transcribe-LM-Q8_0.gguf --audio $M/audio --tokenizer /home/ubuntu/models/nemotron-3.5-transcribe/tokenizer.json --mtp 3 --repeat 6 --clips /home/ubuntu/work/nemotron/perf/clips.jsonl --out $O/n${a}_$(date +%s).jsonl > /dev/null 2>&1
  log "nemotron $a done"; done
# 8. StartLux
G=/home/ubuntu/models/startlux-decision-35b-a3b-gguf
for a in i b; do E=$I/decision_run; [ $a = b ] && E=$B/decision_run
  $H/gpu-run $E --native $G/StartLux-Decision-35B-A3B-Q8_0.gguf --pack $G/pack-Q8_0 --resident-budget-gib 40 --requests /home/ubuntu/work/startlux/perf/perf-short.jsonl --out $O/sl${a}.jsonl --repeat 5 --fresh > $O/sl${a}.log 2>&1
  log "startlux $a rc $?"; done
log done
