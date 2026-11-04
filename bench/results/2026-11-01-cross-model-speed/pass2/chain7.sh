#!/bin/sh
# guild-kernels-cuda perf/prefill-mmvq: build-cu (with the CPU q8 branch), the MMVQ and prefill parity tests,
# Ornith (build vs build-cu), StartLux (build-q8 vs build-cu), Qwen texts (build-cuda12 vs build-cu)
W=/home/ubuntu/work/nemotron
S=/home/ubuntu/work/startlux
G=/home/ubuntu/models/startlux-decision-35b-a3b-gguf
O=$W/opt/cu
P=$W/chain7.progress
log() { echo "$(date -u '+%F %T') $*" >> $P; }
mkdir -p $O
log start
TARGETS="decision_run ornith_prefill_bench ornith_prefill_test mmvq_multi_parity mmvq_staged_parity guild-generate" /home/ubuntu/lanes/bin/build-run sh $W/build-cu.sh
log "build-cu $(cat $W/build-cu.rc)"
grep -q "build 0" $W/build-cu.rc || { log "build failed"; exit 1; }
/home/ubuntu/lanes/bin/gpu-run sh -c "cd $W/build-cu && ctest -R 'mmvq_multi_parity|mmvq_staged_parity|ornith_prefill_test' --output-on-failure --timeout 1800 > $O/ctest.log 2>&1"
log "ctest: $(grep 'tests passed' $O/ctest.log)"
/home/ubuntu/lanes/bin/gpu-run sh -c "cd /home/ubuntu/Guild && for i in 1 2; do for b in build build-cu; do GUILD_NO_KV_CACHE=1 $W/\$b/ornith_prefill_bench --prompt 1024 --generate 16 --runs 3 --batch 256 --context 4096 --csv $O/ornith-\$b-\$i.csv > $O/ornith-\$b-\$i.log 2>&1; done; done"
log "ornith done"
/home/ubuntu/lanes/bin/gpu-run sh -c "for i in 1 2; do for b in build-q8 build-cu; do $W/\$b/decision_run --native $G/StartLux-Decision-35B-A3B-Q8_0.gguf --pack $G/pack-Q8_0 --resident-budget-gib 40 --requests $S/perf/perf-short.jsonl --out $O/startlux-\$b-\$i.jsonl --repeat 5 --fresh > $O/startlux-\$b-\$i.log 2>&1; done; done"
log "startlux done"
/home/ubuntu/lanes/bin/gpu-run sh -c "/home/ubuntu/lanes/int3/tools/qb.sh /home/ubuntu/Guild/build-cuda12/guild-generate $O base; /home/ubuntu/lanes/int3/tools/qb.sh $W/build-cu/guild-generate $O cu" > /dev/null 2>&1
log "qwen done"
touch $W/chain7.done
