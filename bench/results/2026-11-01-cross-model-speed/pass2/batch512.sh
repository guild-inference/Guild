#!/bin/sh
# prefill chunks of 512 (halves of 256) against 256: Ornith (--batch) and StartLux (GUILD_PREFILL_BATCH), build-cu
W=/home/ubuntu/work/nemotron
S=/home/ubuntu/work/startlux
G=/home/ubuntu/models/startlux-decision-35b-a3b-gguf
O=$W/opt/b512
mkdir -p $O
cd /home/ubuntu/Guild
for i in 1 2; do for b in 256 512; do
  GUILD_NO_KV_CACHE=1 $W/build-cu/ornith_prefill_bench --prompt 1024 --generate 16 --runs 3 --batch $b --context 4096 --csv $O/ornith-$b-$i.csv > $O/ornith-$b-$i.log 2>&1
done; done
for i in 1 2; do for b in 256 512; do
  GUILD_PREFILL_BATCH=$b $W/build-cu/decision_run --native $G/StartLux-Decision-35B-A3B-Q8_0.gguf --pack $G/pack-Q8_0 --resident-budget-gib 40 --requests $S/perf/perf-short.jsonl --out $O/startlux-$b-$i.jsonl --repeat 5 --fresh > $O/startlux-$b-$i.log 2>&1
done; done
