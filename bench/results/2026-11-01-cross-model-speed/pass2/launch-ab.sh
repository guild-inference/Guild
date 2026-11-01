#!/bin/sh
# the prefill fronts on a launcher: parity tests, then Ornith and StartLux with GUILD_PREFILL_LAUNCHER=0 / 1, alternated
W=/home/ubuntu/work/nemotron
S=/home/ubuntu/work/startlux
G=/home/ubuntu/models/startlux-decision-35b-a3b-gguf
O=$W/opt/launch
mkdir -p $O
(cd $W/build-cu && ctest -R 'ornith_prefill_test|ornith_gkernel_plugin_test' --output-on-failure --timeout 1800) > $O/ctest.log 2>&1
cd /home/ubuntu/Guild
for i in 1 2; do for v in 0 1; do
  GUILD_PREFILL_LAUNCHER=$v GUILD_NO_KV_CACHE=1 $W/build-cu/ornith_prefill_bench --prompt 1024 --generate 16 --runs 3 --batch 256 --context 4096 --csv $O/ornith-$v-$i.csv > $O/ornith-$v-$i.log 2>&1
done; done
for i in 1 2; do for v in 0 1; do
  GUILD_PREFILL_LAUNCHER=$v $W/build-cu/decision_run --native $G/StartLux-Decision-35B-A3B-Q8_0.gguf --pack $G/pack-Q8_0 --resident-budget-gib 40 --requests $S/perf/perf-short.jsonl --out $O/startlux-$v-$i.jsonl --repeat 5 --fresh > $O/startlux-$v-$i.log 2>&1
done; done
touch $O/done
