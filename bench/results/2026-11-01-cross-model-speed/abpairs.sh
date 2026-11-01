#!/bin/sh
# the paired prefill: ornith_prefill_test (batched against sequential, bitwise), then StartLux-Decision Q8_0:
# build with GUILD_PREFILL_PAIRS=0 / build / build-q8, alternated twice, perf-short.jsonl, 5 runs each
W=/home/ubuntu/work/startlux
N=/home/ubuntu/work/nemotron
G=/home/ubuntu/models/startlux-decision-35b-a3b-gguf
O=$W/abpairs
mkdir -p $O
(cd $N/build && ctest -R ornith_prefill_test --output-on-failure --timeout 1200) > $O/ornith_prefill_test.log 2>&1
echo "ornith_prefill_test rc $?" >> $O/progress
for i in 1 2; do for arm in off pairs q8; do
  case $arm in off) B=$N/build; E="GUILD_PREFILL_PAIRS=0";; pairs) B=$N/build; E="X=1";; q8) B=$N/build-q8; E="X=1";; esac
  env $E $B/decision_run --native $G/StartLux-Decision-35B-A3B-Q8_0.gguf --pack $G/pack-Q8_0 \
    --resident-budget-gib 40 --requests $W/perf/perf-short.jsonl --out $O/$arm-$i.jsonl --repeat 5 --fresh \
    > $O/$arm-$i.log 2>&1
  echo "$arm $i rc $?" >> $O/progress
done; done
