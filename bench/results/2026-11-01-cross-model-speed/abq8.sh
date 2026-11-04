#!/bin/sh
# StartLux-Decision Q8_0: guild-kernels-cpu 5147f69 (build) against perf/q8-avx512 (build-q8), alternated twice:
# perf-short.jsonl (warm-up, 1 and 3 questions on ~500 tokens, a 1K prompt), 5 runs each from a reset session.
# The answers' probabilities must be the same bytes.
W=/home/ubuntu/work/startlux
G=/home/ubuntu/models/startlux-decision-35b-a3b-gguf
O=$W/abq8
mkdir -p $O
for i in 1 2; do for b in build build-q8; do
  /home/ubuntu/work/nemotron/$b/decision_run --native $G/StartLux-Decision-35B-A3B-Q8_0.gguf --pack $G/pack-Q8_0 \
    --resident-budget-gib 40 --requests $W/perf/perf-short.jsonl --out $O/$b-$i.jsonl --repeat 5 --fresh \
    > $O/$b-$i.log 2>&1
  echo "$b $i rc $?" >> $O/progress
done; done
