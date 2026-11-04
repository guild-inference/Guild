#!/bin/sh
O=/home/ubuntu/work/nemotron/opt/r16
D=/home/ubuntu/work/dflash/drafter
ML=/home/ubuntu/work/nemotron/guild/data/dflash_vocab.bin
: > $O/d6.progress
for r in 1 2; do for q in q6_k q5_k; do
  QWEN_RUNS=2 QWEN_BENCH_EXTRA="--prefill auto --spec 8 --spec-min-p 0.5 --extra-args --dflash $D --dflash-vocab $ML --dense $q --hc q8_0" $O/qb.sh $O/gh d6_${q}_$r > $O/d6_${q}_$r.sum 2>&1
  echo "$q $r $(grep SUMMARY $O/d6_${q}_$r.sum) $(grep 'ms/window' $O/b-d6_${q}_$r.log | sed -E 's/.*windows, ([0-9.]+) tokens\/window, ([0-9.]+) ms\/window.*/\1 tok, \2 ms;/' | tr '\n' ' ')" >> $O/d6.progress
done; done
echo done >> $O/d6.progress
