#!/bin/sh
O=/home/ubuntu/work/nemotron/opt/r16
D=/home/ubuntu/work/dflash/drafter
ML=/home/ubuntu/work/nemotron/guild/data/dflash_vocab.bin
: > $O/dk.progress
for r in 1 2; do for e in gh gk; do
  QWEN_RUNS=2 QWEN_BENCH_EXTRA="--prefill auto --spec 8 --spec-min-p 0.5 --extra-args --dflash $D --dflash-vocab $ML --dense q5_k --hc q8_0" $O/qb.sh $O/$e dk_${e}_$r > $O/dk_${e}_$r.sum 2>&1
  echo "$e $r $(grep SUMMARY $O/dk_${e}_$r.sum)" >> $O/dk.progress
done; done
echo done >> $O/dk.progress
