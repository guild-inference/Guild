#!/bin/sh
# Qwen's 31,918-token prompt with setup's settings: chunks of 512 (today's 8 GB setting) and the engine's own (auto)
O=/home/ubuntu/work/nemotron/opt/r14
D=/home/ubuntu/work/dflash/drafter
ML=/home/ubuntu/work/nemotron/guild/data/dflash_vocab.bin
: > $O/p32.progress
for c in auto 512; do
  QWEN_RUNS=1 QWEN_BENCH_EXTRA="--prompts $O/ctx32k.json --timeout 3000 --prefill $c --spec 8 --spec-min-p 0.5 --extra-args --dflash $D --dflash-vocab $ML --dense q5_k --hc q8_0" $O/qb.sh $O/gg3 p32_$c GUILD_NO_KV_CACHE=1 > $O/p32_$c.sum 2>&1
  echo "$c $(grep SUMMARY $O/p32_$c.sum)" >> $O/p32.progress
done
echo done >> $O/p32.progress
