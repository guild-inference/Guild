#!/bin/sh
O=/home/ubuntu/work/nemotron/opt/r16
D=/home/ubuntu/work/dflash/drafter
ML=/home/ubuntu/work/nemotron/guild/data/dflash_vocab.bin
for v in 1 0; do
QWEN_RUNS=1 QWEN_BENCH_EXTRA="--prompts $O/qwen4k.json --timeout 3000 --prefill auto --spec 8 --spec-min-p 0.5 --extra-args --dflash $D --dflash-vocab $ML --dense q5_k --hc q8_0" $O/qb.sh $O/gm pm$v GUILD_NO_KV_CACHE=1 GUILD_PREFILL_TIMING=1 GUILD_PREFILL_DENSE_MMQ=$v > $O/pm$v.sum 2>&1
done
echo done > $O/pm.done
