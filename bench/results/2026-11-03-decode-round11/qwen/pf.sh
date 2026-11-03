#!/bin/sh
# Qwen's ~4K-token prompt with setup's settings (DFlash, --dense q5_k --hc q8_0) at prefill chunks of 1024, 1536 and
# 2048 tokens, each computed as 512-token halves/thirds/quarters (--prefill-rows 512), one run each (no prefix cache)
O=/home/ubuntu/work/nemotron/opt/r14
D=/home/ubuntu/work/dflash/drafter
ML=/home/ubuntu/work/nemotron/guild/data/dflash_vocab.bin
: > $O/pf.progress
for c in 1024 1536 2048; do
  QWEN_RUNS=1 QWEN_BENCH_EXTRA="--prompts $O/qwen4k.json --timeout 3000 --prefill $c --spec 8 --spec-min-p 0.5 --extra-args --dflash $D --dflash-vocab $ML --dense q5_k --hc q8_0 --prefill-rows 512" $O/qb.sh $O/gg pf$c GUILD_NO_KV_CACHE=1 > $O/pf$c.sum 2>&1
  echo "$c $(grep SUMMARY $O/pf$c.sum) $(grep -c 'out of memory\|OOM\|does not fit' $O/b-pf$c.log)" >> $O/pf.progress
done
echo done >> $O/pf.progress
