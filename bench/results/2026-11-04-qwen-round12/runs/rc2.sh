#!/bin/sh
O=/home/ubuntu/work/nemotron/opt/r16
D=/home/ubuntu/work/dflash/drafter
ML=/home/ubuntu/work/nemotron/guild/data/dflash_vocab.bin
: > $O/rc2.progress
for r in 32 64 128 160; do
  QWEN_RUNS=1 QWEN_BENCH_EXTRA="--prompts $O/qwen4k.json --timeout 3000 --prefill auto --spec 8 --spec-min-p 0.5 --extra-args --dflash $D --dflash-vocab $ML --dense q5_k --hc q8_0" $O/qb.sh $O/gr rr$r GUILD_NO_KV_CACHE=1 GUILD_PREFILL_TIMING=1 GUILD_PREFILL_RING_CHUNK=1 GUILD_PREFILL_RING=$r > $O/rr$r.sum 2>&1
  echo "ring $r $(grep SUMMARY $O/rr$r.sum) $(grep 'prefill timing: 4007' $O/b-rr$r.log | tail -1 | grep -o 'wall [0-9]* ms\|wait copy [0-9]*\|gather [0-9]*' | tr '\n' ' ') $(grep -ci 'out of memory' $O/b-rr$r.log)" >> $O/rc2.progress
done
echo done >> $O/rc2.progress
