#!/bin/sh
# (a) short-prompt decode (prose, code) with prefill chunks of 512 and 2048, alternated twice; (b) the ~4K prompt at
# chunks of 3072 and 4096 (--prefill-rows 512)
O=/home/ubuntu/work/nemotron/opt/r14
D=/home/ubuntu/work/dflash/drafter
ML=/home/ubuntu/work/nemotron/guild/data/dflash_vocab.bin
: > $O/pf2.progress
for i in 1 2; do for c in 512 2048; do
  QWEN_RUNS=1 QWEN_BENCH_EXTRA="--prefill $c --spec 8 --spec-min-p 0.5 --extra-args --dflash $D --dflash-vocab $ML --dense q5_k --hc q8_0 --prefill-rows 512" $O/qb.sh $O/gg dec${c}_$i > $O/dec${c}_$i.sum 2>&1
  echo "dec $c $i $(grep SUMMARY $O/dec${c}_$i.sum) $(grep -h 'ms/window' $O/b-dec${c}_$i.log | tail -2 | sed -E 's/.*tokens\/window, ([0-9.]+) ms\/window.*CPU dispatch ([0-9.]+).*/\1 ms cpu \2/' | tr '\n' ';')" >> $O/pf2.progress
done; done
for c in 3072 4096; do
  QWEN_RUNS=1 QWEN_BENCH_EXTRA="--prompts $O/qwen4k.json --timeout 3000 --prefill $c --spec 8 --spec-min-p 0.5 --extra-args --dflash $D --dflash-vocab $ML --dense q5_k --hc q8_0 --prefill-rows 512" $O/qb.sh $O/gg pf$c GUILD_NO_KV_CACHE=1 > $O/pf$c.sum 2>&1
  echo "pf $c $(grep SUMMARY $O/pf$c.sum) $(grep -c 'out of memory\|does not fit' $O/b-pf$c.log)" >> $O/pf2.progress
done
echo done >> $O/pf2.progress
