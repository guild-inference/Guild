#!/bin/sh
# the new setup flags against today's: 4K and 32K prompts, then 256 tokens (one run each, alternated)
O=/home/ubuntu/work/nemotron/opt/r17
D=/home/ubuntu/work/dflash/drafter
ML=/home/ubuntu/work/nemotron/guild/data/dflash_vocab.bin
IM=/home/ubuntu/work/imatrix/dl/imatrix_unsloth.gguf
NEW="--dense q4_k --dense-map ssm_out=q5_k,attn_output=q8_0 --dense-imatrix $IM --hc q8_0 --prefill-mmq dense"
OLD="--dense q5_k --hc q8_0"
: > $O/fin.progress
for p in qwen4k ctx32k; do for a in new old; do
  F="$NEW"; [ $a = old ] && F="$OLD"
  QWEN_RUNS=1 QWEN_BENCH_EXTRA="--prompts $O/$p.json --timeout 3000 --prefill auto --spec 8 --spec-min-p 0.5 --extra-args --dflash $D --dflash-vocab $ML $F" $O/qb.sh $O/gfin fin_${p}_$a GUILD_NO_KV_CACHE=1 > $O/fin_${p}_$a.sum 2>&1
  echo "$p $a $(grep SUMMARY $O/fin_${p}_$a.sum) $(python3 -c "import json; r=json.load(open('$O/b-fin_${p}_$a.json'))['runs'][0]; print(r['prompt_tokens'], r['prompt_ms'])")" >> $O/fin.progress
done; done
echo done >> $O/fin.progress
