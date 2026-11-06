#!/bin/sh
# Qwen3.8 (setup's flags, DFlash): the pool before (gpri) and after (gpool) the claim changes, 2 rounds alternated
O=/home/ubuntu/work/nemotron/opt/r20
D=/home/ubuntu/work/dflash/drafter
ML=/home/ubuntu/work/nemotron/guild/data/dflash_vocab.bin
IM=/home/ubuntu/work/imatrix/dl/imatrix_unsloth.gguf
F="--dense q4_k --dense-map ssm_out=q5_k,attn_output=q8_0 --dense-imatrix $IM --hc q8_0 --prefill-mmq dense"
: > $O/qpool.progress
for r in 1 2; do for a in new old; do
  E=$O/gpool; [ $a = old ] && E=/home/ubuntu/work/nemotron/opt/r19/gpri
  QWEN_RUNS=2 QWEN_BENCH_EXTRA="--prefill auto --spec 8 --spec-min-p 0.5 --extra-args --dflash $D --dflash-vocab $ML $F" $O/qb.sh $E q${a}_$r > $O/q${a}_$r.sum 2>&1
  echo "$a round $r $(grep SUMMARY $O/q${a}_$r.sum) $(python3 -c "import json; print([hash(tuple(x['output_ids']))%100000 for x in json.load(open('$O/b-q${a}_$r.json'))['runs']])")" >> $O/qpool.progress
done; done
echo done >> $O/qpool.progress
