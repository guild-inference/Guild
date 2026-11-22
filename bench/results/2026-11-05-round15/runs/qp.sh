#!/bin/sh
O=/home/ubuntu/work/nemotron/opt/r19
D=/home/ubuntu/work/dflash/drafter
ML=/home/ubuntu/work/nemotron/guild/data/dflash_vocab.bin
IM=/home/ubuntu/work/imatrix/dl/imatrix_unsloth.gguf
F="--dense q4_k --dense-map ssm_out=q5_k,attn_output=q8_0 --dense-imatrix $IM --hc q8_0 --prefill-mmq dense"
: > $O/qp.progress
for r in 1 2; do for p in 1 0; do
  QWEN_RUNS=2 QWEN_BENCH_EXTRA="--prefill auto --spec 8 --spec-min-p 0.5 --extra-args --dflash $D --dflash-vocab $ML $F" $O/qb.sh $O/gp qp${p}_$r GUILD_VERIFY_STREAM_PRIO=$p > $O/qp${p}_$r.sum 2>&1
  echo "prio $p round $r $(grep SUMMARY $O/qp${p}_$r.sum) $(python3 -c "import json; print([hash(tuple(x['output_ids']))%100000 for x in json.load(open('$O/b-qp${p}_$r.json'))['runs']])")" >> $O/qp.progress
done; done
