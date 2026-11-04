#!/bin/sh
# decode without drafts (each setting's own cost a token, whatever its texts draft like): prose + code, 1 run x 256
O=/home/ubuntu/work/nemotron/opt/r17
D=/home/ubuntu/work/dflash/drafter
ML=/home/ubuntu/work/nemotron/guild/data/dflash_vocab.bin
IM=/home/ubuntu/work/imatrix/dl/imatrix_unsloth.gguf
: > $O/plain.progress
arm() { # label dense map
  MV=""; [ -n "$3" ] && MV="GUILD_DENSE_MAP=$3"
  QWEN_RUNS=1 QWEN_BENCH_EXTRA="--prefill auto --spec 8 --spec-min-p 0.5 --extra-args --dflash $D --dflash-vocab $ML --dense $2 --hc q8_0 --decode-strategy normal" $O/qb.sh $O/gmap pl_$1 GUILD_DENSE_IMATRIX=$IM $MV > $O/pl_$1.sum 2>&1
  echo "$1 $(grep SUMMARY $O/pl_$1.sum)" >> $O/plain.progress
}
for r in 1 2; do arm q5_$r q5_k ""; arm A_$r q5_k attn_output=q8_0; arm C_$r q5_k attn_output=q8_0,attn_gate=q4_k,attn_q=q4_k; arm D_$r q6_k attn_output=q8_0; arm q4A_$r q4_k attn_output=q8_0; done
echo done >> $O/plain.progress
