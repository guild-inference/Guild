#!/bin/sh
# decode without drafts (each setting's own cost a token, whatever its texts draft like): prose + code, 1 run x 256
O=/home/ubuntu/work/nemotron/opt/r17
D=/home/ubuntu/work/dflash/drafter
ML=/home/ubuntu/work/nemotron/guild/data/dflash_vocab.bin
IM=/home/ubuntu/work/imatrix/dl/imatrix_unsloth.gguf
: > $O/plain2.progress
arm() { # label dense map
  MV=""; [ -n "$3" ] && MV="GUILD_DENSE_MAP=$3"
  QWEN_RUNS=1 QWEN_BENCH_EXTRA="--prefill auto --spec 8 --spec-min-p 0.5 --extra-args --dflash $D --dflash-vocab $ML --dense $2 --hc q8_0 --decode-strategy normal" $O/qb.sh $O/gmap pl2_$1 GUILD_DENSE_IMATRIX=$IM $MV > $O/pl2_$1.sum 2>&1
  echo "$1 $(grep SUMMARY $O/pl2_$1.sum)" >> $O/plain2.progress
}
for r in 1 2; do arm C_$r q5_k attn_output=q8_0,attn_gate=q4_k,attn_q=q4_k; arm F_$r q5_k attn_output=q8_0,attn_gate=q4_k,attn_q=q4_k,ssm_out=q6_k; arm H_$r q5_k attn_output=q8_0,attn_gate=q4_k,attn_q=q4_k,attn_qkv=q4_k; arm I_$r q5_k attn_output=q6_k,attn_gate=q4_k,attn_q=q4_k; done
echo done >> $O/plain2.progress

