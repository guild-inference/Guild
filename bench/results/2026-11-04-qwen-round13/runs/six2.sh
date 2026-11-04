#!/bin/sh
# six prompts, 2 runs x 256 tokens, alternated: the dense maps A (q5_k, attention output q8_0), C (A with attn_gate
# and attn_q at q4_k) and D (q6_k, attention output q8_0), all with Unsloth's imatrix
O=/home/ubuntu/work/nemotron/opt/r17
D=/home/ubuntu/work/dflash/drafter
ML=/home/ubuntu/work/nemotron/guild/data/dflash_vocab.bin
IM=/home/ubuntu/work/imatrix/dl/imatrix_unsloth.gguf
: > $O/six2.progress
arm() { # label dense map
  for P in def more; do
    PA=""; [ $P = more ] && PA="--prompts $O/more.json"
    QWEN_RUNS=2 QWEN_BENCH_EXTRA="$PA --prefill auto --spec 8 --spec-min-p 0.5 --extra-args --dflash $D --dflash-vocab $ML --dense $2 --hc q8_0" $O/qb.sh $O/gmap six_$1_$P GUILD_DENSE_IMATRIX=$IM GUILD_DENSE_MAP=$3 > $O/six_$1_$P.sum 2>&1
    echo "$1 $P $(grep SUMMARY $O/six_$1_$P.sum)" >> $O/six2.progress
  done
}
for r in 1 2; do arm A_$r q5_k attn_output=q8_0; arm C_$r q5_k attn_output=q8_0,attn_gate=q4_k,attn_q=q4_k; arm D_$r q6_k attn_output=q8_0; done
echo done >> $O/six2.progress
