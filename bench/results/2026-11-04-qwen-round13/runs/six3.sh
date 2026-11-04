#!/bin/sh
# six prompts, 2 runs x 256 tokens, alternated: today's q5_k against map H (imatrix)
O=/home/ubuntu/work/nemotron/opt/r17
D=/home/ubuntu/work/dflash/drafter
ML=/home/ubuntu/work/nemotron/guild/data/dflash_vocab.bin
IM=/home/ubuntu/work/imatrix/dl/imatrix_unsloth.gguf
: > $O/six3.progress
arm() { # label env...
  L=$1; shift
  for P in def more; do
    PA=""; [ $P = more ] && PA="--prompts $O/more.json"
    QWEN_RUNS=2 QWEN_BENCH_EXTRA="$PA --prefill auto --spec 8 --spec-min-p 0.5 --extra-args --dflash $D --dflash-vocab $ML --dense q5_k --hc q8_0" $O/qb.sh $O/gmap six3_${L}_$P "$@" > $O/six3_${L}_$P.sum 2>&1
    echo "$L $P $(grep SUMMARY $O/six3_${L}_$P.sum)" >> $O/six3.progress
  done
}
for r in 1 2; do arm today_$r GUILD_NOTHING=1; arm H_$r GUILD_DENSE_IMATRIX=$IM GUILD_DENSE_MAP=attn_output=q8_0,attn_gate=q4_k,attn_q=q4_k,attn_qkv=q4_k; done
echo done >> $O/six3.progress
