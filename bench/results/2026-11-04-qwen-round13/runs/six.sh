#!/bin/sh
# six prompts (prose, code + story, sql, Chinese, Russian), 2 runs x 256 tokens, alternated: today's q5_k, q5_k and
# q6_k with Unsloth's imatrix
O=/home/ubuntu/work/nemotron/opt/r17
D=/home/ubuntu/work/dflash/drafter
ML=/home/ubuntu/work/nemotron/guild/data/dflash_vocab.bin
IM=/home/ubuntu/work/imatrix/dl/imatrix_unsloth.gguf
: > $O/six.progress
arm() { # label dense imatrix(0/1)
  V=""; [ $3 = 1 ] && V="GUILD_DENSE_IMATRIX=$IM"
  for P in def more; do
    PA=""; [ $P = more ] && PA="--prompts $O/more.json"
    QWEN_RUNS=2 QWEN_BENCH_EXTRA="$PA --prefill auto --spec 8 --spec-min-p 0.5 --extra-args --dflash $D --dflash-vocab $ML --dense $2 --hc q8_0" $O/qb.sh $O/gmap six_$1_$P $V > $O/six_$1_$P.sum 2>&1
    echo "$1 $P $(grep SUMMARY $O/six_$1_$P.sum)" >> $O/six.progress
  done
}
for r in 1 2; do arm q5_$r q5_k 0; arm q5im_$r q5_k 1; arm q6im_$r q6_k 1; done
echo done >> $O/six.progress
