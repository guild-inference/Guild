#!/bin/sh
O=/home/ubuntu/work/nemotron/opt/r20
: > $O/qnuma.mon
(QWEN_RUNS=1 QWEN_BENCH_EXTRA="--prefill auto --spec 8 --spec-min-p 0.5 --extra-args --dflash /home/ubuntu/work/dflash/drafter --dflash-vocab /home/ubuntu/work/nemotron/guild/data/dflash_vocab.bin --dense q4_k --dense-map ssm_out=q5_k,attn_output=q8_0 --dense-imatrix /home/ubuntu/work/imatrix/dl/imatrix_unsloth.gguf --hc q8_0 --prefill-mmq dense" $O/qb.sh /home/ubuntu/work/nemotron/opt/r19/gpri qnuma > $O/qnuma.sum 2>&1) &
B=$!
while kill -0 $B 2>/dev/null; do
  P=$(pgrep -x gpri | head -1)
  [ -n "$P" ] && { date +%T; numastat -p $P | tail -1; } >> $O/qnuma.mon
  sleep 8
done
