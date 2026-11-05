#!/bin/sh
O=/home/ubuntu/work/nemotron/opt/r18
D=/home/ubuntu/work/dflash/drafter
ML=/home/ubuntu/work/nemotron/guild/data/dflash_vocab.bin
IM=/home/ubuntu/work/imatrix/dl/imatrix_unsloth.gguf
: > $O/chk.progress
for c in 1024 2048 4096; do
  QWEN_RUNS=1 QWEN_BENCH_EXTRA="--prompts $O/qwen4k.json --timeout 3000 --max-new 16 --prefill $c --spec 8 --spec-min-p 0.5 --extra-args --dflash $D --dflash-vocab $ML --dense q4_k --dense-map ssm_out=q5_k,attn_output=q8_0 --dense-imatrix $IM --hc q8_0 --prefill-mmq dense --prefill-rows 512" $O/qb.sh $O/gg chk$c GUILD_NO_KV_CACHE=1 GUILD_PREFILL_TIMING=1 > $O/chk$c.sum 2>&1
  echo "chunk $c $(grep 'prefill timing: 4007' $O/b-chk$c.log | tail -1 | grep -o 'wall [0-9]* ms\|gemm gate/up [0-9]*\|gemm down [0-9]*\|wait copy [0-9]*\|dequant [0-9]*' | tr '\n' ' ')" >> $O/chk.progress
done
