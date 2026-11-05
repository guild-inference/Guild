#!/bin/sh
O=/home/ubuntu/work/nemotron/opt/r18
D=/home/ubuntu/work/dflash/drafter
ML=/home/ubuntu/work/nemotron/guild/data/dflash_vocab.bin
IM=/home/ubuntu/work/imatrix/dl/imatrix_unsloth.gguf
QWEN_RUNS=1 QWEN_BENCH_EXTRA="--prompts $O/ctx32k.json --timeout 3000 --max-new 16 --prefill auto --spec 8 --spec-min-p 0.5 --extra-args --dflash $D --dflash-vocab $ML --dense q4_k --dense-map ssm_out=q5_k,attn_output=q8_0 --dense-imatrix $IM --hc q8_0 --prefill-mmq dense" $O/qb.sh $O/gg p32t GUILD_NO_KV_CACHE=1 GUILD_PREFILL_TIMING=1 > $O/p32t.sum 2>&1
