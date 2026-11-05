#!/bin/sh
O=/home/ubuntu/work/nemotron/opt/r18
D=/home/ubuntu/work/dflash/drafter
ML=/home/ubuntu/work/nemotron/guild/data/dflash_vocab.bin
IM=/home/ubuntu/work/imatrix/dl/imatrix_unsloth.gguf
F="--dense q4_k --dense-map ssm_out=q5_k,attn_output=q8_0 --dense-imatrix $IM --hc q8_0 --prefill-mmq dense"
QWEN_RUNS=1 QWEN_BENCH_EXTRA="--prompts $O/short.json --timeout 3000 --max-new 64 --prefill auto --spec 8 --spec-min-p 0.5 --extra-args --dflash $D --dflash-vocab $ML $F" $O/qb.sh $O/gv crsnew GUILD_NO_KV_CACHE=1 > $O/crsnew.sum 2>&1
python3 -c "
import json
d=json.load(open('$O/b-crsnew.json')); print([(r['prompt_tokens'], round(r['prompt_ms']/1000,2), round(r['decode_tok_s'],1)) for r in d['runs']])" > $O/crs2.result
