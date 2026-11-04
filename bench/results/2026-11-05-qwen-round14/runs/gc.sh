#!/bin/sh
O=/home/ubuntu/work/nemotron/opt/r18
D=/home/ubuntu/work/dflash/drafter
ML=/home/ubuntu/work/nemotron/guild/data/dflash_vocab.bin
IM=/home/ubuntu/work/imatrix/dl/imatrix_unsloth.gguf
F="--dense q4_k --dense-map ssm_out=q5_k,attn_output=q8_0 --dense-imatrix $IM --hc q8_0 --prefill-mmq dense"
QWEN_RUNS=1 QWEN_BENCH_EXTRA="--prompts $O/qwen4k.json --timeout 3000 --max-new 64 --prefill auto --spec 8 --spec-min-p 0.5 --extra-args --dflash $D --dflash-vocab $ML $F" $O/qb.sh $O/gc gc4k GUILD_NO_KV_CACHE=1 GUILD_PREFILL_TIMING=1 > $O/gc4k.sum 2>&1
QWEN_RUNS=1 QWEN_BENCH_EXTRA="--prompts $O/ctx32k.json --timeout 3000 --prefill auto --spec 8 --spec-min-p 0.5 --extra-args --dflash $D --dflash-vocab $ML $F" $O/qb.sh $O/gc gc32k GUILD_NO_KV_CACHE=1 GUILD_PREFILL_TIMING=1 > $O/gc32k.sum 2>&1
python3 -c "
import json
a=json.load(open('$O/b-gc4k.json')); r=a['runs'][0]; print('4k', r['prompt_ms'], hash(tuple(r['output_ids']))%100000, a['gpu_after'])
b=json.load(open('$O/b-gc32k.json')); r=b['runs'][0]
o=json.load(open('/home/ubuntu/work/nemotron/opt/r17/b-fin_ctx32k_new.json'))['runs'][0]
print('32k', r['prompt_ms'], 'old', o['prompt_ms'], 'same tokens' if r['output_ids']==o['output_ids'] else 'DIFFERENT', round(r['decode_tok_s'],1), b['gpu_after'])" > $O/gc.result
