#!/bin/sh
O=/home/ubuntu/work/nemotron/opt/r18
D=/home/ubuntu/work/dflash/drafter
ML=/home/ubuntu/work/nemotron/guild/data/dflash_vocab.bin
IM=/home/ubuntu/work/imatrix/dl/imatrix_unsloth.gguf
F="--dense q4_k --dense-map ssm_out=q5_k,attn_output=q8_0 --dense-imatrix $IM --hc q8_0 --prefill-mmq dense"
for e in gc gv; do
  QWEN_CONTEXT=262144 QWEN_RUNS=1 QWEN_BENCH_EXTRA="--prompts $O/ctx32k.json --timeout 3000 --max-new 128 --prefill auto --spec 8 --spec-min-p 0.5 --extra-args --dflash $D --dflash-vocab $ML $F" $O/qb.sh $O/$e c256_$e GUILD_NO_KV_CACHE=1 > $O/c256_$e.sum 2>&1
done
python3 -c "
import json
a=json.load(open('$O/b-c256_gc.json')); b=json.load(open('$O/b-c256_gv.json'))
ra=a['runs'][0]; rb=b['runs'][0]
print('new', ra['prompt_ms'], round(ra['decode_tok_s'],1), a['gpu_after']); print('old', rb['prompt_ms'], round(rb['decode_tok_s'],1), b['gpu_after']); print('same tokens' if ra['output_ids']==rb['output_ids'] else 'DIFFERENT')" > $O/gc256.result
