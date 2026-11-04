#!/bin/sh
O=/home/ubuntu/work/nemotron/opt/r18
D=/home/ubuntu/work/dflash/drafter
ML=/home/ubuntu/work/nemotron/guild/data/dflash_vocab.bin
IM=/home/ubuntu/work/imatrix/dl/imatrix_unsloth.gguf
F="--dense q4_k --dense-map ssm_out=q5_k,attn_output=q8_0 --dense-imatrix $IM --hc q8_0 --prefill-mmq dense"
run() { # label exe env
  QWEN_RUNS=1 QWEN_BENCH_EXTRA="--prompts $O/qwen4k.json --timeout 3000 --max-new 64 --prefill auto --spec 8 --spec-min-p 0.5 --extra-args --dflash $D --dflash-vocab $ML $F" $O/qb.sh $O/$2 $1 GUILD_NO_KV_CACHE=1 GUILD_PREFILL_TIMING=1 $3 > $O/$1.sum 2>&1
}
run gl_on gl ""; run gl_off gl GUILD_PREFILL_GROUP_LOCAL=0; run gl_prev gv ""
python3 -c "
import json
for a in ('gl_on','gl_off','gl_prev'):
    d=json.load(open('$O/b-%s.json'%a)); r=d['runs'][0]; print(a, r['prompt_ms'], hash(tuple(r['output_ids']))%100000, d['gpu_after'])" > $O/gl.result
