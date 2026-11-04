#!/bin/sh
# prompt time by length: the prompt path (default) against the verifier's windows (GUILD_PREFILL_VERIFY_MAX=100000)
O=/home/ubuntu/work/nemotron/opt/r18
D=/home/ubuntu/work/dflash/drafter
ML=/home/ubuntu/work/nemotron/guild/data/dflash_vocab.bin
IM=/home/ubuntu/work/imatrix/dl/imatrix_unsloth.gguf
F="--dense q4_k --dense-map ssm_out=q5_k,attn_output=q8_0 --dense-imatrix $IM --hc q8_0 --prefill-mmq dense"
for v in 0 100000; do
  QWEN_RUNS=1 QWEN_BENCH_EXTRA="--prompts $O/short.json --timeout 3000 --max-new 16 --prefill auto --spec 8 --spec-min-p 0.5 --extra-args --dflash $D --dflash-vocab $ML $F" $O/qb.sh $O/gg crs$v GUILD_NO_KV_CACHE=1 GUILD_PREFILL_VERIFY_MAX=$v > $O/crs$v.sum 2>&1
done
python3 -c "
import json
for v in (0,100000):
    d=json.load(open('$O/b-crs%d.json'%v)); print('verify_max',v, [(r['prompt_tokens'], round(r['prompt_ms']/1000,2)) for r in d['runs']])" > $O/crs.result
