#!/bin/sh
# Qwen3.8-Flash-Next prefill of a ~4K-token prompt: pairs on / off, alternated (qb.sh, 2 runs each, no prefix cache)
W=/home/ubuntu/work/nemotron
QB=/home/ubuntu/lanes/int3/tools/qb.sh
O=$W/opt/qwenlong
mkdir -p $O
export QWEN_RUNS=2 QWEN_BENCH_EXTRA="--prompts $W/opt/qwen4k.json --timeout 3000"
for i in 1 2; do
  $QB $W/build/guild-generate $O pairs$i GUILD_NO_KV_CACHE=1
  $QB $W/build/guild-generate $O off$i GUILD_NO_KV_CACHE=1 GUILD_PREFILL_PAIRS=0
done
