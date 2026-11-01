#!/bin/sh
# Qwen3.8-Flash-Next, the 4K prompt: --prefill 512 against --prefill 1024 --prefill-rows 512, at the bench's 32,768
# context and at the server's 262,144 with the host-only KV cache: the same text, and the prompt's time
W=/home/ubuntu/work/nemotron
QB=/home/ubuntu/lanes/int3/tools/qb.sh
O=$W/opt/qwenrows
one() { # label context prefill extra-engine-args...
  l=$1; c=$2; n=$3; shift 3
  QWEN_CONTEXT=$c QWEN_RUNS=2 QWEN_BENCH_EXTRA="--prompts $W/opt/qwen4k.json --timeout 3000 --prefill $n --extra-args $*" \
    $QB $W/build-cu/guild-generate $O $l GUILD_NO_KV_CACHE=1 GUILD_PREFILL_TIMING=1
  echo "$l rc $?" >> $O/progress
}
one c512 32768 512 --prefill-rows 0
one r1024 32768 1024 --prefill-rows 512
one c512k 262144 512 --kv-host-only --prefill-rows 0
one r1024k 262144 1024 --kv-host-only --prefill-rows 512
