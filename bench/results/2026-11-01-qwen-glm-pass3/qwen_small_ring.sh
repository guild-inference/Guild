#!/bin/sh
# Qwen3.8-Flash-Next, the 4K prompt: chunks of 1024 and 768 with the small staging ring (GUILD_PREFILL_STREAM_MIN
# above the chunk: the routed experts staged as at 512, not the stream-everything ring), against 512
W=/home/ubuntu/work/nemotron
QB=/home/ubuntu/lanes/int3/tools/qb.sh
O=$W/opt/qwenchunk2
one() { # label prefill VAR=value...
  l=$1; n=$2; shift 2
  echo "$(date -u +%H:%M:%S) start $l" >> $O/progress
  QWEN_RUNS=1 QWEN_BENCH_EXTRA="--prompts $W/opt/qwen4k.json --timeout 3000 --prefill $n" \
    $QB $W/build-cu/guild-generate $O $l GUILD_NO_KV_CACHE=1 GUILD_PREFILL_TIMING=1 "$@"
  echo "$(date -u +%H:%M:%S) $l rc $?" >> $O/progress
}
one s1024 1024 GUILD_PREFILL_STREAM_MIN=100000
one s768 768 GUILD_PREFILL_STREAM_MIN=100000
echo done >> $O/progress
