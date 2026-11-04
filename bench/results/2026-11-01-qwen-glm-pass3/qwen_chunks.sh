#!/bin/sh
# Qwen3.8-Flash-Next: a ~4K-token prompt with prefill chunks of 512 (the server's), 1024 and 2048 tokens; every
# chunk streams the experts the GPU does not hold, so a larger chunk reads them for more tokens.  Then 512 with the
# RAM copy of the experts page-locked (GUILD_PIN_EXPERTS=1: DMA straight from it).  The same text?
W=/home/ubuntu/work/nemotron
QB=/home/ubuntu/lanes/int3/tools/qb.sh
O=$W/opt/qwenchunk
one() { # label prefill VAR=value...
  l=$1; n=$2; shift 2
  echo "$(date -u +%H:%M:%S) start $l" >> $O/progress
  QWEN_RUNS=1 QWEN_BENCH_EXTRA="--prompts $W/opt/qwen4k.json --timeout 3000 --prefill $n" \
    $QB $W/build-cu/guild-generate $O $l GUILD_NO_KV_CACHE=1 GUILD_PREFILL_TIMING=1 "$@"
  echo "$(date -u +%H:%M:%S) $l rc $?" >> $O/progress
}
one c512 512
one c1024 1024
one c2048 2048
one pin512 512 GUILD_PIN_EXPERTS=1
one c512b 512
echo done >> $O/progress
