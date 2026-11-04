#!/bin/sh
# prefill-only timing of the 30 s clip: 5 runs in one process, the CPU experts' phase totals
P=/home/ubuntu/work/nemotron/perf
M=/home/ubuntu/models/nemotron-3.5-transcribe-guild
R=${R:-/home/ubuntu/work/nemotron/build/transcribe_run}
O=/home/ubuntu/work/nemotron/opt

GUILD_MOE_TIMING=1 $R --model $M/Nemotron-3.5-Transcribe-LM-Q8_0.gguf --audio $M/audio \
  --tokenizer /home/ubuntu/models/nemotron-3.5-transcribe/tokenizer.json --no-mtp --max-new 1 --repeat 5 \
  $P/wav/perf30s-4970-29093-0006.wav --out $O/${TAG:-base}-prefill.jsonl > $O/${TAG:-base}-prefill.log 2>&1
echo "rc $?" > $O/${TAG:-base}-prefill.rc
