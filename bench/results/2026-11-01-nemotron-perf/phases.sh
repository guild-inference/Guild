#!/bin/sh
# CPU expert phase totals of the drafted decode: the 30 s clip with 3 drafts against its prompt alone (--max-new 1),
# 4 runs each in one process (the first a warm-up, counted in both)
P=/home/ubuntu/work/nemotron/perf
M=/home/ubuntu/models/nemotron-3.5-transcribe-guild
R=${R:-/home/ubuntu/work/nemotron/build/transcribe_run}
O=/home/ubuntu/work/nemotron/opt
T=${TAG:-new}
for n in 1 400; do
  GUILD_MOE_TIMING=1 $R --model $M/Nemotron-3.5-Transcribe-LM-Q8_0.gguf --audio $M/audio \
    --tokenizer /home/ubuntu/models/nemotron-3.5-transcribe/tokenizer.json --mtp ${K:-3} --max-new $n --repeat 4 \
    $P/wav/perf30s-4970-29093-0006.wav --out $O/$T-phases$n.jsonl > $O/$T-phases$n.log 2>&1
done
