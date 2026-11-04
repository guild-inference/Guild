#!/bin/sh
# model 2's speed: the 5 s and 30 s clips, greedy and with 1-3 MTP drafts, 6 runs each in one process (the first a
# warm-up, dropped); one JSON line per run (transcribe_run --out)
P=/home/ubuntu/work/nemotron/perf
M=/home/ubuntu/models/nemotron-3.5-transcribe-guild
R=/home/ubuntu/work/nemotron/build/transcribe_run
for mode in greedy mtp1 mtp2 mtp3; do
  case $mode in greedy) E="--no-mtp";; *) E="--mtp ${mode#mtp}";; esac
  $R --model $M/Nemotron-3.5-Transcribe-LM-Q8_0.gguf --audio $M/audio --tokenizer /home/ubuntu/models/nemotron-3.5-transcribe/tokenizer.json \
    $E --repeat 6 --clips $P/clips.jsonl --out $P/perf-$mode.jsonl > $P/perf-$mode.log 2>&1
  echo "$mode rc $?" >> $P/progress
done
