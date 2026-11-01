#!/bin/sh
# drafted passes against a greedy pass's logits (TRANSCRIBE_CHECK_SPEC), the two perf clips, 1-3 drafts
P=/home/ubuntu/work/nemotron/perf
M=/home/ubuntu/models/nemotron-3.5-transcribe-guild
R=${R:-/home/ubuntu/work/nemotron/build/transcribe_run}
O=/home/ubuntu/work/nemotron/opt
T=${TAG:-new}
for k in 1 2 3; do
  TRANSCRIBE_CHECK_SPEC=1 $R --model $M/Nemotron-3.5-Transcribe-LM-Q8_0.gguf --audio $M/audio \
    --tokenizer /home/ubuntu/models/nemotron-3.5-transcribe/tokenizer.json --mtp $k --clips $P/clips.jsonl \
    --out $O/$T-spec$k.jsonl > $O/$T-spec$k.log 2>&1
  echo "mtp$k rc $? check lines $(grep -c '^check:' $O/$T-spec$k.log)" >> $O/$T-spec.progress
done
