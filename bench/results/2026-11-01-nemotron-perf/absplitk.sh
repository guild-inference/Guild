#!/bin/sh
# split against one verify window for 1, 2 and 3 drafts: the 30 s clip, 3 runs + warm-up, alternated twice
P=/home/ubuntu/work/nemotron/perf
M=/home/ubuntu/models/nemotron-3.5-transcribe-guild
R=/home/ubuntu/work/nemotron/build/transcribe_run
O=/home/ubuntu/work/nemotron/opt/absplitk
mkdir -p $O
for i in 1 2; do for k in 1 2 3; do for s in 1 0; do
  GUILD_MOE_TIMING=1 GUILD_SPEC_SPLIT=$s $R --model $M/Nemotron-3.5-Transcribe-LM-Q8_0.gguf --audio $M/audio \
    --tokenizer /home/ubuntu/models/nemotron-3.5-transcribe/tokenizer.json --mtp $k --repeat 4 \
    $P/wav/perf30s-4970-29093-0006.wav --out $O/k$k-s$s-$i.jsonl > $O/k$k-s$s-$i.log 2>&1
done; done; done
