#!/bin/sh
# cpu_moe's down items of 16 (the default) and 64 rows: Nemotron 3.5 Transcribe (the speed clips, greedy and three
# drafts, 6 runs a process, the first dropped; processes alternated twice), then GLM-5.3-Flash's 2,051-token prompt
# (one process, alternated)
P=/home/ubuntu/work/nemotron/perf
M=/home/ubuntu/models/nemotron-3.5-transcribe-guild
R=/home/ubuntu/work/nemotron/build-cu/transcribe_run
D=/home/ubuntu/work/nemotron/opt/drb
cd $D
for round in 1 2; do for drb in 16 64; do for mode in greedy mtp3; do
  case $mode in greedy) E="--no-mtp";; *) E="--mtp 3";; esac
  GUILD_MOE_DRB=$drb $R --model $M/Nemotron-3.5-Transcribe-LM-Q8_0.gguf --audio $M/audio \
    --tokenizer /home/ubuntu/models/nemotron-3.5-transcribe/tokenizer.json $E --repeat 6 --clips $P/clips.jsonl \
    --out $D/nem-$mode-$drb-$round.jsonl > $D/nem-$mode-$drb-$round.log 2>&1
  echo "nem $mode $drb $round rc $?" >> progress
done; done; done
B=/home/ubuntu/work/nemotron/build-cu/glm5_run
GM=/home/ubuntu/models/GLM-5.3-Flash/UD-IQ4_XS/GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf
GUILD_GLM_TIMING=1 $B $GM 0 16 --ctx 32768 --prompts ../glmiq4/prompts.txt --ab GUILD_MOE_DRB=64,16 --reps 3 > glmpf.out 2> glmpf.log
echo "glm rc $?" >> progress
