#!/bin/sh
# Drama 3: the tower's codes 2..9 beside the next backbone row (default) against GUILD_DRAMA3_OVERLAP=0, alternated
O=/home/ubuntu/work/nemotron/opt/r15
TEXT="The old lighthouse keeper climbed the spiral stairs every evening, counting each step out loud. Ninety-seven steps, he would say, and not one of them has ever changed. Tonight, though, the storm was louder than his voice, and the lamp at the top flickered like it was trying to tell him something."
V=/mnt/models-ssd/Strata-data/drama3/voices
: > $O/ab2.progress
run() { # label overlap seed voice-args...
  l=$1; ov=$2; seed=$3; shift 3
  (cd /home/ubuntu/Guild && GUILD_DRAMA3_OVERLAP=$ov $O/$EXE --native /mnt/models-ssd/Strata-data/models/drama3/Drama3-Q4_K_M.gguf --pack /mnt/models-ssd/Strata-data/packs/drama3 --drama3 /mnt/models-ssd/Strata-data/drama3 --max-context 10240 --kv fp16 --kv-host-only --resident-budget-gib 24 --max-new 2000 --seed $seed --wav $O/$l.wav "$@" --say "$TEXT" > $O/$l.log 2>&1)
  echo "$l $(grep -o '[0-9.]* s of audio.*' $O/$l.log)" >> $O/ab2.progress
}
for r in 1 2; do
  for seed in 11 12; do
    EXE=gg2 run b_${seed}_$r 1 $seed
    EXE=gg run t_${seed}_$r 1 $seed
    EXE=gg2 run o_${seed}_$r 0 $seed
  done
  EXE=gg2 run en0b_11_$r 1 11 --ref-wav $V/en0.wav --ref-text "$(cat $V/en0.txt)"
  EXE=gg run en0t_11_$r 1 11 --ref-wav $V/en0.wav --ref-text "$(cat $V/en0.txt)"
done
echo done >> $O/ab2.progress
