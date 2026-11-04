#!/bin/sh
# Drama 3: the tower's codes 2..9 beside the next backbone row (default) against GUILD_DRAMA3_OVERLAP=0, alternated
O=/home/ubuntu/work/nemotron/opt/r15
TEXT="The old lighthouse keeper climbed the spiral stairs every evening, counting each step out loud. Ninety-seven steps, he would say, and not one of them has ever changed. Tonight, though, the storm was louder than his voice, and the lamp at the top flickered like it was trying to tell him something."
V=/mnt/models-ssd/Strata-data/drama3/voices
: > $O/ab.progress
run() { # label overlap seed voice-args...
  l=$1; ov=$2; seed=$3; shift 3
  (cd /home/ubuntu/Guild && GUILD_DRAMA3_OVERLAP=$ov $O/gg --native /mnt/models-ssd/Strata-data/models/drama3/Drama3-Q4_K_M.gguf --pack /mnt/models-ssd/Strata-data/packs/drama3 --drama3 /mnt/models-ssd/Strata-data/drama3 --max-context 10240 --kv fp16 --kv-host-only --resident-budget-gib 24 --max-new 2000 --seed $seed --wav $O/$l.wav "$@" --say "$TEXT" > $O/$l.log 2>&1)
  echo "$l $(grep -o '[0-9.]* s of audio.*' $O/$l.log)" >> $O/ab.progress
}
for r in 1 2; do
  for seed in 11 12; do
    run none_on_${seed}_$r 1 $seed
    run none_off_${seed}_$r 0 $seed
  done
  run en0_off_11_$r 0 11 --ref-wav $V/en0.wav --ref-text "$(cat $V/en0.txt)"
  run en0_on_11_$r 1 11 --ref-wav $V/en0.wav --ref-text "$(cat $V/en0.txt)"
done
echo done >> $O/ab.progress
