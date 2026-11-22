#!/bin/sh
# Drama 3, the bench text, with and without the expert profile (gq: the build with the Q4_K/Q6_K GPU expert rows)
O=/home/ubuntu/work/nemotron/opt/r20/d3
TEXT="The old lighthouse keeper climbed the spiral stairs every evening, counting each step out loud. Ninety-seven steps, he would say, and not one of them has ever changed. Tonight, though, the storm was louder than his voice, and the lamp at the top flickered like it was trying to tell him something."
: > $O/cab.progress
run() { # label profile(0/1) seed
  PF=""; [ $2 = 1 ] && PF="--expert-profile $O/d3-profile.bin"
  (cd /home/ubuntu/Guild && /home/ubuntu/lanes/bin/gpu-run /home/ubuntu/work/nemotron/opt/r19/gq --native /mnt/models-ssd/Strata-data/models/drama3/Drama3-Q4_K_M.gguf --pack /mnt/models-ssd/Strata-data/packs/drama3 --drama3 /mnt/models-ssd/Strata-data/drama3 --max-context 10240 --kv fp16 --kv-host-only --resident-budget-gib 24 --max-new 2000 --seed $3 --wav $O/$1.wav $PF --say "$TEXT" > $O/$1.log 2>&1)
  echo "$1 $(grep -o '[0-9.]* frames/s[^)]*' $O/$1.log) $(md5sum < $O/$1.wav | cut -c1-8) | $(grep -o 'GPU cache [0-9]* slots[^;]*' $O/$1.log | head -1)" >> $O/cab.progress
}
for r in 1 2; do run c1_11_$r 1 11; run c0_11_$r 0 11; done
echo done >> $O/cab.progress
