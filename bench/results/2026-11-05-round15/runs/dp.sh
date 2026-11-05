#!/bin/sh
O=/home/ubuntu/work/nemotron/opt/r19
TEXT="The old lighthouse keeper climbed the spiral stairs every evening, counting each step out loud. Ninety-seven steps, he would say, and not one of them has ever changed. Tonight, though, the storm was louder than his voice, and the lamp at the top flickered like it was trying to tell him something."
: > $O/dp.progress
run() { # label prio seed
  (cd /home/ubuntu/Guild && GUILD_VERIFY_STREAM_PRIO=$2 $O/gp --native /mnt/models-ssd/Strata-data/models/drama3/Drama3-Q4_K_M.gguf --pack /mnt/models-ssd/Strata-data/packs/drama3 --drama3 /mnt/models-ssd/Strata-data/drama3 --max-context 10240 --kv fp16 --kv-host-only --resident-budget-gib 24 --max-new 2000 --seed $3 --wav $O/$1.wav --say "$TEXT" > $O/$1.log 2>&1)
  echo "$1 $(grep -o '[0-9.]* frames/s[^)]*' $O/$1.log) $(md5sum < $O/$1.wav | cut -c1-8)" >> $O/dp.progress
}
for r in 1 2; do for s in 11 12; do run p1_${s}_$r 1 $s; run p0_${s}_$r 0 $s; done; done
echo done >> $O/dp.progress
