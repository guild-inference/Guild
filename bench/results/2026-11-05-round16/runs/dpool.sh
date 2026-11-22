#!/bin/sh
# Drama 3: the pool before (gpri) and after (gpool) the claim changes; same text, two seeds, 2 rounds alternated
O=/home/ubuntu/work/nemotron/opt/r20
TEXT="The old lighthouse keeper climbed the spiral stairs every evening, counting each step out loud. Ninety-seven steps, he would say, and not one of them has ever changed. Tonight, though, the storm was louder than his voice, and the lamp at the top flickered like it was trying to tell him something."
: > $O/dpool.progress
run() { # label exe seed
  (cd /home/ubuntu/Guild && /home/ubuntu/lanes/bin/gpu-run $2 --native /mnt/models-ssd/Strata-data/models/drama3/Drama3-Q4_K_M.gguf --pack /mnt/models-ssd/Strata-data/packs/drama3 --drama3 /mnt/models-ssd/Strata-data/drama3 --max-context 10240 --kv fp16 --kv-host-only --resident-budget-gib 24 --max-new 2000 --seed $3 --wav $O/$1.wav --say "$TEXT" > $O/$1.log 2>&1)
  echo "$1 $(grep -o '[0-9.]* frames/s[^)]*' $O/$1.log) $(md5sum < $O/$1.wav | cut -c1-8)" >> $O/dpool.progress
}
for r in 1 2; do for s in 11 12; do run new_${s}_$r $O/gpool $s; run old_${s}_$r /home/ubuntu/work/nemotron/opt/r19/gpri $s; done; done
echo done >> $O/dpool.progress
