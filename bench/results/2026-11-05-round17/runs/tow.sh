#!/bin/sh
# Drama 3's tower formats: loss (teacher-forced on the Q8_0 run's frames, KL of the tower's logits) and speed
O=/home/ubuntu/work/nemotron/opt/r20/tow
E=/home/ubuntu/work/nemotron/opt/r20/gtow
TEXT="The old lighthouse keeper climbed the spiral stairs every evening, counting each step out loud. Ninety-seven steps, he would say, and not one of them has ever changed. Tonight, though, the storm was louder than his voice, and the lamp at the top flickered like it was trying to tell him something."
run() { # label seed [VAR=value ...]
  L=$1; S=$2; shift 2
  (cd /home/ubuntu/Guild && env "$@" /home/ubuntu/lanes/bin/gpu-run $E --native /mnt/models-ssd/Strata-data/models/drama3/Drama3-Q4_K_M.gguf --pack /mnt/models-ssd/Strata-data/packs/drama3 --drama3 /mnt/models-ssd/Strata-data/drama3 --max-context 10240 --kv fp16 --kv-host-only --resident-budget-gib 24 --max-new 2000 --seed $S --wav $O/$L.wav --say "$TEXT" > $O/$L.log 2>&1)
  echo "$L $(grep -o '[0-9.]* frames/s[^)]*' $O/$L.log) $(md5sum < $O/$L.wav | cut -c1-8) $(grep -o 'tower matrices re-quantized.*' $O/$L.log)" >> $O/tow.progress
}
: > $O/tow.progress
run ref 11 GUILD_DRAMA3_DUMP_FRAMES=$O/ref.frames GUILD_DRAMA3_DUMP_LOGITS=$O/ref.logits
for T in q6_k q5_k q4_k; do
  run kl_$T 11 GUILD_DRAMA3_TOWER=$T GUILD_DRAMA3_FORCE=$O/ref.frames GUILD_DRAMA3_DUMP_LOGITS=$O/$T.logits
  /home/ubuntu/Guild/.venv/bin/python $O/kl.py $O/ref.logits $O/$T.logits >> $O/tow.progress
done
for r in 1 2; do for T in q8_0 q6_k q5_k q4_k; do for s in 11 12; do
  if [ $T = q8_0 ]; then run sp_${T}_${s}_$r $s; else run sp_${T}_${s}_$r $s GUILD_DRAMA3_TOWER=$T; fi
done; done; done
echo done >> $O/tow.progress
