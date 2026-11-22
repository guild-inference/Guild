#!/bin/sh
# Drama 3's routing on 8 texts unlike the bench's, for an expert profile
O=/home/ubuntu/work/nemotron/opt/r20/d3
rm -f $O/route.bin
i=0
while IFS= read -r T; do
  i=$((i+1))
  (cd /home/ubuntu/Guild && /home/ubuntu/lanes/bin/gpu-run /home/ubuntu/work/nemotron/opt/r19/gq --native /mnt/models-ssd/Strata-data/models/drama3/Drama3-Q4_K_M.gguf --pack /mnt/models-ssd/Strata-data/packs/drama3 --drama3 /mnt/models-ssd/Strata-data/drama3 --max-context 10240 --kv fp16 --kv-host-only --resident-budget-gib 24 --max-new 2000 --seed $i --wav $O/t$i.wav --dump-routing $O/route.bin --say "$T" > $O/t$i.log 2>&1)
  echo "$i $(ls -la $O/route.bin | awk '{print $5}') $(grep -o '[0-9.]* frames/s' $O/t$i.log)" >> $O/trace.progress
done < $O/texts.txt
echo done >> $O/trace.progress
