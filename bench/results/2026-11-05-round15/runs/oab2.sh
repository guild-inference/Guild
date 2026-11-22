#!/bin/sh
# Ornith with its server settings, with and without an expert profile (so --expert-cache auto keeps a GPU cache)
O=/home/ubuntu/work/nemotron/opt/r19/orn
: > $O/oab2.progress
arm() { # label profile(0/1)
  PF=""; [ $2 = 1 ] && PF="--expert-profile $O/ornith-profile.bin"
  for P in def more; do
    PA=""; [ $P = more ] && PA="--prompts $O/more.json"
    (cd /home/ubuntu/Guild && /home/ubuntu/lanes/bin/gpu-run .venv/bin/python /home/ubuntu/work/nemotron/guild/tools/bench_runtime.py --exe /home/ubuntu/work/nemotron/opt/r19/gq \
     --model /home/ubuntu/models/ornith-1.5-35b/Ornith-1.5-35B-Q4_K_M.gguf --pack /mnt/models-ssd/Strata-data/packs/ornith-1.5-35b \
     --mtp /mnt/models-ssd/Strata-data/mtp/ornith-rt --context 32768 --resident-budget-gib 24 --runs 2 --max-new 256 --spec 4 $PA \
     --out $O/b2-$1_$P.json --extra-args $PF > $O/2_$1_$P.out 2>&1)
    echo "$1 $P $(grep SUMMARY $O/2_$1_$P.out) | $(grep -o 'GPU cache [0-9]* slots[^;]*' $O/b2-$1_$P.log | head -1)" >> $O/oab2.progress
  done
}
for r in 1 2; do arm prof_$r 1; arm none_$r 0; done
echo done >> $O/oab2.progress
