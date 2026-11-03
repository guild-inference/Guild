#!/bin/sh
# Ornith-1.5-35B with its MTP layer: --spec 3..6 x --spec-min-p 0.3 / 0.5 / 0.7, prose and code, 1 run x 256 tokens
O=/home/ubuntu/work/nemotron/opt/r14
: > $O/orn.progress
for s in 3 4 5 6; do for p in 0.3 0.5 0.7; do
  (cd /home/ubuntu/Guild && /home/ubuntu/lanes/bin/gpu-run .venv/bin/python /home/ubuntu/work/nemotron/guild/tools/bench_runtime.py --exe $O/gg4 --model /home/ubuntu/models/ornith-1.5-35b/Ornith-1.5-35B-Q4_K_M.gguf --pack /mnt/models-ssd/Strata-data/packs/ornith-1.5-35b --mtp /mnt/models-ssd/Strata-data/mtp/ornith-rt --context 32768 --resident-budget-gib 24 --runs 1 --max-new 256 --spec $s --spec-min-p $p --out $O/b-orn_${s}_$p.json > $O/orn_${s}_$p.out 2>&1)
  echo "spec $s min-p $p $(grep SUMMARY $O/orn_${s}_$p.out) $(grep -h 'ms/window' $O/b-orn_${s}_$p.log | tail -2 | sed -E 's/.*windows, ([0-9.]+) tokens\/window, ([0-9.]+) ms\/window.*/\1 tok\/win \2 ms/' | tr '\n' ';')" >> $O/orn.progress
done; done
echo done >> $O/orn.progress
