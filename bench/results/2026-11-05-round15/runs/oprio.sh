#!/bin/sh
O=/home/ubuntu/work/nemotron/opt/r19/orn
: > $O/oprio.progress
arm() { # label split
  for P in def more; do
    PA=""; [ $P = more ] && PA="--prompts $O/more.json"
    (cd /home/ubuntu/Guild && GUILD_VERIFY_STREAM_PRIO=$2 /home/ubuntu/lanes/bin/gpu-run .venv/bin/python /home/ubuntu/work/nemotron/guild/tools/bench_runtime.py --exe /home/ubuntu/work/nemotron/opt/r19/gpri \
     --model /home/ubuntu/models/ornith-1.5-35b/Ornith-1.5-35B-Q4_K_M.gguf --pack /mnt/models-ssd/Strata-data/packs/ornith-1.5-35b \
     --mtp /mnt/models-ssd/Strata-data/mtp/ornith-rt --context 32768 --resident-budget-gib 24 --runs 2 --max-new 256 --spec 4 $PA \
     --out $O/b-pr_$1_$P.json > $O/pr_$1_$P.out 2>&1)
    echo "$1 $P $(grep SUMMARY $O/pr_$1_$P.out) $(python3 -c "import json; print([hash(tuple(x['output_ids']))%100000 for x in json.load(open('$O/b-pr_$1_$P.json'))['runs']])")" >> $O/oprio.progress
  done
}
for r in 1 2; do arm s0_$r 0; arm s1_$r 1; done
echo done >> $O/oprio.progress
