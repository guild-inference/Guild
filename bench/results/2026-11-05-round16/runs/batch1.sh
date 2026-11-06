#!/bin/sh
# 1. Drama 3 CPU pool profile with the new pool (cache off / on), 2. Ornith and 3. Qwen, new (gnew) against
# before the pool and placement changes (gpri), memory in its normal state; 4. Drama 3 under nsys.
O=/home/ubuntu/work/nemotron/opt/r20
TEXT="The old lighthouse keeper climbed the spiral stairs every evening, counting each step out loud. Ninety-seven steps, he would say, and not one of them has ever changed. Tonight, though, the storm was louder than his voice, and the lamp at the top flickered like it was trying to tell him something."
: > $O/batch1.progress
for PF in none prof; do
  X=""; [ $PF = prof ] && X="--expert-profile $O/d3/d3-profile.bin"
  (cd /home/ubuntu/Guild && GUILD_CPU_PROFILE=$O/d3/nprof_$PF.csv /home/ubuntu/lanes/bin/gpu-run $O/gnew --native /mnt/models-ssd/Strata-data/models/drama3/Drama3-Q4_K_M.gguf --pack /mnt/models-ssd/Strata-data/packs/drama3 --drama3 /mnt/models-ssd/Strata-data/drama3 --max-context 10240 --kv fp16 --kv-host-only --resident-budget-gib 24 --max-new 2000 --seed 11 --wav $O/d3/nprof_$PF.wav $X --say "$TEXT" > $O/d3/nprof_$PF.log 2>&1)
  echo "drama3 profile $PF $(grep -o '[0-9.]* frames/s' $O/d3/nprof_$PF.log)" >> $O/batch1.progress
done
for r in 1 2; do for a in new old; do
  E=$O/gnew; [ $a = old ] && E=/home/ubuntu/work/nemotron/opt/r19/gpri
  (cd /home/ubuntu/Guild && /home/ubuntu/lanes/bin/gpu-run .venv/bin/python /home/ubuntu/work/nemotron/guild/tools/bench_runtime.py --exe $E \
   --model /home/ubuntu/models/ornith-1.5-35b/Ornith-1.5-35B-Q4_K_M.gguf --pack /mnt/models-ssd/Strata-data/packs/ornith-1.5-35b \
   --mtp /mnt/models-ssd/Strata-data/mtp/ornith-rt --context 32768 --resident-budget-gib 24 --runs 2 --max-new 256 --spec 4 \
   --out $O/b-orn_${a}_$r.json > $O/orn_${a}_$r.out 2>&1)
  echo "ornith $a $r $(grep SUMMARY $O/orn_${a}_$r.out) $(python3 -c "import json; print([hash(tuple(x['output_ids']))%100000 for x in json.load(open('$O/b-orn_${a}_$r.json'))['runs']])")" >> $O/batch1.progress
done; done
D=/home/ubuntu/work/dflash/drafter
ML=/home/ubuntu/work/nemotron/guild/data/dflash_vocab.bin
IM=/home/ubuntu/work/imatrix/dl/imatrix_unsloth.gguf
F="--dense q4_k --dense-map ssm_out=q5_k,attn_output=q8_0 --dense-imatrix $IM --hc q8_0 --prefill-mmq dense"
for r in 1 2; do for a in new old; do
  E=$O/gnew; [ $a = old ] && E=/home/ubuntu/work/nemotron/opt/r19/gpri
  QWEN_RUNS=2 QWEN_BENCH_EXTRA="--prefill auto --spec 8 --spec-min-p 0.5 --extra-args --dflash $D --dflash-vocab $ML $F" $O/qb.sh $E qn${a}_$r > $O/qn${a}_$r.sum 2>&1
  echo "qwen $a $r node0 free $(numastat -m | awk '/MemFree/{print $2}') $(grep SUMMARY $O/qn${a}_$r.sum) $(python3 -c "import json; print([hash(tuple(x['output_ids']))%100000 for x in json.load(open('$O/b-qn${a}_$r.json'))['runs']])")" >> $O/batch1.progress
done; done
$O/ns/d3nsys.sh
echo "nsys $(tail -1 $O/ns/d3.log)" >> $O/batch1.progress
echo done >> $O/batch1.progress
