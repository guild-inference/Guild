#!/bin/sh
# Ornith-1.5-35B Q4_K_M prefill, 1024-token prompt, batch 256: pairs on / off, alternated twice (ornith_prefill_bench)
B=/home/ubuntu/work/nemotron/build/ornith_prefill_bench
O=/home/ubuntu/work/nemotron/opt/ornith
cd /home/ubuntu/Guild
for i in 1 2; do for arm in pairs off; do
  case $arm in off) E="GUILD_PREFILL_PAIRS=0";; *) E="X=1";; esac
  env GUILD_NO_KV_CACHE=1 $E $B --prompt 1024 --generate 16 --runs 3 --batch 256 --context 4096 \
    --csv $O/$arm-$i.csv --json-summary $O/$arm-$i.json > $O/$arm-$i.log 2>&1
  echo "$arm $i rc $?" >> $O/progress
done; done
