#!/bin/sh
# Drama 3 under nsys (CUDA kernels, graph nodes), one short text
O=/home/ubuntu/work/nemotron/opt/r20/ns
cd /home/ubuntu/Guild && /home/ubuntu/lanes/bin/gpu-run nsys profile -t cuda --cuda-graph-trace=node -o $O/d3 -f true --export sqlite /home/ubuntu/work/nemotron/opt/r20/gnew --native /mnt/models-ssd/Strata-data/models/drama3/Drama3-Q4_K_M.gguf --pack /mnt/models-ssd/Strata-data/packs/drama3 --drama3 /mnt/models-ssd/Strata-data/drama3 --max-context 10240 --kv fp16 --kv-host-only --resident-budget-gib 24 --max-new 2000 --seed 11 --wav $O/d3.wav --say "Tonight, though, the storm was louder than his voice, and the lamp at the top flickered." > $O/d3.log 2>&1
echo done >> $O/d3.log
