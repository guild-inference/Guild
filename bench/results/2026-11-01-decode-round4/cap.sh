#!/bin/sh
# nsys captures of decode with the current build: Qwen3.8-Flash-Next at the server's settings (prose and code
# requests), GLM-5.3-Flash (5-token prompt, 64 tokens --spec)
D=/home/ubuntu/work/nemotron/opt/prof4
B=/home/ubuntu/work/nemotron/build-cu
cd /home/ubuntu/Guild
GUILD_DECODE_TIMING=1 .venv/bin/python /home/ubuntu/lanes/s1/prof/drive.py $D/qwen_req.txt $D/q.err -- \
 nsys profile -t cuda --cuda-graph-trace=node -s none --cpuctxsw=none -f true -o $D/q \
 $B/guild-generate --serve \
 --native /mnt/models-ssd/Strata-data/models/unsloth-UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf \
 --pack /mnt/models-ssd/Strata-data/packs/unsloth-ud-iq4_xs --tokenizer /mnt/models-ssd/Strata-data/packs/unsloth-ud-iq4_xs/tokenizer \
 --max-context 262144 --kv fp16 --kv-host-only --spec 4 --expert-cache auto --prefill 512 \
 --expert-profile /home/ubuntu/Guild/data/expert-profile.bin --mtp /mnt/models-ssd/Strata-data/mtp/rt --resident-budget-gib 56 \
 > $D/q.out 2>&1
echo "qwen $?" >> $D/progress
GUILD_GLM_TIMING=1 GLM5_RUN_STEPS=1 nsys profile -t cuda -s none --cpuctxsw=none -f true -o $D/g \
 $B/glm5_run /home/ubuntu/models/GLM-5.3-Flash/UD-IQ4_XS/GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf \
 785,6722,315,9621,374 64 --spec --ctx 32768 > $D/g.out 2> $D/g.err
echo "glm $?" >> $D/progress
