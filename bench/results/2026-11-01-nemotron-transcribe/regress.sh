#!/bin/sh
# regress.sh <branch build dir> <out dir>: the branch against build-cuda12 (feat/glm5), run whole under ONE gpu-run.
#   1. the branch build's ctest suite (GPU tests included);
#   2. Qwen3.8-Flash-Next: qwen-bench's runs (server flags, prose + code, 3 x 256 tokens) of build-cuda12's
#      guild-generate and the branch's, A B B A; texts compared;
#   3. GLM-5.3-Flash: glm5_run (glm-run's defaults) base / branch, A B B A B A; and guild-generate on GLM (the runtime's
#      GLM glue: glm5_generate) base / branch, the same prompt, ids compared;
#   4. session_score_test on GLM-5.3-Flash (GuildSession::score through the GLM engine).
B=$1
O=$2
BASE=/home/ubuntu/Guild/build-cuda12
QB=/home/ubuntu/lanes/int3/tools/qb.sh
GR=/home/ubuntu/lanes/int3/tools/gr.sh
GLM=/home/ubuntu/models/GLM-5.3-Flash/UD-IQ4_XS/GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf
GTOK=/home/ubuntu/models/GLM-5.3-Flash/pack/tokenizer
mkdir -p $O
log() { echo "$(date -u '+%F %T') $*" >> $O/progress; }
log start
(cd $B && ctest --output-on-failure --timeout 3600) > $O/ctest.log 2>&1
log "ctest rc $?"
$QB $BASE/guild-generate $O qwen-base1 > /dev/null 2>&1; log qwen-base1
$QB $B/guild-generate $O qwen-new1 > /dev/null 2>&1; log qwen-new1
$QB $B/guild-generate $O qwen-new2 > /dev/null 2>&1; log qwen-new2
$QB $BASE/guild-generate $O qwen-base2 > /dev/null 2>&1; log qwen-base2
for r in base1 new1 new2 base2 new3 base3; do
  case $r in base*) E=$BASE/glm5_run ;; *) E=$B/glm5_run ;; esac
  $GR $E $O/glm-$r.out > /dev/null 2>&1; log glm-$r
done
for r in base new; do
  case $r in base) E=$BASE/guild-generate ;; *) E=$B/guild-generate ;; esac
  (cd /home/ubuntu/Guild && $E --native $GLM --tokenizer $GTOK --max-context 8192 --tokens 785,6722,315,9621,374 \
    --max-new 128 > $O/glm-gen-$r.out 2>&1); log "glm-gen-$r rc $?"
done
$B/session_score_test --native $GLM --tokenizer $GTOK --max-context 8192 > $O/sst-glm.log 2>&1
log "sst-glm rc $?"
log end
touch $O/done
