#!/bin/sh
# glm2.sh: GLM-5.3-Flash, round 19 against round 20 in separate loads, B I B: base20's glm5_run (one arm, 3 rounds),
# then glm.sh (lane20/int's build, arms C = round 19's GPU path and A = defaults, 3 rounds), then base20 again.  The
# verifier's two prompts (3K tokens and 5 tokens), --spec, --ctx 32768.  Each load holds the machine lock itself.
O=/home/ubuntu/lanes/r20/int/glm; V=/home/ubuntu/lanes/r20/glm-attend3/verify; B=/home/ubuntu/lanes/r20/base20
mkdir -p $O
b() { echo "$1 start $(date -u +%T)" >> $O/progress
  /home/ubuntu/lanes/bin/glm-run $B/glm5_run $O/$1.out 0 256 --ctx 32768 --spec --prompts $V/prompts.txt --reps 3 \
    -- GLM5_RUN_STEPS=1 > $O/$1.summary 2>&1
  echo "$1 rc $? $(date -u +%T)" >> $O/progress; }
b b1
echo "int start $(date -u +%T)" >> $O/progress
/home/ubuntu/lanes/r20/int/glm.sh $O
echo "int done $(date -u +%T)" >> $O/progress
b b2
echo "done $(date -u +%T)" >> $O/progress
