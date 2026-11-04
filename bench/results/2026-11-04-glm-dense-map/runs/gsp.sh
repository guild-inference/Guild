#!/bin/sh
# GLM-5.3-Flash decode speed by attention format: the four chat prompts, 256 tokens each, NextN drafts, 262,144-token
# context, 2 rounds alternated (each arm its own texts: compare the step medians); the page cache kept between loads
O=/home/ubuntu/work/nemotron/opt/r18
P=/home/ubuntu/work/dflash/glm/prompts.txt
: > $O/gsp.progress
arm() { # label glm5_run format flag map
  M=""; [ -n "$3" ] && M="GUILD_DENSE_MAP=$3"
  /home/ubuntu/lanes/bin/glm-run $O/gr $O/gsp_$1.out 0 256 --ctx 262144 --spec $2 --prompts $P -- GUILD_GLM_KEEP_CACHE=1 GLM5_RUN_IGNORE_EOS=1 $M > /dev/null 2>&1
  echo "$1 $(grep -o 'step median [0-9.]*; [0-9.]* tok/s' $O/gsp_$1.out | tr '\n' ' ') | $(grep -o 'VRAM:.*' $O/gsp_$1.out | head -1)" >> $O/gsp.progress
}
for r in 1 2; do
  arm q6_$r --q6k ""; arm q5_$r --q5k ""; arm q5o6_$r --q5k attn_output=q6_k; arm q4_$r --q4k ""; arm q4o6_$r --q4k attn_output=q6_k; arm q4o8_$r --q4k attn_output=q8_0
done
echo done >> $O/gsp.progress
