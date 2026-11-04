#!/bin/sh
# GLM-5.3-Flash KL screen (text 1): the GGUF's own Q8_0 attention as the base; each arm's output removed once compared
O=/home/ubuntu/work/nemotron/opt/r18
G=/home/ubuntu/models/GLM-5.3-Flash/UD-IQ4_XS/GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf
TK=/home/ubuntu/models/GLM-5.3-Flash/pack/tokenizer
IM=/home/ubuntu/work/imatrix/glm/imatrix_unsloth.gguf
T=${T:-1}
cd /home/ubuntu/Guild
: >> $O/gkl$T.progress
[ -s $O/base$T.f16 ] || GUILD_GLM_KEEP_CACHE=1 GUILD_GLM_DENSE=q8_0 $O/gk2 $G $TK $O/text$T.txt $O/base$T.f16 > $O/base$T.out 2> $O/base$T.err
echo "base $(cat $O/base$T.out)" >> $O/gkl$T.progress
run() { # label dense imatrix(0/1) map
  V=""; [ $3 = 1 ] && V="GUILD_DENSE_IMATRIX=$IM"
  M=""; [ -n "$4" ] && M="GUILD_DENSE_MAP=$4"
  env GUILD_GLM_KEEP_CACHE=1 GUILD_GLM_DENSE=$2 $V $M $O/gk2 $G $TK $O/text$T.txt $O/a.f16 $O/base$T.f16 > $O/$1_$T.out 2> $O/$1_$T.err
  rm -f $O/a.f16
  echo "$1 $(grep -o 'top-1.*max [0-9.]*' $O/$1_$T.out) | $(grep -o 'attention matrices.*' $O/$1_$T.err | head -1)" >> $O/gkl$T.progress
}
for a in $ARMS; do
  case $a in
    q6) run q6 q6_k 0 ;; q6im) run q6im q6_k 1 ;; q5im) run q5im q5_k 1 ;; q4im) run q4im q4_k 1 ;;
    q4imo) run q4imo q4_k 1 attn_output=q8_0 ;;
    q4o5) run q4o5 q4_k 1 attn_output=q5_k ;; q4o6) run q4o6 q4_k 1 attn_output=q6_k ;; q5o6) run q5o6 q5_k 1 attn_output=q6_k ;;
  esac
done
echo done >> $O/gkl$T.progress
