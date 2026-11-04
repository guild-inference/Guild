#!/bin/sh
# GLM host latents: the same ids with most of the latent rows in host memory (16k prompt), then a 262,144-token
# context with a 32k prompt against the all-VRAM 32,768-token context
B=/home/ubuntu/work/nemotron/build-cu/glm5_run
M=/home/ubuntu/models/GLM-5.3-Flash/UD-IQ4_XS/GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf
P=/home/ubuntu/lanes/g3/prompts
D="$(dirname "$0")"
cd $D
one() { # name env... -- args
  n=$1; shift
  echo "$(date -u +%H:%M:%S) start $n" >> progress
  env "$@" > $n.out 2> $n.log
  echo "$(date -u +%H:%M:%S) $n rc $?" >> progress
}
one vram16 GLM5_RUN_STEPS=1 $B $M @$P/ctx16000.ids 48 --spec --ctx 32768 --logits vram16.f32
one host16 GLM5_RUN_STEPS=1 GUILD_GLM_KV_NEAR=4096 $B $M @$P/ctx16000.ids 48 --spec --ctx 32768 --logits host16.f32
one host256 GLM5_RUN_STEPS=1 $B $M @$P/ctx32000.ids 64 --spec --ctx 262144 --logits host256.f32
one vram32 GLM5_RUN_STEPS=1 $B $M @$P/ctx32000.ids 64 --spec --ctx 32768 --logits vram32.f32
echo done >> progress
