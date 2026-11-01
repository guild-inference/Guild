#!/bin/sh
# the bench against a guild-kernels-cuda tree ($1, default the perf/prefill-mmvq worktree) -> $2 (default bench_many)
K=${1:-/home/ubuntu/work/kcuda-pf}
O=${2:-/home/ubuntu/work/kcuda-pf-bench/bench_many}
G=/home/ubuntu/work/nemotron/build/_deps/guild_llamacpp-src/ggml
nvcc -O3 -std=c++17 -arch=sm_61 --use_fast_math -I$K/include -I$G/src -I$G/include \
  /home/ubuntu/work/kcuda-pf-bench/bench_many.cu $K/src/math/native_mmvq.cu $K/src/math/iq_kernels.cu -o $O
