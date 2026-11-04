#!/bin/sh
# glm5_kernels_test: the tests (the attention kernels bit for bit, the gathers from host memory), then the attention
# bench (legacy / new kernel, each query's rows / the union's from mapped host memory) and the indexer bench to 256K.
# The layouts compared while choosing the kernel (padded, swizzled, one or three blocks an SM) were bench-only
# builds; runs/attend-v-*.out are their second rounds.
T=${T:-/home/ubuntu/work/nemotron/build-cu/glm5_kernels_test}
$T > kernels-test.out 2>&1; echo "test rc $?"
$T attend > attend-gather.out 2>&1
$T indexer > indexer.out 2>&1
