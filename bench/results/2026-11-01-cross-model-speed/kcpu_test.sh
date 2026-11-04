#!/bin/sh
# the kernel repo's own tests on perf/q8-avx512 (standalone build: SDK 1.3, ggml-cpu AVX-512 as the main checkout's)
K=/home/ubuntu/work/kcpu-q8
if [ ! -f $K/build/CMakeCache.txt ]; then
  cmake -S $K -B $K/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/home/ubuntu/guild-sdk-1.3 \
    -DGUILD_CPU_GGML_AVX512=ON -DGGML_NATIVE=OFF \
    -DFETCHCONTENT_SOURCE_DIR_LLAMA=/home/ubuntu/guild-kernels-cpu/build/_deps/llama-src > /home/ubuntu/work/kcpu-q8-run/configure.log 2>&1 || { echo "configure $?" > /home/ubuntu/work/kcpu-q8-run/test.rc; exit 1; }
fi
cmake --build $K/build -j 48 > /home/ubuntu/work/kcpu-q8-run/build.log 2>&1 || { echo "build $?" > /home/ubuntu/work/kcpu-q8-run/test.rc; exit 1; }
(cd $K/build && ctest --output-on-failure) > /home/ubuntu/work/kcpu-q8-run/ctest.log 2>&1
echo "ctest $?" > /home/ubuntu/work/kcpu-q8-run/test.rc
GUILD_Q8_512=0 $K/build/iq_ggml_order_test > /home/ubuntu/work/kcpu-q8-run/order-avx2.log 2>&1
echo "avx2 order $?" >> /home/ubuntu/work/kcpu-q8-run/test.rc
$K/build/iq_ggml_order_test > /home/ubuntu/work/kcpu-q8-run/order-default.log 2>&1
echo "default order $?" >> /home/ubuntu/work/kcpu-q8-run/test.rc
