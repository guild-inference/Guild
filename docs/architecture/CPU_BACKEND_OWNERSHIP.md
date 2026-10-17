# CPU source ownership and bundled/update verification

The main native runtime now consumes CPU arithmetic from the independent
`guild-inference/guild-kernels-cpu` repository. The pinned source revision is
`264227c74d0151b914d826b0fa35a0f576a117b8` (implementation/package 1.1.1).
`cmake/GuildCPUSources.cmake` in that repository owns both source lists and the
per-translation-unit SIMD flags for bundled and standalone builds.

## Build modes

Default/bundled:

```sh
cmake -S . -B build-cuda12 -DGUILD_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=61 -DGUILD_EXPERIMENTAL_SM60=ON
```

The initial build fetches the immutable CPU revision. An offline build can reuse
that cached checkout; configuration verifies its Git HEAD matches the lock.
Absent or stale offline dependencies fail explicitly. The resulting executable
contains the CPU baseline and needs no external package or network to use it.
`GUILD_CPU_SOURCE_DIR` is an explicit development override for an independent
checkout; it is not the pinned-release mode.

External-only CPU arithmetic:

```sh
cmake -S . -B build-cuda12 -DGUILD_BUILTIN_CPU=OFF
cmake --build build-cuda12 --target guild external_runner_parity_test
python3 tools/check_cpu_ownership.py --external-only \
  --output bench/results/phase2-continuation/cpu-external-only-ownership.json
```

The CPU archive in this mode has exactly `pool.cpp.o`, `pack_layout.cpp.o`,
`features.cpp.o`, `isa.cpp.o` and `external_only.cpp.o`. Feature-probe-only builds
exclude the dispatch-to-arithmetic bodies. Legacy entry points are fail-closed
stubs. No expert, quantizer, SIMD row, router, vector or backend arithmetic
translation unit is linked from that target. Model startup requires the complete
external CPU contract; it cannot silently use one of those stubs as a fallback.

The old `GUILD_BUILTIN_NATIVE_CPU=OFF` option now also selects this stronger
external-only configuration. Existing CPE1 plugins remain usable in bundled
builds, but do not satisfy the complete external-only contract.

## Ownership retained in Core

- NUMA-aware worker scheduling, thread placement, job grouping and phase barriers.
- Expert pack/file interpretation and byte-offset ownership (`pack_layout.cpp`).
- Memory allocation, residency, transfers and lifetime policy.
- Independent input/output finite-value checks and tensor/buffer validation.
- Prefetch prediction policy (which experts to warm), using a bound backend
  callback for the actual BF16 router computation.

Canonical planar-Q2 activation images and arithmetic callbacks are public C
contracts in SDK 1.3. Native format construction, all expert projections,
quantization, SwiGLU, ordered combine and router dot arithmetic are backend-owned.
The earlier duplicate Core arithmetic sources/headers have been removed.
GPU-side and other model-specific extraction remains separate work.

## Verification

- Independent C/C++ contract builds and CPU oracle tests pass. The tests exercise
  canonical planar Q2, native Q2 rows, routing, combine failure, format construction,
  activation conversion and bitwise Q4_K multi-token dispatch at NT=1/2/3/4/8.
- Forced AVX2 tests pass. The portable GGML variant requires AVX2/FMA/F16C/BMI2;
  the explicit AVX-512 variant additionally requires F/BW/DQ/VL/CD. Physical ISA
  checks precede GGML initialization. Core no longer infers BMI2 from AVX2, and
  AVX-512 reporting requires the OS save state and DQ as well as F/BW/VL.
- Bundled builds from the fetched pin pass 89 Core CTests with six existing skips;
  `ple_parity` is separately excluded for its absent canonical-Q2 GGUF fixture.
- Real Ornith prefill/decode passes with all bundled CPU arithmetic disabled,
  using independently built/signed CPU, CUDA-router and Qwen-runner packages.
  Vocabulary logits, eight greedy IDs, recurrent state and tested host-KV prefixes
  match the preserved bundled reference bit-for-bit.
- The same unchanged executable ran bundled inference, external CPU 1.0.1,
  independently built 1.1.1, rollback to 1.0.1, and bundled fallback after disabling
  CPU/CUDA overrides. Bindings and DSO execution counters are checked at each
  external step; fallback checks the actual bundled provider IDs.

A full real canonical-planar-Q2 model artifact is still absent and is not counted
as passing. A Qwen4Exp IQ4_XS model exists locally and requires its own runner/PLE
verification. Windows and the adjusted experimental SYCL consumer are not
validated by these Linux/CUDA tests. Package selection races, durable install
failure and production signing infrastructure remain separate open gates.
