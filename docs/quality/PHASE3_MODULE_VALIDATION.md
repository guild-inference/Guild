# Phase 3 module validation

Validation host: Linux x86-64, dual Intel Xeon Gold 6130 (32 physical cores,
64 threads), 377 GiB RAM, GTX 1070 8 GiB (SM61), CUDA 12.0.140, driver 580.178.04.
Core baseline: `88c81179674f6fb55fbbfe006686059f9cba4c17` (Phase 2E).

## Automated coverage

`gkmodule_test` uses real C99 shared libraries, signed manifests, a GKPK archive
with `.gkmodule` suffix, isolated registries and a private test store. It covers:

- Two modules with multiple operations, descriptor copying from plugin stacks,
  inclusive/exclusive sequence-scan outputs through the C service.
- Deterministic priority/tie resolution, explicit fallback and strict ID locking.
- Version, capability, dtype, shape, backend, numerical, sync and memory rejection.
- ABI/descriptor rejection, duplicate registration and partial-init cleanup.
- Trust rejection, pinned dependency versions/publishers/digests, missing or
  disabled dependencies, module cycles and replacement incompatibility.
- Bindings preventing unload; eight execution threads racing disable/reload.
- Registry-write failures for install, enable, disable and idempotent re-enable;
  previous-version rollback and four install fault boundaries.
- Process termination at publication/registry-write boundaries and startup
  recovery of the previously committed module version.

`gkmodule_cuda_test` executes a signed CUDA module on a non-default stream,
captures and launches a CUDA graph, verifies exact outputs, rejects busy unload,
and checks that releasing the binding waits for a recorded completion event.

The existing `modular_abi_test`, `kv_host_only_test` and
`ornith_gkernel_plugin_test` retain the Phase 2C–2E checks, including signed
legacy backends, model/session lifetime, concurrent sessions, package rollback,
archive rejection and CUDA-resource teardown.

## Regression results

- Host CTest: **26 passed, 2 skipped**, no failures (28 registered tests).
- CUDA CTest: **92 passed, 4 skipped, 1 fixture failure** (97 registered tests).
  `ple_parity` requires the historical
  `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf`, matching dense pack and
  `bench/micro/ple_{in,out}.bin` captures. These are absent on this host. The
  same failure occurred during Phase 2E verification; no test was converted to
  a pass or disabled. The available model is the distinct Unsloth IQ4_XS
  artifact and cannot substitute for that exact numerical oracle.
- Skips: CPU expert-pack fixtures (`expert_multi_test`, `pool_stress`), plus
  `prefill_fused_moe_test` and `prefill_fused_iq_test` on this SM61 device.
- Installed SDK C99/C++17 ABI contract: passed. The scan module also built using
  only the installed SDK from `sdk/`, not Core headers or libraries.
- Python: 13 policy/release/updater/packaging tests passed; 270 installer tests
  passed and one skipped; CLI provenance/corrupt-store checks passed.
- Signed existing CPU 1.1.1, CUDA 1.4.0 and Qwen runner 1.1.0 packages passed
  fresh-process registration. Combined external execution and bundled fallback
  matched first-token logits and eight greedy Ornith tokens exactly.

Sanitizer builds use `RelWithDebInfo`, portable CPU settings and
`-fno-omit-frame-pointer`. `modular_abi_test` and `gkmodule_test` pass under
ASan+UBSan and TSan (`setarch x86_64 -R`); ASan uses `detect_leaks=0`, so this
does not claim leak-sanitizer coverage. Build trees and complete local logs are
under `build-cuda12/phase3-{asan,tsan}`. CI runs both test executables for each
sanitizer configuration.

## Scope of the evidence

This validates reusable operation infrastructure and existing Ornith inference,
not new complete model architectures or audio pipelines. Windows dynamic loading
is still unavailable. HIP/macOS module runtime and physical multi-GPU execution
were not validated on this single-NVIDIA-card Linux host. Installed SDK checks
on other platforms are configured in CI, not represented as local test results.

Performance, binary identities, parity hashes and raw measurements are recorded
in [the Phase 3 performance report](../performance/PHASE3_MODULES.md).
