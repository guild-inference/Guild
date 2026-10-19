# Phase 2D reconciliation before Phase 2E

On 2026-10-19 the local implementation was uncommitted. Remote main still matched
Core `2953e4f431c1920ab0afaa128b4dbce1e380f641`, CUDA
`cab46da5380bf83683d54a7f8d61c43f5de83f49`, and runner
`0ccff556724fc7fc8d66795ce157485e23dfbdaa`.

Published dependency implementations:

- CPU 1.1.1: `264227c74d0151b914d826b0fa35a0f576a117b8` (unchanged).
- CUDA 1.4.0: `fe4c0808dab38e72d31b9b6bfcc0943ee6e3c79a`.
- Runner 1.1.0: `650b09cef3a1dec184a96605111fa6d44622ebe6`.
- SDK 1.4.0; Kernel ABI 1.0.0.

Core now pins these sources. Removed implicit sibling-source fallbacks that could
mix locked CUDA arithmetic with unpinned prefill and report misleading provenance.
Explicit development source overrides remain available.

## Frozen local binary evidence

Preserved before rebuilding under `build-cuda12/phase2e-frozen-phase2d/reference/`:

| Binary | SHA-256 |
|---|---|
| guild | `d688d8afa5388477c88b73d6fc86ed72685fda8fb54d45f8e069a635dce76b08` |
| ornith_prefill_bench | `b42eb775193e57dbed5dde4384aad004d037d82fe7625afd64cd91ded944e68f` |
| external_runner_parity_test | `ed225ba9659884aaaacef75e3e5e79ef2e3bc1694393dcde63b2170ae84d24d9` |

The snapshot includes CMake configuration and build metadata. Existing raw results
in `bench/results/phase2d-decode-scaling/` are retained. The source was dirty when
these binaries were built; their embedded Core SHA alone is not their provenance.

## Reverification and qualifications

- Independent Release CUDA: 5 passed on GTX 1070; runner: 1 passed.
- Rebuilt host Release suite: 25 passed, 2 skipped (missing expert fixtures).
- Rebuilt CUDA `kv_host_only_test` and `modular_abi_test`: 2 passed.
- Historical Phase 2D fast suite: 87 passed, 4 skipped, not 91 executed passes.
- Historical real Ornith prefill parity and Qwen reference parity passed; the
  latter did not install an external runner or exercise MTP.
- Prefill arithmetic source ownership moved, but the legacy C++ prefill interface
  is still statically linked from backend sources, not exposed through a new
  dynamic prefill ABI. No claim of complete external-only Qwen prefill is made.
- Multi-GPU, Windows, macOS, crash durability and concurrent replacement were not
  established by Phase 2D. They remain Phase 2E gates.
