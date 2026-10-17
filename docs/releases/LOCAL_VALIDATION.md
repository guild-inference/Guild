# Release infrastructure validation — 2026-10-17

Measured locally on Linux x86-64 (Ubuntu 24.04), two Xeon Gold 6130 CPUs,
377 GiB RAM and a GTX 1070 8 GiB (SM61), driver 580.178.04, CUDA 12.0.140,
GCC 13.3 for C/C++ and GCC12 as nvcc's host compiler. The SDK is 1.3.0,
Kernel ABI 1.0.0, pinned CPU source
`264227c74d0151b914d826b0fa35a0f576a117b8`.

| Check | Result |
| --- | --- |
| Version/revision, release gate, updater and deterministic archive Python tests | 11 passed |
| Existing no-download installer tests | 270 passed, one existing skip |
| Installed SDK C99 plugin / C++17 dynamic-loader host, Release | One passed |
| Fresh portable Linux management/ABI build, CUDA off | 24 passed, four pack-fixture skips |
| Independent CPU implementation, current installed SDK | Two passed |
| Independent Qwen runner, current installed SDK | One passed |
| Ephemeral signed CPU/runner install and fresh-process registration | Passed; Core executable SHA unchanged |
| CUDA12/SM61 Release suite, serial | 89 passed, six skips; canonical-Q2 `ple_parity` excluded for its absent GGUF fixture |
| Human/JSON version identity and corrupt-trust-store failure | Passed for management and CUDA builds |
| Local CUDA distribution dependency/launcher checks | Passed |
| Packaged executable serving actual Ornith, greedy generation | 16 completion tokens matched the prior local build |
| Workflow static validation | actionlint 1.7.7 passed |

The first full CUDA test command hit its outer 300-second tool timeout during
Ornith prefill. The interrupted test and remaining tests were then completed
serially with a larger timeout; Ornith prefill passed in 122.29 seconds. Two
fused CUDA tests skipped on Pascal; four CPU pack tests skipped for missing
fixtures. No numerical threshold was changed.

The local archive
`build-dist-validation/guild-linux-cuda-verified.tar.gz` has SHA-256
`a475d15d6d450a089c4c686efffc1c2a3685869edd472249042449d1f979cfa1`.
It is an explicitly host-specific **development** artifact built with tracked
changes over Core `5a10a82b7534ad19f6933e3997883f09d1607936`, using
`--allow-native-cpu`. Its machine-readable manifest records that provenance.
The archive includes the linked CUDA12 libraries; its SM61 validation is
separate from the CUDA13/SM75+ hosted distribution workflow.

For the real-model launcher test, a temporary fixture was captured from the
earlier locally packaged binary with Ornith Q4_K_M, context32768, prefill
batch256, a 16-token prompt and 16 greedy completion tokens. The new package
matched that completion through its HTTP API. This exercises the exact packaged
binary, launcher, model-manifest staging and GPU inference. Existing numerical
kernel/model tests supply the separate parity coverage. The temporary fixture,
server logs, responses and hashes remain under `build-workflow-tools/`.

At validation time GitHub reported no environments and no self-hosted runners
for this repository. Publication therefore requires the documented release
reviewers, GPU runner and maintained reference fixture to be configured.
Hosted Windows SDK, Ubuntu22.04 and CUDA13 checks are workflow results to inspect
after pushing; the local results above do not stand in for them.

The first hosted run confirmed the installed SDK on all three OS targets and
the pinned independent CPU/runner contracts, signed registration and CUDA13
router compilation. It exposed a missing-tokenizer fixture registration and a
container-mounted Git ownership error before distribution compilation. The
follow-up keeps unloaded-tokenizer checks always active, registers full golden
parity only where its explicit tokenizer paths exist, and scopes the container
Git ownership exception through per-command environment settings. It does not
alter the local Git configuration or change tokenization comparisons.
