# Independent backends: verified migration slices

**Phase 2 is in progress.** These are functioning independent repositories and
packages, but complete CPU/CUDA/architecture decoupling is not finished.

Continuation: [CPU source ownership](CPU_BACKEND_OWNERSHIP.md) now uses pinned
independent sources for bundled builds and proves an external-only CPU arithmetic
configuration. The earlier preview scope and repository table below describe the
original migration slice, not the complete current CPU operation set.

[CUDA source ownership](CUDA_BACKEND_OWNERSHIP.md) now covers CDD1 dense,
CDA1 attention/recurrent and CDM1 MoE/device primitives in CUDA package 1.4.0,
using SDK 1.4.0. All three extracted groups passed external-only Ornith inference.
[Phase 2E package lifecycle](PHASE2E_PACKAGE_LIFECYCLE.md) and
[Phase 2E concurrency](PHASE2E_CONCURRENCY.md) establish reference-counted
provider bindings (`KernelBinding` and `std::shared_ptr<const IModelRunner>`),
cross-registry lifecycle transactions, durable `fsync` installation with
same-version immutability, and resource-aware KV staging. Current immutable
revisions are in `backends.lock.json`; the table below is the historical preview
snapshot.

| Repository | Original preview HEAD | Original verified scope |
|---|---|---|
| https://github.com/guild-inference/guild-kernels-cpu | `5c2a76f` | Native expert activation/hidden quantization, gate/up and down arithmetic; own pinned GGML build |
| https://github.com/guild-inference/guild-kernels-cuda | `aa1c565` | SM 6.1 BF16 router projection |
| https://github.com/guild-inference/guild-runners-qwen | `0ccff55` | Qwen-family model-description callbacks; real Ornith inference verified |

SDK/Core support starts at Guild `4b4ded0`. SDK package version is 1.1.0;
Kernel ABI remains v1.0.0. All three repositories have independent CMake projects,
tests, MIT attribution, manifest templates, signing instructions and CI build
workflows. Their build inputs are installed SDK files and declared dependencies.
They do not link private Core libraries or use its source headers.

Hosted CI passed independently for
[CPU](https://github.com/guild-inference/guild-kernels-cpu/actions/runs/38939706726),
[runner](https://github.com/guild-inference/guild-runners-qwen/actions/runs/38939705475),
and [CUDA compilation](https://github.com/guild-inference/guild-kernels-cuda/actions/runs/38939705014).
The actual CUDA execution test was local on the GTX 1070, not hosted CPU-only CI.

## Published preview artifacts

The [SDK 1.1 archive](https://github.com/guild-inference/Guild/releases/tag/kernel-sdk-v1.1.0-preview.1)
can be installed without building Core. Each external repository has an
`integration-v1.0.0-preview.1` prerelease with its signed integration-proof
packages and public verification receipt. CPU includes both 1.0.0 and 1.0.1.
These are explicitly test-signed previews, not production signing identities.
Full source SHAs, package/version names and bundle hashes are recorded in
`bench/results/external-kernels/package-verification.json`.

## Developer workflow

Install the SDK, then clone and build each repository separately using its README:

```sh
cmake -S sdk -B build-sdk -DCMAKE_INSTALL_PREFIX=/opt/guild-sdk
cmake --install build-sdk
# In each external checkout:
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/guild-sdk
cmake --build build -j
ctest --test-dir build --output-on-failure
cmake --install build --prefix "$PWD/build/package"
```

CUDA additionally needs its toolkit/compiler and `CMAKE_CUDA_ARCHITECTURES=61`.
The CUDA manifest currently restricts support to SM 6.1; compiling another target
requires matching metadata and hardware validation. CPU has an AVX2/FMA/F16C
baseline and an explicit `GUILD_CPU_GGML_AVX512` variant. The measurements used the
AVX-512 variant on dual Xeon Gold 6130, with no VNNI requirement.

Use `guild kernel sign PACKAGE_DIR --key-id KEY --seed-file PRIVATE_FILE
--bundle OUTPUT.gkernel`, explicitly trust the corresponding public key, then
install with `guild kernel install`. Never distribute the seed. Default Core
trust no longer embeds the former published development key. Existing trust
stores must remove/revoke that development identity if they inherited it.
CI build artifacts are unsigned staging directories, not production releases;
signing is a separate explicit release step using the maintainer's private key.

Package identities:

- `org.guild.kernels.cpu@1.0.0` / `@1.0.1`, operation `external.cpu.native_expert.v1`.
- `org.guild.kernels.cuda@1.0.0`, operation `external.cuda.router.bf16.v1`.
- `org.guild.runners.qwen@1.0.0`, `runner.external.*.v1` providers.

`guild kernel active --json` shows provider IDs. The real-model test additionally
checks bound CPU/CUDA package IDs and DSO counters: installation alone is not
accepted as execution proof. Enabled package loading errors now fail model load
instead of being discarded in favor of built-in execution.

## Integration and independent updates

```sh
python3 tools/test_external_runner.py \
  --package /checkouts/guild-runners-qwen/build/package \
  --additional-package /checkouts/guild-kernels-cpu/build/package \
  --additional-package /checkouts/guild-kernels-cuda/build/package \
  --upgrade-package /checkouts/guild-kernels-cpu/build-upgrade/package \
  --artifact-output build-cuda12/external-dist
```

The driver creates a private temporary test key, explicitly trusts its public key,
signs and installs packages, and compares fresh-process inference against the
built-in reference. It then installs independently built CPU 1.0.1 and rolls back
to 1.0.0. The test checks descriptor version and actual callback execution at
each step and asserts that the Core executable's SHA-256 did not change. Test
bundles and their public verification identity can be retained with
`--artifact-output`; the private seed is deleted. These test identities are not
production release keys.

`GUILD_BUILTIN_QWEN_RUNNERS=OFF` removes built-in runner registration.
`GUILD_BUILTIN_NATIVE_CPU=OFF` replaces built-in native arithmetic with explicit
errors; a missing external native provider then fails model loading. Real Ornith
inference passed with both options off. Neither option disables the rest of Core's
CUDA backend or canonical planar Q2 implementation.

## Remaining requirements before Phase 2 completion

1. Finish the remaining legacy CUDA GEMM/MMQ and Qwen4Exp prefill extraction;
   extend external-only validation beyond the tested Ornith path.
2. Maintain the completed CPU source-ownership gates documented in
   `CPU_BACKEND_OWNERSHIP.md` while extending architecture coverage.
3. Move remaining Qwen topology/tensor-validation/state policy out of Core;
   validate Qwen4Exp/PLE/MTP with its actual model and the external runner.
4. Selection/lease and cross-registry transaction races, same-version
   immutability, durable `fsync` staging/registry commit, and 6-boundary fault
   injection + subprocess crash recovery are verified in
   [Phase 2E package lifecycle](PHASE2E_PACKAGE_LIFECYCLE.md),
   [Phase 2E concurrency](PHASE2E_CONCURRENCY.md), and
   [Phase 2E security and concurrency audit](../quality/PHASE2E_SECURITY_AND_CONCURRENCY_AUDIT.md).
5. Finish independent signed production release key provisioning outside CI.
6. Paired single-session and multi-session (`1, 2, 4, 8` sessions) performance
   evidence is recorded in
   [Phase 2E multi-session validation](../performance/PHASE2E_MULTI_SESSION_VALIDATION.md).

## Later phases (not started here)

Phase 3: reusable `.gkmodule` and measured model-specialized fusion. Guild
dispatches `.gkernel`, never `.gkmodule` operations. Phase 4: GPACK/GEXEC and
reproducible frozen selections. Phase 5: hardware-aware and multi-GPU optimization.
Phase 6: Qwen/long-context decode (16K milestones: 22+, 25+, stretch 30 tok/s).
Phase 7: externally developed GLM runner. No empty future repositories were made.
