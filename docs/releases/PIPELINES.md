# Build and release pipelines

## Checked configurations

| Workflow | Work performed | Evidence boundary |
| --- | --- | --- |
| `ci.yml` | Python policy/installer tests, installed C99/C++17 SDK on Ubuntu 22.04/24.04, Windows 2022 and macOS 14, Linux management/ABI tests, and ASan+UBSan / TSan `modular_abi_test` sanitizer gates | Windows/macOS SDK checks do not validate a Windows/macOS inference engine; the Linux non-CUDA build cannot perform CPU-only inference |
| `backend-integration.yml` | Build/test exact CPU and runner commits against the current installed SDK, fresh-process signed registration, full pinned CUDA backend compilation and manifest/operation checks | Signing uses a temporary test identity; hosted CUDA compilation has no device execution |
| `dev-build.yml` | Linux CUDA 13 inference executable, portable CPU baseline, linked CUDA runtime/BLAS, licenses, manifest and checksums | Development artifact; no real-GPU or full-model claim |
| `release.yml` | Tag/revision validation, all CI checks, distribution build, dedicated-GPU shipped-binary golden test, reviewer-protected publication and OIDC provenance | Publication waits for real GPU evidence and configured release protection |

The CUDA build container is pinned by its amd64 digest. It builds SM75, SM86,
SM89 and SM120 images, with SM120 PTX. These are compilation targets, not a
claim that every GPU architecture has passed inference. Pascal validation on the
development machine uses CUDA 12/SM61 and remains distinct from this CUDA13
distribution. HIP, SYCL, Windows inference and CPU-only inference are not release
artifacts of these workflows.

Development artifacts are available in each successful Actions run, identified
by the full Core SHA. They do not overwrite a moving development release/tag.

## Distribution layout

`tools/package_linux_cuda.py --build build --output build-dist/guild-linux-x86_64-cuda13`
stages:

- `guild`: launcher that resolves bundled libraries relative to its own location;
- `guild.bin`: the actual CUDA executable;
- `lib/`: the exact linked libcudart, libcublas and libcublasLt SONAMEs;
- `licenses/`: Guild/Strata, independent CPU and CUDA backends, GGML and NVIDIA CUDA license files;
- `guild-release.json`: structured build identity, dependency hashes and Linux
  system-library prerequisites.

The adjacent `.tar.gz` and `.sha256` preserve executable permissions. Packaging
checks the launcher, unchanged executable provenance and actual bundled-library
resolution with a fresh empty package store. It requires `GUILD_PORTABLE=ON` for
public artifacts. `--allow-native-cpu` is limited to explicitly host-specific
local development artifacts. The NVIDIA driver and listed Linux system
libraries are prerequisites. Model GGUFs, prepared packs and tokenizers are not
included. Install/import a matching model manifest with those paths before use.

The CUDA baseline must include all three extracted groups (dense, attention and
MoE), and its actual source revision must match the embedded lock. If an existing
CMake cache retains a previous CUDA pin, reconfigure with
`-UGUILD_CUDA_BACKEND_REVISION` before building and packaging.

## Release setup and real-GPU gate

Before a release, configure the GitHub environment **`release`** with required
reviewers. The validation job checks this configuration; it does not create an
unprotected environment as a substitute. The release tag must identify a clean
commit reachable from `main`, match `VERSION.json`, satisfy maturity/revision
gates, and have no existing GitHub release.

Provide a dedicated runner with labels `self-hosted`, `linux`, `x64`,
`guild-cuda-validation`, a compatible NVIDIA driver/GPU, sufficient RAM and
prepared model fixtures. This runner is used only by the validated tag workflow,
not by pull-request jobs. Set the release environment variable
`GUILD_RELEASE_GPU_FIXTURE` to a local golden-fixture JSON path:

```json
{
  "model": "ornith:35b",
  "model_manifest": "/srv/guild-fixtures/ornith-manifest.json",
  "prefill_batch": 256,
  "request": {
    "model": "ornith:35b",
    "messages": [{"role": "user", "content": "Count from one to ten."}],
    "max_tokens": 16
  },
  "expected_text": "EXACT COMPLETION FROM THE VALIDATED REFERENCE"
}
```

The example is a schema illustration, not an actual golden. Maintain the real
fixture from a separately validated reference with exact GGUF/pack/tokenizer
versions. The manifest is copied into an isolated store and supplies absolute
GGUF, prepared-pack and tokenizer paths. The test starts the **downloaded shipped executable**, bound to
127.0.0.1 with an isolated package/model store, forces greedy non-streamed
generation, requires eight completed decode steps, and compares exact output.
It uploads response, server log, GPU identity and binary/fixture hashes. This
smoke gate supplements the numerical model tests; it does not replace full
logit/state/KV parity or testing on every supported GPU.

The publication job waits for this evidence and environment review. It verifies
archive hashes, creates GitHub OIDC build-provenance attestations, and publishes
new assets. Verify downloads using their checksums and `gh attestation verify
ARCHIVE --repo guild-inference/Guild`. This authenticates Core release provenance;
production Ed25519 signing of external `.gkernel` packages is a separate,
unfinished release pipeline. No production package private key is in these
workflows.
