# `.gkernel` & `.gkmetadata` Declarative Package Format

Headers, Schema & Implementation:
- [`include/guild/abi/gkernel_package.hpp`](../../include/guild/abi/gkernel_package.hpp)
- [`src/abi/gkernel_package.cpp`](../../src/abi/gkernel_package.cpp)
- [`include/guild/abi/ed25519.hpp`](../../include/guild/abi/ed25519.hpp) / [`src/abi/ed25519.cpp`](../../src/abi/ed25519.cpp)
- [`schemas/gkmetadata.v1.schema.json`](../../schemas/gkmetadata.v1.schema.json)

A `.gkernel` package is a cryptographically signed, content-addressed bundle of hardware kernels and/or model architecture runners described by a strictly declarative `.gkmetadata` JSON manifest.

---

## 1. Three-Tier Package Boundary (`gkernel` vs `gkmodule`)

Guild enforces a strict three-tier architectural boundary in `.gkmetadata`:

1. **`package_kind: "gkernel"` (Executable Plugin Package)**
   - Allowed `scope`: `"cpu-kernels"`, `"cuda-kernels"`, `"hip-kernels"`, `"runner"`, or `"hybrid"`.
   - Exports `guild_plugin_entry_v1` from its declared shared library and registers `guild_kernel_op_desc_v1_t` operations and/or `guild_runner_desc_v1_t` model runners with Guild Core.
   - May declare `module_dependencies` recording reusable `.gkmodule` building blocks linked or consumed at build time.
2. **`package_kind: "gkmodule"` (Non-Executable Building Block)**
   - Allowed `scope`: `"module"`.
   - Must **not** declare `provided_ops`, `provided_runners`, or `library_path`.
   - **Direct Execution Rejection**: Guild Core (`validate_gkernel_package` and `GKernelPackageManager`) rejects any attempt to directly install, validate, or `dlopen` a `.gkmodule` package as a runtime plugin with `GUILD_STATUS_ERR_SECURITY_VIOLATION`.

---

## 2. Security Model, Ed25519 Signing & Hardened Staging

1. **Declarative Metadata Only — Zero Arbitrary Install Scripts**
   - `.gkmetadata` manifests are parsed by a deterministic, dependency-free parser (`GKernelMetadata::parse_json`) that rejects unknown top-level keys and any executable hook keys (`install_script`, `post_install`, `pre_install`, `build_script`, `shell_command`, `exec_hook`, `python_hook`) with `GUILD_STATUS_ERR_SECURITY_VIOLATION`.
2. **RFC 8032 Ed25519 Cryptographic Signatures & Trust Store**
   - `GKernelMetadata::compute_payload_sha256()` computes the SHA-256 digest of the canonical unsigned JSON representation of `.gkmetadata` (excluding the `signature` block).
   - `GKernelMetadata::sign_with_seed_hex()` signs the domain-separated message `"GUILD-GKERNEL-SIG-V1:" + payload_sha256` using Guild's self-contained FIPS 180-4 SHA-512 + RFC 8032 pure Ed25519 implementation (`src/abi/ed25519.cpp`), which enforces canonical scalar bounds (`S < L`) and constant-time buffer comparisons.
   - `validate_gkernel_package()` verifies:
     - `signature.algorithm == "ed25519"`
     - `signature.payload_sha256 == meta.compute_payload_sha256()` (constant-time check)
     - `(publisher_id, key_id)` is present in `PackageSecurityPolicy::trusted_publishers` and mapped to a 32-byte Ed25519 public key in `trusted_public_keys`
     - Neither `key_id` (`revoked_key_ids`) nor `package_id@version` (`revoked_releases`) has been revoked
     - `crypto::ed25519_verify_hex(signing_message, signature_hex, pubkey_hex)` succeeds
3. **Hardened Archive & Directory Staging (`stage_gkernel_source`)**
   - Supports directory sources, `GKPK` v1 binary bundles (`create_gkernel_bundle`), and POSIX USTAR `.tar` archives.
   - Enforces strict pre-`dlopen` safety invariants:
     - Rejects path traversal (`..`), absolute paths (`/`, `\`), drive letters (`C:`), and NUL/control characters.
     - Rejects symbolic links (`fs::is_symlink` on `symlink_status`) anywhere in the package directory or archive.
     - Enforces archive bomb limits (`max_entries = 64`, `max_total_bytes = 256 MiB`).
     - Rejects duplicate entry paths and any shared library (`.so`, `.dll`, `.dylib`) on disk that is not declared in `artifacts[]`.
     - Verifies every declared artifact's exact `byte_size` and 64-character hex `sha256` digest against the staged file on disk immediately before `dlopen`.

---

## 3. Canonical `.gkmetadata` Schema (v1)

Validated against [`schemas/gkmetadata.v1.schema.json`](../../schemas/gkmetadata.v1.schema.json):

```json
{
  "format_version": 1,
  "package_kind": "gkernel",
  "package_id": "ext.guild-kernels-ornith-prefill",
  "scope": "hybrid",
  "version": "1.0.0",
  "publisher_id": "org.illatio.guild",
  "entrypoint_symbol": "guild_plugin_entry_v1",
  "library_path": "lib/libguild_ext_ornith_prefill.so",
  "kernel_abi": {
    "major": 1,
    "minor_min": 0,
    "minor_max": 0
  },
  "target_platform": {
    "os": "linux",
    "arch": "x86_64"
  },
  "required_core_caps": 0,
  "required_cpu_isa": 0,
  "required_gpu_caps": 0,
  "min_sm_version": 60,
  "max_sm_version": 0,
  "backends": ["cpu", "cuda"],
  "supported_architectures": ["qwen35moe", "qwen4exp"],
  "provided_ops": [
    "ext.cpu.finite_check.v1",
    "ext.cuda.moe.router_topk_bf16"
  ],
  "provided_runners": [],
  "required_ops": [],
  "dependencies": [
    {
      "package_id": "builtin.guild-kernels-cuda",
      "min_version": "1.0.0",
      "max_version": "1.99.99",
      "publisher_id": "org.illatio.guild"
    }
  ],
  "module_dependencies": [
    {
      "module_id": "mod.guild.math-tile-primitives",
      "min_version": "1.0.0",
      "abi_tag": "cxx17-sm61",
      "publisher_id": "org.illatio.guild",
      "sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    }
  ],
  "artifacts": [
    {
      "path": "lib/libguild_ext_ornith_prefill.so",
      "sha256": "<64-char-hex-sha256>",
      "byte_size": 70616,
      "role": "shared_library"
    }
  ],
  "provenance": {
    "spdx_license": "MIT",
    "source_repo": "https://github.com/guild-inference/Guild",
    "source_revision": "9d77159000000000000000000000000000000000"
  },
  "signature": {
    "algorithm": "ed25519",
    "key_id": "illatio-release-2026-01",
    "signature_hex": "<128-char-hex-ed25519-signature>",
    "payload_sha256": "<64-char-hex-canonical-payload-sha256>"
  }
}
```

---

## 4. `guild kernel` CLI Package Management & Signing Workflow

Guild's CLI (`src/cli/guild_main.cpp`) provides full `.gkernel` lifecycle and trust-store management:

```bash
# 1. Generate a new Ed25519 signing keypair (64-hex private seed + 64-hex public key)
guild kernel keygen [--seed <64-hex>] [--json]

# 2. Register a trusted publisher public key in the local trust store (<data_dir>/kernels/trust_store.json)
guild kernel trust-key --publisher <pub_id> --key-id <id> --pubkey <64-hex> [--data-dir PATH]

# 3. Recompute artifact SHA-256 digests, sign .gkmetadata, and optionally emit a .gkernel bundle
guild kernel sign <pkg_dir> --key-id <id> --seed <64-hex> [--bundle <out.gkernel>]

# 4. Stage, verify Ed25519 signature + SHA-256 digests, and install a .gkernel bundle or directory
guild kernel install <path> [--no-enable] [--allow-unsigned-dev] [--data-dir PATH] [--json]

# 5. List installed packages, inspect metadata, or view active runners and hardware kernels
guild kernel list [--data-dir PATH] [--json]
guild kernel inspect <package_id|path> [--data-dir PATH] [--json]
guild kernel active [--data-dir PATH] [--json]

# 6. Enable, disable, rollback to previous version, or remove an installed package
guild kernel enable <package_id> [--data-dir PATH]
guild kernel disable <package_id> [--data-dir PATH]
guild kernel rollback <package_id> [--data-dir PATH]
guild kernel remove <package_id> [--data-dir PATH]

# 7. Revoke a compromised signing key or specific package release
guild kernel revoke-key <key_id> [--data-dir PATH]
guild kernel revoke-release <package_id@version> [--data-dir PATH]
```

---

## 5. GEXEC Frozen Kernel Locks (`GexecKernelLock`)

When Guild compiles or caches an execution plan (`GexecKernelLock`), it records the `package_id`, `package_version`, `impl_id`, `abi_version`, and `artifact_sha256`:
- `verify_gexec_kernel_lock(lock, meta, op_desc, err)` verifies that the active package and kernel descriptor match the locked version, ABI, and SHA-256 artifact digest.
- `KernelRegistry::select_kernel()` honors `Query::preferred_impl_id` and `Query::require_preferred_impl`: if `require_preferred_impl` is set and the locked kernel is unavailable or disabled, selection fails deterministically with `GUILD_STATUS_ERR_MISSING_KERNEL`.
