# `.gkernel` & `.gkmetadata` Declarative Package Format

Headers & Implementation:
- [`include/guild/abi/gkernel_package.hpp`](../../include/guild/abi/gkernel_package.hpp)
- [`src/abi/gkernel_package.cpp`](../../src/abi/gkernel_package.cpp)

A `.gkernel` package is a versioned bundle of hardware kernels and/or model architecture runners described by a strictly declarative `.gkmetadata` JSON manifest.

---

## 1. Security Model & Non-Negotiable Invariants

1. **Declarative Metadata Only — Zero Arbitrary Install Scripts**
   - `.gkmetadata` manifests are parsed by a deterministic, dependency-free parser (`parse_gkmetadata_json`).
   - Any manifest containing executable hook keys (`install_script`, `post_install`, `pre_install`, `build_script`, `shell_command`, `exec_hook`, `python_hook`) is immediately rejected with `GUILD_STATUS_ERR_UNTRUSTED_PACKAGE`.
2. **Path Traversal Protection**
   - Artifact paths (`binary_artifacts[].relative_path`) must be relative paths inside the package directory. Absolute paths (`/`, `\`), drive letters (`C:`), or parent directory segments (`..`) are rejected with `GUILD_STATUS_ERR_UNTRUSTED_PACKAGE`.
3. **Cryptographic Integrity & Publisher Trust**
   - Every binary artifact must declare a 64-character hexadecimal `sha256` digest verified against the on-disk binary (`compute_file_sha256_hex`) before loading.
   - Unless `SecurityPolicy::allow_unsigned_local_dev` is explicitly enabled for local development, packages must include a cryptographic signature (`signature_ed25519` + `signing_key_id`) from a trusted publisher in `SecurityPolicy::trusted_publishers`.
   - Revoked signing keys (`revoked_key_ids`) and revoked publishers (`revoked_publishers`) are rejected even if otherwise listed as trusted.
4. **ABI & Hardware Capability Gating**
   - `abi_version_min` and `abi_version_max` are checked against `GUILD_KERNEL_ABI_VERSION_V1` via `guild_abi_is_compatible`.
   - Each exported operation and runner is validated against the host's `guild_device_capabilities_v1_t` (CPU ISA flags, GPU compute capability range, scratch budget, and required dtypes).
5. **Active-Session Isolation**
   - Active `GuildSession` instances hold `SessionRunnerLease` and `SessionKernelLease` locks. Attempting to unload or unregister a package while any session is active fails with `GUILD_STATUS_ERR_ACTIVE_SESSION_BUSY`.

---

## 2. `.gkmetadata` Schema (v1)

```json
{
  "format_version": 1,
  "package_name": "guild-kernels-pascal",
  "package_version": "1.0.0",
  "publisher": "illatio",
  "abi_version_min": 65536,
  "abi_version_max": 65536,
  "signing_key_id": "key-illatio-prod-01",
  "signature_ed25519": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  "supported_architectures": ["qwen4exp", "qwen35moe"],
  "dependencies": [
    {
      "package_name": "builtin.guild-kernels-cpu",
      "min_version": "1.0.0"
    }
  ],
  "binary_artifacts": [
    {
      "relative_path": "lib/libguild_pascal.so",
      "os": "linux",
      "cpu_arch": "x86_64",
      "gpu_backend": "cuda",
      "min_sm": 61,
      "max_sm": 61,
      "sha256": "<64-char-hex-sha256>",
      "size_bytes": 1048576
    }
  ],
  "exported_ops": [
    {
      "impl_id": "cuda.gemm_bf16.sm61.v1",
      "impl_version": 100,
      "op_kind": "gemm_bf16",
      "device_kind": "cuda",
      "required_cpu_isa": 0,
      "min_gpu_arch": 61,
      "max_gpu_arch": 90,
      "priority": 100,
      "numerical_flags": 1
    }
  ],
  "exported_runners": [
    {
      "runner_id": "runner.qwen35moe.v1",
      "architecture_name": "qwen35moe",
      "runner_version": 100
    }
  ]
}
```

---

## 3. GEXEC Frozen Kernel Locks (`GexecKernelLock`)

When Guild compiles or caches an execution plan (GEXEC), it can record the exact `impl_id`, `impl_version`, and `package_sha256` selected for each `guild_op_kind_t` in a `GexecKernelLock`.

At execution time:
- `verify_gexec_kernel_lock(lock, active_selection)` verifies that the bound kernel matches the locked `op_kind`, `impl_id`, and `impl_version`.
- `KernelRegistry::select(criteria)` honors `criteria.preferred_impl_id` and `criteria.preferred_impl_version`: if `require_exact_impl` is set and the locked kernel is unavailable or incompatible with current hardware, selection fails cleanly with `GUILD_STATUS_ERR_NOT_FOUND` rather than silently substituting a different kernel.
