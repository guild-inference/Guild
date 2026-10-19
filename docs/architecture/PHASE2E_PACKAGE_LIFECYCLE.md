# Phase 2E package lifecycle and durability contract

This document specifies how Guild stages, verifies, publishes, loads, replaces,
rolls back, and removes `.gkernel` packages in `GKernelPackageManager`
(`include/guild/abi/gkernel_package.hpp`, `src/abi/gkernel_package.cpp`).

## Store layout

```text
<data_dir>/kernels/
  trust_store.json          # Ed25519 trusted publisher keys, revocations, replacement policy (0600)
  registry.json             # Committed package state and version_dirs map (0600)
  staging/                  # Isolated per-install temporary directories: stage_<pid>_<seq>_<mgr> (0700)
  packages/
    <package_id>/
      <version>/            # Immutable verified package contents (.gkmetadata, lib/*.so, fixtures)
```

## State transitions

1. **Stage (`stage_gkernel_source`)**:
   - Extracts a `.gkernel` (`GKPK` v1 or POSIX `.tar`) archive or copies a source directory into a unique directory under `staging/`.
   - Rejects symlinks, non-regular files, path traversal (`..`, leading `/`, `\`, `:`), duplicate entries, archives exceeding `max_entries` (64) or `max_total_bytes` (256 MiB), truncated `.tar` blocks/padding, and trailing `GKPK` bytes.
   - Parses `.gkmetadata` without executing plugin code and rejects any extracted file not listed in `artifacts` or `validation_fixtures`.
   - Flushes each staged file with `fsync` (`0500` for shared libraries, `0400` for metadata/fixtures) and `fsync`s parent directories.
2. **Validate (`validate_gkernel_package` + `validate_package_dependency_graph`)**:
   - Enforces `package_kind == "gkernel"` (non-executable `.gkmodule` archives are rejected for direct loading).
   - Checks release, publisher, and key revocations against `PackageSecurityPolicy`.
   - Verifies the canonical `.gkmetadata` SHA-256 payload digest and Ed25519 signature (`GUILD-GKERNEL-SIG-V1:<payload_sha256>`).
   - Verifies Kernel ABI v1 compatibility, target OS/architecture, host CPU ISA, GPU SM bounds, every declared artifact's byte size and SHA-256 digest, every on-disk validation fixture's SHA-256 digest, and the topological dependency graph across enabled packages.
3. **Same-version immutability check**:
   - Once `package_id@version` exists in `registry.json` (`active_version` or `version_dirs`) or on disk under `packages/<package_id>/<version>`, its signed content is immutable unless `allow_same_version_replacement` is explicitly enabled in `trust_store.json` (default `false`).
   - Installing `package_id@version` with **identical** verified `.gkmetadata` payload SHA-256 and on-disk artifacts is idempotent (or re-enables and loads the package if it was disabled).
   - Installing `package_id@version` with **different** content is rejected with `GUILD_STATUS_ERR_DUPLICATE_REGISTRATION`.
4. **Publish and atomic registry swap**:
   - If an existing version of `package_id` is loaded and holds active leases (`active_package_leases(package_id) > 0` in either `KernelRegistry` or `ModelRunnerRegistry`), installation fails with `GUILD_STATUS_ERR_ACTIVE_SESSION_BUSY` before modifying `packages/` or unloading the active version.
   - The verified staging directory is renamed into `packages/<package_id>/<version>` and its parent directory is `fsync`ed before the previous version is unloaded.
   - Under `abi::provider_lifecycle_mutex()`, the previous version (if loaded) is unloaded and the new version is loaded via `register_plugin_transactional`. If loading fails, the new directory is removed and the previous version is immediately reloaded within the same lifecycle transaction.
5. **Commit (`save_registry_unlocked`)**:
   - Writes `registry.json.tmp`, flushes with `fsync`, renames atomically to `registry.json`, and `fsync`s the store root directory.
   - If committing `registry.json` fails, in-memory registry state and on-disk directories are rolled back to the previous version.
6. **Rollback (`rollback_package`)**:
   - Verifies `previous_version` exists in `version_dirs` and its `.gkmetadata` is valid.
   - Under `abi::provider_lifecycle_mutex()`, unloads `active_version` (if no leases are held) and loads `previous_version`. If loading `previous_version` or committing `registry.json` fails, `active_version` is restored.
7. **Remove (`remove_package`)**:
   - Verifies no other installed package depends on `package_id` and `active_package_leases(package_id) == 0`.
   - Unloads `package_id` from `KernelRegistry` and `ModelRunnerRegistry`, commits the updated `registry.json` **first**, and only then removes `packages/<package_id>` from disk.
8. **Startup recovery (`init`)**:
   - Removes any leftover directories in `staging/`, removes stale `registry.json.tmp` / `trust_store.json.tmp` files, loads `trust_store.json` and `registry.json` (failing closed if either file or any registered package's `.gkmetadata` is corrupted), and sweeps unreferenced directories under `packages/`.
