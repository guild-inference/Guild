# Phase 2E package durability, signing, and rollback verification

This document records the durability, immutability, and fault-injection guarantees
for `.gkernel` package management in Guild Core.

## Durability and immutability rules

1. **Durable staging and atomic metadata persistence**:
   - Every file extracted into `<data_dir>/kernels/staging/` is written via `write_file_durable` (`open` with `O_CREAT | O_TRUNC | O_NOFOLLOW`, full `write` loop, `fsync`, `fchmod`, `close`) followed by `sync_directory_durable` (`fsync` on the parent directory).
   - `registry.json` and `trust_store.json` are updated via `atomic_write_json_file`: serialized JSON is written to `<file>.tmp`, flushed with `fsync`, atomically renamed over `<file>`, and committed by `fsync`ing `<data_dir>/kernels`.
2. **Same-version immutability (`allow_same_version_replacement = false`)**:
   - Re-installing `package_id@version` with identical signed `.gkmetadata` (`compute_payload_sha256()`) and valid on-disk artifact hashes is idempotent (or re-enables a disabled package without deleting its directory).
   - Attempting to install `package_id@version` with different content while that version is active or retained in `version_dirs` is rejected with `GUILD_STATUS_ERR_DUPLICATE_REGISTRATION`.
3. **Ordering of publish, unload, and delete**:
   - On install/upgrade (`1.0.0 -> 1.1.0`), the staged directory is renamed to `packages/<id>/1.1.0` and `fsync`ed before `1.0.0` is unloaded.
   - Under `abi::provider_lifecycle_mutex()`, `1.0.0` is unloaded and `1.1.0` is loaded in one atomic provider transaction. If `1.1.0` fails to load or `registry.json` fails to commit, `packages/<id>/1.1.0` is removed and `1.0.0` is restored in memory and on disk.
   - On `remove_package`, `registry.json` is committed **before** `packages/<id>` is deleted from disk.
4. **Startup crash recovery (`GKernelPackageManager::init`)**:
   - Cleans up any leftover `staging/stage_*` directories and `*.json.tmp` files.
   - Sweeps `packages/` and removes any unreferenced package or version directory that was published before an interrupted `registry.json` commit.
   - Fails closed if `trust_store.json`, `registry.json`, or any registered package's `.gkmetadata` is missing or corrupted.

## Verified fault-injection and crash-recovery matrix

All cases below are executed automatically in `tests/core/modular_abi_test.cpp`
(Section 5b and Section 7d):

| Fault / attack scenario | Fault point / input | Expected status | Verified post-condition |
|---|---|---|---|
| Failure during archive read | `PackageFaultPoint::ArchiveRead` | `GUILD_STATUS_ERR_INTERNAL` | `staging/` cleaned up; `1.0.0` remains active and selectable |
| Failure after staging before verification | `PackageFaultPoint::AfterStagingBeforeVerify` | `GUILD_STATUS_ERR_INTERNAL` | `staging/` cleaned up; `1.0.0` remains active and selectable |
| Failure after verification before publish | `PackageFaultPoint::AfterVerifyBeforePublish` | `GUILD_STATUS_ERR_INTERNAL` | `packages/<id>/1.1.0` not created; `1.0.0` remains active |
| Failure after publish before DSO load | `PackageFaultPoint::AfterPublishBeforeLoad` | `GUILD_STATUS_ERR_INTERNAL` | `packages/<id>/1.1.0` rolled back; `1.0.0` remains active |
| Failure during `registry.json` write on upgrade | `PackageFaultPoint::DuringRegistryWrite` | `GUILD_STATUS_ERR_INTERNAL` | `1.1.0` unloaded and removed; `1.0.0` reloaded and active |
| Failure during `registry.json` write on remove | `PackageFaultPoint::DuringRegistryWrite` | `GUILD_STATUS_ERR_INTERNAL` | `packages/<id>/1.0.0` preserved on disk; `1.0.0` reloaded |
| Failure while loading rollback target | `PackageFaultPoint::DuringRollbackLoad` | `GUILD_STATUS_ERR_INTERNAL` | Active `1.1.0` restored in registries and `registry.json` |
| Child process kill (`_Exit(99)`) after publish | `AfterPublishBeforeLoad` (`abort_process = true`) | Child exits `99` | Parent `init()` removes unreferenced `1.1.0` dir; `1.0.0` loads cleanly |
| Child process kill (`_Exit(99)`) before registry rename | `DuringRegistryWrite` (`abort_process = true`) | Child exits `99` | Parent `init()` removes `registry.json.tmp` and `1.1.0`; `1.0.0` loads cleanly |
| Same-version replacement with modified content | Signed `1.0.0` bundle with altered `scope` | `GUILD_STATUS_ERR_DUPLICATE_REGISTRATION` | Rejected both when `1.0.0` is active and when retained as `previous_version` |
| Undeclared `.so` or non-library file in package | `undeclared_evil.so`, `undeclared_payload.txt` | `GUILD_STATUS_ERR_SECURITY_VIOLATION` | Rejected during staging before verification or `dlopen` |
| Trailing bytes in `GKPK` or truncated `.tar` | Appended 8 bytes / partial 200-byte tar block | `GUILD_STATUS_ERR_SECURITY_VIOLATION` | Rejected during staging |
