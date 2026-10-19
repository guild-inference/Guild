# Phase 2E concurrency and provider lifetime model

This document describes the lock hierarchy, reference-counted provider handles,
and GPU/CPU teardown ordering that prevent use-after-unload and cross-registry
races in Guild Core.

## Lock hierarchy

All cross-registry and package-lifecycle operations obey a strict lock acquisition
order:

1. `GKernelPackageManager::mutex_` (package store, staging, and `packages_` map)
2. `abi::provider_lifecycle_mutex()` (`include/guild/abi/lifecycle.hpp` — recursive transaction mutex serializing cross-registry publication, enable/disable, unload, and selection)
3. `KernelRegistry::mutex_` or `ModelRunnerRegistry::mutex_` (never held simultaneously; each registry acquires its own mutex only inside `abi::provider_lifecycle_mutex()`).
4. `GuildModel::Impl::generation_mutex` (serializes inference and `reset()` on a loaded `GuildModel`; never acquired while holding registry or package-manager locks).

## Reference-counted provider bindings and per-package leases

- **`KernelBinding` (`std::shared_ptr<const guild_kernel_op_desc_v1_t>`)**:
  `KernelRegistry::select`, `find_by_id`, and `list_operations` return an aliasing `std::shared_ptr` that shares ownership of the internal `StoredOp` record (which holds deep-copied descriptor strings, extension tables, and `std::shared_ptr<void> library_keepalive` for the `dlopen` handle).
- **Runner shared ownership (`std::shared_ptr<const IModelRunner>`)**:
  `ModelRunnerRegistry::resolve_for_gguf`, `resolve_by_arch`, `find_by_id`, and `list_runners` return `std::shared_ptr<const IModelRunner>`. External `AbiModelRunnerAdapter` instances retain `library_keepalive` for their plugin DSO.
- **Composite resolution tables**:
  `ResolvedPrefillKernels`, `ResolvedCudaDense`, `ResolvedCudaAttn`, `ResolvedCudaMoE`, and `RunnerExecutionPlan` retain their resolved `KernelBinding` handles directly alongside function-pointer tables.
- **Lease accounting (`active_package_leases(package_id)`)**:
  Under `abi::provider_lifecycle_mutex()` and the registry mutex, `active_package_leases(package_id)` sums explicit `SessionKernelLease` / `SessionRunnerLease` counts plus `(op.use_count() - 1)` across all registered providers belonging to `package_id`.
  - As soon as `select()` or `resolve_*()` returns a handle to a caller, `use_count() > 1` prevents `unregister_package` or `unload_package_unlocked` from unloading that package even before `SessionKernelLease` is constructed.
  - Unloading or disabling a package checks `active_package_leases(package_id) > 0` for that specific `package_id`, so active sessions bound to package `A` do not block disabling, upgrading, or removing an unrelated package `B`.

## Model, session, and CUDA graph teardown ordering

1. **Shared model execution state (`std::shared_ptr<GuildModel::Impl>`)**:
   `GuildModel` and every `GuildSession` created via `GuildModel::create_session()` share ownership of `GuildModel::Impl`, `Tokenizer`, and the atomic `ready` flag. If `GuildModel` is destroyed while a `GuildSession` still exists, `GuildModel::~GuildModel()` sets `ready = false`; subsequent `GuildSession::generate()` or `reset()` calls lock `generation_mutex`, observe `!ready`, and return cleanly without use-after-free.
2. **Destruction sequence (`GuildModel::Impl::~Impl()`)**:
   When the last reference to `GuildModel::Impl` is released:
   - `prefill`, `batched_prefill`, `ver` (`Verifier`, which owns captured `cudaGraphExec_t` instances and commit streams/events), `mtp` (`MtpDrafter`), and `pool` (`ExpertPool` CPU worker threads) are destroyed first while `ScopedCudaDense`, `ScopedCudaAttn`, and `ScopedCudaMoE` scopes remain active.
   - `main_stream` is synchronized via `cudaStreamSynchronize(main_stream)`, and device buffers (`sbuf`, `d_res`, `arena`, `d_mrope`, `window_R`, `db`) are freed.
   - `cuda_moe`, `cuda_attn`, `cuda_dense`, `cpu_binding`, `runner_exec.bindings`, `runner`, `kernel_lease` (which also synchronizes `main_stream` when the last stream lease for a package drops to zero), and `runner_lease` are reset.
   - `cudaStreamDestroy(main_stream)` is called last.
   Because `StoredOp` and `AbiModelRunnerAdapter` hold `library_keepalive`, `dlclose` cannot run until all CUDA graphs, streams, CPU worker threads, model bindings, and session leases have been destroyed.
