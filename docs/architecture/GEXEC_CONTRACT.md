# GEXEC v1 Hardware-Specialized Execution Contract (`GEXC`)

Phase 4 establishes the strict architectural boundary and implementation interfaces between **GPACK** (`.gpack`, the portable, immutable model artifact) and **GEXEC** (`.gexec`, the disposable, hardware- and runtime-bound execution optimization cache prepared for Phase 5).

Header: [`include/guild/gexec/gexec_contract.hpp`](../../include/guild/gexec/gexec_contract.hpp)  
Implementation: [`src/gexec/gexec_contract.cpp`](../../src/gexec/gexec_contract.cpp)  
Test Suite: [`tests/core/gexec_contract_test.cpp`](../../tests/core/gexec_contract_test.cpp)

---

## 1. Separation of Concerns: GPACK vs. GEXEC

| Concern | GPACK (`.gpack` — Portable) | GEXEC (`.gexec` — Machine-Bound Cache) |
| :--- | :--- | :--- |
| **Lifespan** | Immutable source-of-truth model container; distributable across machines | Disposable local optimization artifact; rebuilt automatically when invalidated |
| **Weights** | Canonical GGML/Guild quantization blocks (`[gate \| up \| down]` per expert) | Hardware-specific repacked layouts (e.g. interleaved `[gate_row_i \| up_row_i]`, SIMD tile transposes) |
| **Kernels** | Declares required abstract operations (`guild_op_kind_t`) and capability flags | Locks autotuned `impl_id` / `package_id` winners per `(op, phase, dtype, M, N, K)` |
| **Memory & NUMA** | Agnostic to host RAM size, NUMA sockets, or VRAM capacity | Stores calibrated CPU/GPU/PCIe latencies, NUMA interleave/THP policy, and VRAM expert slot assignments |
| **Compiled Graphs** | None | Stores CUDA/HIP runtime/driver/SM-keyed graph and kernel compilation cache keys |

---

## 2. Hardware Fingerprint & Cache Identity (`GExecArtifactIdentity`)

Every GEXEC artifact is bound to a `GExecArtifactIdentity` whose deterministic SHA-256 cache key (`compute_cache_key()`) incorporates:

1. **`format_version`**: `kGExecVersionV1` (`1`)
2. **`gpack_manifest_sha256`**: 64-hex SHA-256 digest of the source `.gpack` manifest (which covers all section SHA-256 digests).
3. **`core_abi_version` & `module_abi_version`**: Kernel ABI (`GUILD_KERNEL_ABI_VERSION_V1`) and Module ABI (`GUILD_MODULE_ABI_VERSION_V1`).
4. **`runner_id` & `runner_version`**: Resolved `RMD1` model runner identity (e.g. `guild.runner.qwen3_5_moe.v1`).
5. **`bound_packages`**: Canonical sorted list of `(package_id, version, payload_sha256)` for all loaded `.gkernel` and `.gkmodule` packages.
6. **`GExecHardwareFingerprint`**:
   - Host CPU architecture (`x86_64`, `aarch64`), ISA feature flags (`AVX2`, `FMA`, `F16C`, `AVX512F`, `AVX512BW`, `AVX512VL`, `AVX512VNNI`), L1d/L2/L3 cache sizes, physical core count, and NUMA node count.
   - GPU device name, compute capability (`gpu_cc_major`, `gpu_cc_minor`), SM count, total VRAM bytes, L2 cache bytes, memory bus width, and CUDA driver/runtime versions.
7. **Execution Plan Parameters**: `kv_format`, `kv_mode`, `max_context`, `prefill_batch`, and `mtp_spec_tokens`.

---

## 3. Deterministic Invalidation Rules & Fallback Behavior

`GExecArtifactIdentity::validate(cached, current, out_reason)` enforces 10 explicit invalidation checks (`GExecInvalidationReason`):

1. `MagicOrVersionMismatch`: `cached.format_version != current.format_version`
2. `GPackIdentityMismatch`: `cached.gpack_manifest_sha256 != current.gpack_manifest_sha256`
3. `CoreAbiMismatch`: `cached.core_abi_version != current.core_abi_version || cached.module_abi_version != current.module_abi_version`
4. `RunnerIdentityMismatch`: `cached.runner_id != current.runner_id || cached.runner_version != current.runner_version`
5. `ModuleOrKernelDigestMismatch`: Any bound `.gkernel` or `.gkmodule` package ID, version, or payload SHA-256 differs.
6. `CpuIsaMismatch`: Host CPU architecture or required SIMD ISA bitmask differs.
7. `NumaTopologyMismatch`: Number of allowed NUMA nodes differs.
8. `GpuArchitectureMismatch`: GPU device count, compute capability (`SM major.minor`), SM count, or GPU name differs.
9. `CudaRuntimeOrDriverMismatch`: CUDA runtime version differs or driver version was downgraded.
10. `ExecutionPlanMismatch`: `kv_format`, `kv_mode`, `max_context`, `prefill_batch`, or `mtp_spec_tokens` differs.
11. `CorruptPayloadDigest`: Payload SHA-256 verification fails on load.

**Fallback Rule**: An invalid, missing, or corrupt GEXEC cache **never** aborts model loading. Guild logs the exact `GExecInvalidationReason` and falls back to portable GPACK (or GGUF) execution and runtime operation negotiation.

---

## 4. Phase 5 Preparation Interfaces Implemented in Phase 4

### 4.1 Offline Expert Gate/Up Row Interleaving (`GExecExpertPackMode::GateUpRowInterleaved`)
In canonical GGUF/GPACK expert blobs, an expert's weights are stored as `[gate_rows (n_ff * gate_row_bytes) | up_rows (n_ff * up_row_bytes) | down_rows]`. During CPU expert evaluation, each worker thread computes `gate[row]` and `up[row]` simultaneously for the same quantized input activation `q8`, requiring two separate memory streams separated by `n_ff * gate_row_bytes` (~400–800 KiB).

- `pack_expert_gate_up_interleaved(src_blob, dst_blob, n_ff, gate_row_bytes, up_row_bytes, down_bytes)` interleaves row `i` of `gate` immediately adjacent to row `i` of `up` (`[gate_row_0 | up_row_0 | gate_row_1 | up_row_1 | ... | down_rows]`), halving CPU prefetcher stream pressure while keeping `down_rows` contiguous.
- `unpack_expert_gate_up_interleaved(...)` restores the exact canonical layout (verified bit-for-bit in `gexec_contract_test`: `128 us` per 1.2 MiB expert blob).

### 4.2 Operation-Registry Autotune Lock Table (`GExecAutotuneTable`)
- `autotune_operation(op_kind, operation_name, phase, backend, quant_type, m, n, k, required_caps)` queries `OperationRegistry::instance()` (Phase 3) for all registered kernel candidates matching the operation and hardware capabilities, records the selected `impl_id`, `package_id`, and `fallback_impl_id`, and serializes the table with a tamper-evident `payload_sha256` digest.

### 4.3 NUMA / THP & Heterogeneous Placement Plan (`GExecHeterogeneousPlan`)
- Captures the calibrated CPU/GPU/PCIe expert latencies (`calibrated_cpu_ms`, `calibrated_gpu_ms`, `calibrated_copy_ms`), per-layer GPU eligibility (`gpu_layers`), PCIe offload numerators (`pcie_numerators`), VRAM expert cache slot count (`expert_cache_slots`), NUMA policy (`interleave` vs. `process`), and 2 MiB transparent hugepage (`use_thp_hugepages`) selection with SHA-256 integrity verification.

### 4.4 Compilation Cache Key (`GExecCompilationKey`)
- Produces a deterministic 64-hex SHA-256 key over `(cuda_runtime_version, cuda_driver_version, gpu_cc_major, gpu_cc_minor, kernel_package_sha256, compile_flags, graph_topology_signature)` for Phase 5 offline/persistent CUDA graph and kernel binary caching.
