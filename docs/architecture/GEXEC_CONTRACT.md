# GEXEC v1 Hardware-Specialized Execution Contract (`GEXC`)

GPACK (`.gpack`) is the portable model artifact. GEXEC (`.gexec`, magic `GEXC`) is a disposable, hardware- and runtime-bound execution cache. Phase 5 adds a binary container, an offline optimizer, CPU expert providers, and runtime loading to the Phase 4 contract.

Header: [`include/guild/gexec/gexec_contract.hpp`](../../include/guild/gexec/gexec_contract.hpp)  
Implementation: [`src/gexec/gexec_contract.cpp`](../../src/gexec/gexec_contract.cpp)  
Test Suite: [`tests/core/gexec_contract_test.cpp`](../../tests/core/gexec_contract_test.cpp)

Optimizer: [`src/gexec/gexec_optimizer.cpp`](../../src/gexec/gexec_optimizer.cpp)
Lifecycle and kernel tests: [`tests/core/gexec_optimizer_test.cpp`](../../tests/core/gexec_optimizer_test.cpp)

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

`GExecArtifactIdentity::validate(cached, current, out_reason)` checks compatibility (`GExecInvalidationReason`); the container reader also checks payload integrity:

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
11. `CorruptedDigest`: Header, section table, or payload SHA-256 verification fails on load.

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

## 5. Binary container and installation

The little-endian v1 file has a 64-byte header followed by 64-byte section entries.
The header contains the total length and SHA-256 of the header fields and section
table. Each entry records kind, flags, alignment, offset, length and SHA-256.
Metadata sections are JSON, aligned to 64 bytes. `PackedExpertsBlob` is optional
and aligned to at least 4096 bytes. The reader checks bounds, overlap, alignment,
duplicate section kinds, metadata identities and embedded expert lengths.

Sections: `IdentityAndManifest`, `AutotuneTable`, `HeterogeneousPlan`,
`WorkspaceRecipe`, `CompilationProvenance`, `ExpertPackMeta`,
`PackedExpertsBlob`, and `DiagnosticLog`.
The JSON description is in [`schemas/gexec.v1.schema.json`](../../schemas/gexec.v1.schema.json).

Writers hold `<artifact>.lock`, write a same-directory temporary file, flush it,
and atomically replace the destination. Existing readers keep their mapped
snapshot. Lock files remain in place to avoid locking different inodes under
the same pathname. Cache cleanup takes the same writer lock.

Built-in provider identities include a configure-time SHA-256 of runtime,
kernel and runner source bytes plus compiler options. Installed packages use
their package payload digests. Changing a built-in implementation therefore
invalidates its artifacts even in an uncommitted development build.

## 6. Optimizing and using a model

```sh
guild optimize model.gpack --mode auto --context 2048 --prefill-batch 64 -o model.gexec
guild inspect model.gexec --json
guild verify model.gexec --gpack model.gpack --json
guild run model.gpack --gexec model.gexec
guild cache-clean --dir .gexec_cache --model model.gpack
```

The context, KV configuration, prefill batch and speculative-token count must
match the runtime plan. `--spec` sets the speculative-token count. `--gexec` is
also accepted by `serve`; runtime loading otherwise checks the keyed cache,
`GUILD_GEXEC_PATH`, and a sibling `.gexec`. `GUILD_DISABLE_GEXEC=1` disables it.
An incompatible or incomplete artifact falls back to GPACK with a reason in
the startup profile.

Modes control sampling: `skip` performs no optimization; `quick` uses two
warmups and five timed samples; `auto` uses three and nine; `exhaustive` uses
six and nineteen. Each candidate must support the host and pass bitwise
gate/up, quantization and down-projection checks on real model weights for
every distinct expert format. Single-expert timings, weighted by each format's
layer count, rank the valid candidates. A counterbalanced multi-worker pool and
combine trial then tests every valid candidate against the portable baseline at
the model's decode token-group size and checks bitwise parity before selecting
the winner. The portable provider remains the fallback. Pool timings are
recorded separately from full-model throughput.

Four providers implement fused or row-interleaved execution with AVX2 or
AVX-512. They have lower registry priority than the ordinary CPU provider;
the artifact freezes the chosen ID and version. The interleaved layout is
consumed directly by the CPU kernels and by the native MMQ prefill path.
The fused expert pool publishes quantized gate/up results with release/acquire
synchronization before scheduling down-projection slices.

GPU decode cache and PCIe-offload paths currently require portable expert
layout. An interleaved artifact falls back when those paths are requested.
Zero-copy GPACK mmap likewise needs a portable layout, unless the artifact
contains its own packed expert blob.

## 7. Budgets, resume and regeneration

`--max-time` limits seconds; `--max-memory` and `--max-disk` limit bytes.
`--embed-experts` streams packed weights into the artifact using an
expert-sized working set. Without it, the runtime interleaves during its
ordinary expert load. The portable GPACK is never rewritten.

Each completed pass checkpoints to `<artifact>.progress.json`. The checkpoint
has an identity and digest; stale or corrupt checkpoints are ignored. Budget
exhaustion may install a partial artifact for inspection and resume. Partial
artifacts are not activated for inference. `--no-resume` restarts optimization.

`guild optimize model.gpack -o model.gexec --section workspace` regenerates
one section. Supported names are `autotune`, `plan`, `workspace`, and
`provenance`. Retuning that changes an embedded layout requires a complete
rebuild. Selective regeneration streams the existing expert section and keeps
unrelated section digests intact.

The current workspace recipe records dense arena sizes. GPU/PCIe calibration
and graph capture still run through the runtime when the artifact has no
measured calibration or captured graph. The optimizer does not label CPU pool
timings as end-to-end decode savings or invent a break-even token count.

## 8. Measured results

Measured on a dual-socket Intel Xeon Gold 6130 (32 physical cores, 64 threads,
2 NUMA nodes) with an NVIDIA GeForce GTX 1070 (8 GiB GDDR5, SM 6.1) and NVMe
SSD (`/mnt/models-ssd`), using counterbalanced A/B/B/A runs of
`ornith_prefill_bench` (`--runs 2`, steady-state run 2 reported per trial):

- **Ornith-1.5-35B** (`ornith-1.5-35b-ext.gpack`, 40 layers, 256 experts/layer,
  8 active, 512 prompt tokens, 128 generated tokens, `--batch 512`, `--spec 0`):
  `guild optimize --mode auto` took `272.3 ms` (`19,832 B` artifact) and
  selected `cpu.expert_pool.gexec_fused_avx512`.
  - Load time: `6,232.3 ms` portable vs. `6,190.5 ms` GEXEC.
  - Prefill: `2,248.5 ms` (`227.3 tok/s`) portable vs. `2,230.5 ms` (`229.1 tok/s`) GEXEC.
  - Decode (128 tokens): `2,736.6 ms` (`46.77 tok/s`) portable vs. `2,568.4 ms`
    (`49.84 tok/s`, `+6.56%`) GEXEC.
  - CPU expert pool + combine per 128 decode tokens: `decode_cpu_pool_ms` dropped
    from `1,023.5 ms` to `850.0 ms` (`-17.0%`), `decode_cpu_quant_ms` from
    `69.4 ms` to `6.2 ms`, and `decode_cpu_validation_ms` from `55.7 ms` to `3.7 ms`.
  - Memory: `2.41 GiB` VRAM on both; `21.6 GiB` vs. `21.7 GiB` peak host RSS.
  - Output parity: identical token IDs across all four runs.
  - Break-even: `1.314 ms/token` saved in decode pays back the `272.3 ms`
    optimization pass after `208` generated tokens.

- **Qwen3.8-Flash-Next** (`qwen3.8-flash-next-ext.gpack`, 48 layers, 512
  experts/layer, 10 active, 512 prompt tokens, 64 generated tokens,
  `--batch 512`, `--spec 1`):
  `guild optimize --mode auto` took `604.4 ms` (`23,684 B` artifact) and
  selected `cpu.expert_pool.gexec_fused_avx512`.
  - Load time: `9,481.7 ms` portable vs. `9,428.4 ms` GEXEC.
  - Prefill: `5,364.5 ms` (`95.26 tok/s`) portable vs. `5,357.2 ms` (`95.39 tok/s`) GEXEC.
  - Decode (64 tokens, 38 MTP windows): `2,906.7 ms` (`22.02 tok/s`) portable vs.
    `2,825.0 ms` (`22.66 tok/s`, `+2.90%`) GEXEC.
  - CPU expert pool + combine per 64 decode tokens: `decode_cpu_pool_ms` dropped
    from `1,258.9 ms` to `1,175.0 ms` (`-6.7%`), `decode_cpu_down_ms` from
    `355.5 ms` to `264.4 ms` (`-25.6%`), `decode_cpu_quant_ms` from `21.4 ms` to
    `1.4 ms`, and `decode_cpu_validation_ms` from `56.0 ms` to `5.5 ms`.
  - Memory: `7.14 GiB` VRAM on both; `64.7 GiB` vs. `65.3 GiB` peak host RSS.
  - Output parity: identical token IDs across all four runs.
  - Break-even: `1.276 ms/token` saved in decode pays back the `604.4 ms`
    optimization pass after `474` generated tokens.
