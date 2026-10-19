# CUDA backend ownership and extraction gates

## Current boundary (Phase 2D, 2026-10-19)

`guild-kernels-cuda` 1.4.0 owns the dense, attention/recurrent, MoE/device, and
prefill GEMM/MMQ/fused-MoE primitive groups formerly under Core's
`src/kernels/cuda/` and `src/prefill/` (`gemm.cu`, `kernels.cu`, `moe_fused.cu`,
`moe_fused_iq.cu`, `moe_mmq.cu`, `ggml_cuda_host.cu`, and `hipblaslt_tuning.hpp`).
Core builds these groups from `guild-kernels-cuda` (`cmake/CUDABackend.cmake` and
`cmake/GuildCUDAPrefillSources.cmake`) or binds their independently built
`.gkernel` callbacks. SDK 1.4.0 exposes CDD1, CDA1 and CDM1; Kernel ABI remains
v1.0.0.

Only four `.cu` files remain under Core's `src/`
([`bench/results/phase2d-decode-scaling/ownership/ownership.json`](../../bench/results/phase2d-decode-scaling/ownership/ownership.json)):
`src/core/device.cu` and `src/core/pinned.cu` (device allocation/mapping
orchestration) and the two standalone parity test programs
`src/prefill/{gdn_rec_parity,gemm_bf16_parity}.cu` (down from 55 `.cu` files and
297 device kernels in Stage A, and 10 `.cu` files and 62 device kernels after
Phase 2C). Both Ornith-1.5-35B and Qwen3.8-Flash-Next (`UD-IQ4_XS`) pass bitwise
first-token logit and greedy continuation parity.

See the [bug audit](../quality/PHASE2C_CUDA_AUDIT.md) and
[paired measurements](../performance/PHASE2C_CUDA_VALIDATION.md).

## Preserved Stage A baseline

Audit base: Core `73905fae6bf1df79d1e6c824aa87313d4fa67ed8`, CPU
`264227c74d0151b914d826b0fa35a0f576a117b8`, CUDA
`aa1c565762217eb23082f94c95cad19b26661e4f`, Qwen runners
`0ccff556724fc7fc8d66795ce157485e23dfbdaa`. All four local HEADs matched
their published `main` references before this audit. Core remains v0.2.0-dev.

## Exhaustive source inventory

[`ownership-table.md`](../../bench/results/phase2c/ownership-baseline/ownership-table.md)
classifies every one of the 55 CUDA translation units under `src/`.
[`ownership.json`](../../bench/results/phase2c/ownership-baseline/ownership.json)
records file hashes, includes, named host definitions, and 297 detected device
kernel definitions with line numbers. The scanner is a source inventory, not a
C++ AST; macros/templates and multiline declarations need the matching header
review. No device-kernel definitions were found in Core's `.hpp` files.

Reproduce with `python3 tools/audit_cuda_ownership.py --output OUTPUT`.
`--snapshot-build BUILD` additionally preserves executables, compile commands,
CMake settings and build provenance without replacing an existing reference.
The initial reference is under the ignored `build-phase2c-audit/reference/`:

| Executable | SHA-256 |
|---|---|
| guild | `6d9449c31725284fdde4c790d74f46d61aac5d597b4523ab1a1999c6926c4caf` |
| ornith_prefill_bench | `735f737b9913d4da4cd00ceb7b5559be4c87a26961ff957ffef41e0b5d679115` |
| external_runner_parity_test | `48a2dba5334d343093f8911a54bc60e28c6ac0dbbb9bf22e76f0e482940b89ac` |

The separately preserved Phase1 binary and its paired results remain intact.

## Classification and dependency graph

- **A — generic computation:** projections, quantization, GEMV/MMQ/GEMM,
  normalization, routing, elementwise operations, attention, GDN, KV movement,
  RoPE, sampling and verification device primitives belong in the CUDA backend.
- **B — specialized computation:** fixed-width hyper-connections, PLE and cvec
  kernels belong in an explicitly isolated Qwen specialization within the backend
  for this extraction. Model topology/operation policy stays in the runner/Core.
- **C — orchestration:** `device.cu`, `pinned.cu`, `prefill.cpp`, `layer.cpp`,
  `verify.cpp`, `mtp.cpp`, expert residency and runtime/session code retain memory
  ownership, graph construction, scheduling and high-level sequencing in Core.
  `device.cu`'s poison kernel is an allocation diagnostic, not inference math.
- **D — interfaces/utilities:** public C descriptors, shape records and callbacks
  belong in the SDK. Precision helpers and pinned GGML codebooks belong to the
  arithmetic source owner. Tests remain independent consumers. The MMQ caching
  pool and cuBLAS handle are backend-private implementation resources; the host
  selects budgets, streams and caller-owned buffers through public contracts.

```text
installed C SDK -> independent CUDA arithmetic -> cudart/cuBLAS/pinned GGML
      |                       ^
Core package loader -> model-time compatible selection + lifetime pin
      |                       |
Core allocation/scheduling -> direct callbacks / captured device graphs
```

`guild_kernels -> guild_core` is a present static-library dependency. Moving
`guild_engine` wholesale would move scheduling and create a dependency cycle.
`remote_expert_opt.cu` was split in Stage D: `reduce_experts`/`copy_cpu_rows`
now live in the backend's `src/moe/remote_expert_kernels.cu`; Core's
`src/core/remote_expert_opt.cpp` retains reduction bookkeeping, peer residency,
allocation and adaptation.
The two prefill parity `.cu` programs remain tests, not runtime providers.
The GGML MMQ quantizer and template-instance sources selected by CMake are
additional pinned dependency inputs, not Core-owned CUDA implementations.

## Public contract decisions

Kernel ABI v1 already provides tensors, sizes/strides, memory ownership/lifetime,
device capabilities, scratch, stream/capture modes, status codes and model-time
selection. Its descriptor layouts and the old seven-pointer prefill table remain
frozen. The PFKE router/GDN extension and previously released packages remain
readable.

New execution groups need independent tagged, versioned, size-checked C
extensions. BF16/native-MMVQ is the first group: caller-owned buffers, explicit
geometry/strides/stream, no backend allocation or waits on the normal ordered
stream, error status instead of a C++ exception crossing the DSO boundary. A
model binds once; no general registry lookup belongs in a layer/token loop.

Stage B (`guild-kernels-cuda` `ee219c2aa25b67768a51d1c3d3801fd00fb5eb51`, SDK 1.4
[`cuda_dense_abi.h`](../../include/guild/abi/cuda_dense_abi.h)) migrated
`bf16_gemv.cu`, `native_bf16.cu`, `native_mmvq.cu`, `iq_kernels.cu`, and the six
shared precision headers (`bf16_gemv.hpp`, `native_mmvq.hpp`, `iq_kernels.hpp`,
`bf16_bits.hpp`, `dp4a.hpp`, `q8_1_finite.hpp`) out of Core into
`guild-kernels-cuda`. Core binds the copied `guild_cuda_dense_ext_v1_t` (`CDD1`)
payload from `GUILD_OP_QUANTIZED_GEMV` once per model load (`resolve_cuda_dense`)
and dispatches batched prefill, verifier graph capture, and decode BF16/Q8_1/MMVQ
calls through [`cuda_dense_bridge.cpp`](../../src/kernels/cuda_dense_bridge.cpp)
without in-tree BF16 or native-MMVQ CUDA kernels. `-DGUILD_BUILTIN_CUDA_DENSE=OFF`
omits `libguild_cuda_dense.a` from Core and requires a signed CDD1 `.gkernel`
package (`org.guild.kernels.cuda@1.1.0`), verified for fail-closed missing-provider
rejection, `guild_cuda_dense_call_count_v1` launch instrumentation, and bitwise
parity on Ornith-1.5-35B against the frozen Stage A reference binary.

Stage C (`guild-kernels-cuda` `51919eb5bd01d123a91a78ff9e6f136b2d807444`, SDK 1.4
[`cuda_attn_abi.h`](../../include/guild/abi/cuda_attn_abi.h)) migrated all 18
attention, recurrent (GDN), RoPE, KV movement/quantization, and QSA selection CUDA
files (`fused_gdn.cu`, `gdn.cu`, `gdn_multi.cu`, `kv_q4.cu`, `kv_q8.cu`,
`kv_stream.cu`, `native_flash_attn.cu`, `native_gdn.cu`, `native_gdn_preprocess.cu`,
`native_qsa.cu`, `native_qsa_indexer.cu`, `native_qsa_score.cu`, `native_rope.cu`,
`qsa.cu`, `qsa_decode_attn.cu`, `qsa_prompt_attn.cu`, `qsa_select.cu`, `rope.cu`)
and 22 shared headers out of Core into `guild-kernels-cuda` v1.2.0. Core binds the
copied `guild_cuda_attn_ext_v1_t` (`CDA1`) payload from `GUILD_OP_ATTENTION_DECODE`
once per model load (`resolve_cuda_attn`) and dispatches batched prefill, verifier
graph capture, and decode attention/GDN/RoPE/KV/selection calls through
[`cuda_attn_bridge.cpp`](../../src/kernels/cuda_attn_bridge.cpp).
`-DGUILD_BUILTIN_CUDA_ATTN=OFF` omits `libguild_cuda_attn.a` from Core and requires
a signed CDA1 `.gkernel` package (`org.guild.kernels.cuda@1.2.0`).

Stage D migrated MoE routing/combine/shared/grouped experts, S-family GEMV and
dequantization, elementwise/doorbell, sampler and verifier kernels, and the
isolated Qwen GR/PLE/cvec specialization. The package exports
`external.cuda.moe.v1`; Core copies `guild_cuda_moe_ext_v1_t` from
`GUILD_OP_MOE_COMBINE` at model load. `cuda_moe_bridge.cpp` translates host records
to C POD commands; `cuda_moe_layout.cpp` retains host scratch-layout calculations
and `cvec.cpp` owns control-vector files and allocations. All three model-bound
tables are scoped across execution and component teardown, while package leases
retain the backend code used by captured graphs.

Stage E/F evidence on CUDA12/SM61:

- Installed SDK 1.4.0, standalone backend build using that SDK, and four
  standalone CTests passed. Core's default FetchContent build reported the locked
  CUDA revision and `pinned: true`.
- `cuobjdump` found no cubin/PTX in `libguild_kernels.a`, `libguild_engine.a`,
  `libguild_runtime.a` or `libguild_server.a`. The separate backend archives own
  the extracted arithmetic. `libguild_prefill.a`/`libguild_mmq.a` are residual
  Core CUDA and are outside that zero-fatbin claim.
- With `GUILD_BUILTIN_CUDA_DENSE`, `GUILD_BUILTIN_CUDA_ATTN` and
  `GUILD_BUILTIN_CUDA_MOE` all OFF, absent providers reject model loading. With
  only MoE OFF, the missing operation is `moe_combine`.
- Signed external packages passed 248,320 first-token logits, eight greedy IDs,
  recurrent-state bytes and tested host-KV-prefix equality. Model-bound package
  IDs and DSO counters verify dense, attention and MoE execution. Bundled fallback
  also passed after disabling the external arithmetic packages.

Remaining boundaries and validation limits:

| Area | Current boundary requiring adaptation |
|---|---|
| GEMM | `Gemm` owns cuBLAS, scratch, workspace and Pascal conversion caches; expose an opaque backend handle with C create/rebind/execute/destroy, preserving full projection batching |
| MMQ | `Context`, `Product`, `GatherGroup`, private GGML pool/state; public POD products and opaque backend-local context, pinned dependency, initialize local device tables before capture |
| Device configuration | CDA1/CDM1 copy state explicitly; multi-device RoPE/table transfer and concurrent per-model configuration still need hardware validation |
| Graph lifetime | Single-device teardown is exercised; concurrent replacement and stream completion across multiple GPUs remain unverified |
| Qwen4Exp | Specialized kernels are externally owned, but complete external PLE/MTP and legacy prefill inference has not passed a real-model gate |

The `.cu` files still under `src/` are `core/{device,pinned}.cu` and
`prefill/{gemm,kernels,moe_fused,moe_fused_iq,moe_mmq,ggml_cuda_host,gdn_rec_parity,gemm_bf16_parity}.cu`.
The last two are test programs. The device allocation diagnostic also remains
Core-owned. Ornith external execution does not establish that these residual
implementations have moved.

The [post-CDM1 inventory](../../bench/results/phase2c/ownership-after-cdm1/ownership.json)
records ten remaining Core `.cu` files and 62 scanner-detected device-kernel
definitions (including test/diagnostic code), down from 55 files and 297
definitions in the preserved Stage A inventory. This count is source evidence,
not a measurement of which kernels a model executes.

## Correctness and performance gates

Keep `platform::configure_device_runtime_policy()` before discovery/package
validation and preserve an embedding application's explicit module-loading choice.
All real inference and benchmarks run serially without concurrent compilation.
Use the same Ornith Q4_K_M, host-only KV, 512MiB reserve, context32768, B256 and
128 generated tokens. Preserve exact logits/greedy IDs/recurrent-state/KV
comparisons and independent numerical oracles. Do not infer execution from a
registration record; bind identities and device-launch instrumentation are
required. Maintain missing-provider and legacy-package compatibility tests.

The Phase 2C paired experiment compares preserved Phase 1, current pinned bundled
arithmetic, and all external overrides in ten counterbalanced rounds at
512/4096/8192/16384 tokens. Its raw receipts and limitations are linked above.
External-only correctness is a separate gate; this experiment's `all` case still
has bundled fallback available. CUDA13 architecture compilation remains separate
from SM61 real-device tests.
