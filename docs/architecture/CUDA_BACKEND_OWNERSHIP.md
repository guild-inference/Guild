# CUDA backend ownership and extraction gates

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
`remote_expert_opt.cu` must be split: `reduce_experts`/`copy_cpu_rows` move;
host reduction bookkeeping, peer residency, allocation and adaptation stay.
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

Other required boundaries are explicit pending implementation gates:

| Area | Current boundary requiring adaptation |
|---|---|
| GEMM | `Gemm` owns cuBLAS, scratch, workspace and Pascal conversion caches; expose an opaque backend handle with C create/rebind/execute/destroy, preserving full projection batching |
| MMQ | `Context`, `Product`, `GatherGroup`, private GGML pool/state; public POD products and opaque backend-local context, pinned dependency, initialize local device tables before capture |
| KV/attention | `QsaShapes`, `QsaAttnPools`, `KvHostPools`, `KvStreamMap` are private C++ records; define public size-checked POD snapshots, keep page tables/state/budgets host-owned |
| RoPE | `RopeScaling&` and global configuration cannot be exported; public numerical configuration snapshot and backend-local capture-consistent state |
| Native toggles | Process-global enable/configuration flags need explicit transfer before capture; a DSO does not share Core's globals |
| Graph lifetime | Captured kernel nodes retain DSO code references beyond host callback return; destroy graphs and finish all used streams/devices before releasing the model's package pin |
| Multi-device | Two arithmetic kernels mixed into `RemoteExpertOpt`; extract launches while retaining peer stream/residency policy |

These boundaries are unfinished work, not evidence of external execution. Full
external-only CUDA prefill/decode, lifecycle and performance gates remain open
until their implementations and tests pass. A partial-group external-only mode
must name its group and must not be described as an external-only CUDA backend.

## Correctness and performance gates

Keep `platform::configure_device_runtime_policy()` before discovery/package
validation and preserve an embedding application's explicit module-loading choice.
All real inference and benchmarks run serially without concurrent compilation.
Use the same Ornith Q4_K_M, host-only KV, 512MiB reserve, context32768, B256 and
128 generated tokens. Preserve exact logits/greedy IDs/recurrent-state/KV
comparisons and independent numerical oracles. Do not infer execution from a
registration record; bind identities and device-launch instrumentation are
required. Maintain missing-provider and legacy-package compatibility tests.

The prior paired experiment's approximately -0.57% 4K prefill comparison against
the rebuilt baseline remains open. New source-owner/bundled/external/external-only
cases must be compared against preserved executables in ten counterbalanced
rounds at 512/4096/8192/16384 tokens before claiming performance parity.
CUDA13 architecture compilation remains separate from SM61 real-device tests.
