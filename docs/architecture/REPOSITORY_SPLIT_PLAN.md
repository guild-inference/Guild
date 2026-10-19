# Repository split: ownership and ABI audit

Audit base: `42018db2e63e7febbdb7bcf58a6392166518cdc1`. Local `.opencode/`
is unrelated user work. No implementation files had local edits before the audit.

## Phase 1 verification

The CUDA Release build succeeded on the Xeon Gold 6130 / GTX 1070 (SM 6.1,
CUDA 12.0, driver 580.178.04). Serial CTest: 89 passed, 6 skipped. The external
Ornith router test ran real inference, checked full-vocabulary bitwise logits,
eight greedy tokens, package busy protection and fallback. `ple_parity` needs
an absent external GGUF fixture and is recorded separately, not counted as passing.

The current binary is preserved and benchmarked by `tools/phase2_baseline.py`.
Raw measurements and binary digest belong under `bench/results/external-kernels/`;
the executable and model data are not release source assets.

Clean baseline (one full-shaped warmup, ten runs; context 32768, batch 256,
128 generated tokens):

| Prompt | Prefill median tok/s | Decode median tok/s |
|---:|---:|---:|
| 512 | 240.874 | 43.553 |
| 4096 | 221.878 | 32.997 |
| 8192 | 200.260 | 26.191 |
| 16384 | 173.265 | 18.102 |

Binary SHA-256: `71dca3597ba64f135517e0d32f5dd585171d4913b18a1919a537f926c2decc7c`.

## Ownership map

| Owner | Sources / headers at audit base | Dependencies / lifetime |
|---|---|---|
| Core | `src/runtime/`, `src/server/`, `src/spec/`, `src/memory/`, `src/models/`, `src/abi/` | Model/session ownership, scheduling, HTTP, tokenizer, package store, leases |
| Core | `src/core/{session,weights,expert_source,expert_cache,verify,mtp,layer,native_head,native_dense}.*` | Memory and graph orchestration; invokes backend arithmetic. These files are not wholesale backend candidates. |
| Core | `src/kernels/cpu/pool.cpp`, `pool_affinity_*.hpp` | NUMA-aware work scheduling, progress, core selection. Worker arithmetic must bind once; policy stays here. |
| Core | `src/kernels/cpu/expert_layout.cpp` (pack parsing), `src/ngram/ple_reader.cpp`, `src/platform/direct_file.cpp` | Pack/file interpretation, resident backing, shared lifetime policy |
| CPU | `src/kernels/cpu/{expert,q2_avx2,native_expert,iq_avx512,iq_avx2,kq_avx2,kq_avx1}.cpp` and corresponding arithmetic headers | Quantization, gate/up/down rows, ggml-cpu traits; no CUDA dependency |
| CPU / Core probe | `features.cpp`, ISA probes currently in `expert_layout.cpp` | CPUID and OS XCR0 checked before specialized functions; baseline dispatch TU must have no AVX-512 flags |
| CUDA | `src/kernels/cuda/*.cu`, arithmetic headers in `include/guild/kernels/` | CUDA runtime, pure shape/stride/POD contracts, selected GGML codebooks; graph-capturable stream-ordered launches |
| CUDA | `src/prefill/{gemm,kernels,moe_fused,moe_fused_iq,moe_mmq,ggml_cuda_host}.*` | cuBLAS / pinned GGML MMQ; preserve original compiler arithmetic flags |
| Core | `src/prefill/prefill.cpp`, `src/runtime/batched_prefill.cpp` | Buffer ownership, batching, CPU/GPU work division, pre-bound dispatch |
| Qwen runners | Qwen classes in `src/model/{runner,archetype}.cpp` | GGUF recognition, tensor names, dimensions, PLE, layer topology and requirements |
| Core SDK | `include/guild/abi/`, CMake package, schema | Versioned C ABI, descriptors, host metadata/tensor callbacks; no STL/private runtime classes |

## Dependency graph and cycles

```text
Guild SDK (C structs, callbacks, package schema)
  ^                 ^                         ^
CPU plugin       CUDA plugin             Qwen runner plugin
  |                 |                         |
ggml-cpu        cudart / cuBLAS        host GGUF/tensor read callbacks
  \_________________|_________________________/
                    v
             Core package loader
                    v
      registries -> model binding -> direct calls / CUDA graphs
                    v
     Core scheduler, resident weights, host KV, sessions
```

At the audit base `guild_kernels` links `guild_core`; `guild_engine` links both.
Moving `guild_engine` into a backend would create a scheduling/compute cycle.
`native_expert.cpp` uses a C++ `NativeFmt&` containing no STL, but other CPU APIs
return `std::string` and the expert layout contains vectors. Neither can cross
the plugin boundary as a private C++ object. The compute boundary needs public
C format descriptors and row/quantize callbacks, with Core adapters for its
existing worker interfaces. The pool itself must not be exported.

CUDA currently has process-global configuration (RoPE, native fast paths,
capability caches) and many direct C++ calls from graph construction. Extraction
must preserve configuration state and capture semantics, not just move `.cu`
files. The legacy seven-pointer prefill table has `bool` and `RopeScaling&`;
freeze that table for compatibility and introduce a separate size-checked C
extension for new independently compiled backends.

## SDK and loader blockers found before moving code

1. Runner ABI v1 has no recognition/descriptor/geometry callbacks. Its adapter
   calls Core's `ArchetypeRegistry`; the smoke runner is registration-only.
2. The installed SDK contains only `kernel_abi.h`; private C++ prefill types are
   not standalone C contracts. The documentation's status values and prefill
   layout do not match the header and need correction.
3. `GuildModel::load` discards enabled-package loading errors, masking corruption
   with built-in execution. Fail explicitly instead.
4. Default trust uses a public development signing seed. That identity cannot
   authenticate production packages; tests need an explicitly trusted fixture
   key and real releases need independently supplied private keys.
5. Runner registration rejects a second provider for an architecture, even
   with a distinct runner ID. Provider priority/selection is needed for verified
   fallback and external runner testing.
6. Session counters currently prevent unrelated package unloads. Raw descriptor
   selection followed by later lease acquisition also needs atomic pinning.
7. Installation overwrites version directories before durable registry commit;
   same-version replacement, save failure, and rollback need explicit tests.

## Migration gates

Stages A–H follow the task's order. Each extracted repository must configure
using an installed SDK, have no Guild source/build path in its build inputs,
retain MIT/Strata/GGML notices, pass its own tests, and produce a signed package.
External execution is proven by bound package IDs and instrumentation, not by
package presence alone. No stage is complete merely because registration works.

Qwen4Exp and Qwen35MoE stay together: both use hybrid GDN/attention and MoE;
Ornith is a model variant, not an independent ownership boundary. Future GLM,
AMD and `.gkmodule` repositories are planned only. Guild dispatches `.gkernel`
operations; modules remain build-time reusable dependencies, not dispatchable
runtime providers. GPACK/GEXEC compilation and later optimization phases are
outside this extraction.

## Implemented migration milestone

Guild `4b4ded0` supplies standalone SDK 1.1, `RMD1` model descriptions and
`CPE1` native CPU arithmetic. Three independent repositories now build and
pass their scoped integration tests; see `EXTERNAL_BACKENDS.md`. Real Ornith
inference passed with external native CPU experts, external CUDA router and
external model descriptions together, and with built-in Qwen registration /
native CPU arithmetic disabled. CPU upgrade/rollback ran without a Core rebuild.

This initial slice was subsequently expanded through Phase 2B–2E:
[CPU source ownership](CPU_BACKEND_OWNERSHIP.md) (`org.guild.kernels.cpu@1.1.1`),
[CUDA source ownership](CUDA_BACKEND_OWNERSHIP.md) (`org.guild.kernels.cuda@1.4.0`),
`org.guild.runners.qwen@1.1.0`,
[Phase 2E concurrency](PHASE2E_CONCURRENCY.md),
[Phase 2E package lifecycle](PHASE2E_PACKAGE_LIFECYCLE.md), and
[Phase 2E multi-session validation](../performance/PHASE2E_MULTI_SESSION_VALIDATION.md).
