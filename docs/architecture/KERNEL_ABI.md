# Guild Kernel ABI v1 and standalone SDK 1.4

Normative headers:

- `include/guild/abi/kernel_abi.h`: frozen Kernel ABI v1 descriptors and registration.
- `include/guild/abi/runner_abi.h`: optional `RMD1` model-description extension.
- `include/guild/abi/cpu_expert_abi.h`: optional `CPE1` native expert arithmetic.
- `include/guild/abi/cpu_backend_abi.h`: extended CPU backend callbacks.
- `include/guild/abi/cuda_dense_abi.h`, `cuda_attn_abi.h`, `cuda_moe_abi.h`:
  optional `CDD1`, `CDA1`, `CDM1` CUDA execution groups.

The standalone SDK builds without a compiler, GPU toolkit, GGML or Core runtime:

```sh
cmake -S sdk -B build-sdk -DCMAKE_INSTALL_PREFIX=/opt/guild-sdk
cmake --install build-sdk
cpack --config build-sdk/CPackConfig.cmake -B build-sdk
```

External projects using all current contracts use `find_package(GuildKernelSDK 1.4 REQUIRED CONFIG)` and
`guild_configure_gkernel_plugin(target)`. The installed files contain only public
C headers, the manifest schema and CMake configuration. `CUDA_RUNTIME_LIBRARY
Shared` keeps plugins on the shared CUDA runtime. The helper itself does not
prove unloading safety; Core must still retain the package and finish stream work.

## Versioning and compatibility

`GUILD_KERNEL_ABI_MAKE_VERSION(major, minor, patch)` packs 8/12/12 bits:
major at bits 24–31, minor at 12–23, patch at 0–11. ABI v1.0.0 is `0x01000000`.
The `GUILD_MAKE_ABI_VERSION` and `GUILD_ABI_VERSION_*` aliases use the same format.
SDK version 1.4 does **not** change the layout or numeric ABI version of existing
v1 descriptors. New functionality uses size/tag/version-checked extension payloads.

Plugins export `guild_plugin_entry_v1` (`GUILD_PLUGIN_ENTRY_SYMBOL_V1`) with the
`guild_plugin_entry_fn` signature and `guild_plugin_descriptor_v1_t` result.
No STL objects, private C++ classes or exceptions cross the public boundary.
Plugins must handle their own initialization failures and return a status.

## Status values (matching kernel_abi.h)

| Code | Value |
|---|---:|
| OK | 0 |
| INVALID_ARGUMENT | -1 |
| ABI_MISMATCH | -2 |
| UNSUPPORTED_HARDWARE | -3 |
| CAPABILITY_MISMATCH | -4 |
| MISSING_RUNNER | -5 |
| MISSING_KERNEL | -6 |
| DUPLICATE_REGISTRATION | -7 |
| DEPENDENCY_MISSING | -8 |
| INVALID_TENSOR | -9 |
| BUFFER_LIFETIME | -10 |
| OUT_OF_MEMORY | -11 |
| EXECUTION_FAILED | -12 |
| NONFINITE_OUTPUT | -13 |
| SECURITY_VIOLATION | -14 |
| ACTIVE_SESSION_BUSY | -15 |
| INTERNAL | -99 |

Names except `OK` have the `GUILD_STATUS_ERR_` prefix; `OK` is `GUILD_STATUS_OK`.
`guild_error_info_t` carries a fixed message buffer and optional component name.

## Extension contracts

### Legacy prefill

The old seven-pointer `direct_fn_table` is frozen. It is a legacy C++ contract,
including `RopeScaling&`, and must not be used as the public interface of a new
standalone backend. The public `guild_prefill_kernel_ext_v1_t` has a 12-byte
`guild_prefill_ext_header_t` (`struct_size`, `tag`, `version`) followed by `gdn_ab`
and `router_projection` function pointers. Its tag is `0x50464b45` (`PFKE`) and
version is 1. It is not a `guild_ext_header_t` linked list.

### Model description (`RMD1`)

A runner descriptor attaches `guild_runner_model_ext_v1_t` through `next_ext`.
Core copies the extension before plugin init's stack storage can disappear.
It provides recognition, model description, numerical validation and selection
priority. Distinct runner IDs may cover the same architecture at different
priorities; enabled higher-priority providers are considered first.

`guild_model_reader_v1_t` exposes read-only numeric/string metadata and tensor
shape callbacks. Core retains all GGUF files and payloads. Description values
are copied into `guild_runner_model_v1_t`, including model dimensions, recurrent
geometry, periodic full-attention interval, MoE parameters, RoPE, PLE requirements,
prefill strategy and canonical tensor roles. No registry lookup occurs per token.

The current execution contract is limited to canonical tensor names and periodic
hybrid topology. Unsupported mappings are rejected explicitly. This extension
does not yet move every layer equation, tensor validation rule or state policy
out of Core. See `EXTERNAL_BACKENDS.md` for the remaining extraction gates.

### Native CPU experts (`CPE1`)

`guild_cpu_expert_ext_v1_t` attaches to `GUILD_OP_BATCHED_EXPERT_EXEC`. It has
size/tag/version fields and four synchronous callbacks: activation quantization,
hidden quantization, gate/up rows and down rows. `guild_cpu_expert_format_v1_t`
contains fixed-width scalar format values and byte offsets, not backend state.
Core's pool copies the function table once at model load and retains package
leases. Workers use those bound callbacks; scheduling and memory ownership stay
in Core. The independently built DSO initializes its own GGML tables.

## Lifetime and dispatch

SDK 1.2 adds `cpu_backend_abi.h`. `guild_cpu_backend_ext_v2_t` retains the
unchanged CPE1 prefix and appends format-description, canonical planar Q2,
native-Q2 rows, CPU BF16 router and ordered combine callbacks. The canonical
activation image is explicitly 64-byte aligned and 3584 bytes; C and C++ builds
check its frozen layout. Old CPE1 plugins remain readable, but do not supply
the newly appended operations. Core copies and validates only the supported
size-checked prefix before init's stack storage disappears.

The independent CPU 1.1 implementation has oracle tests for the appended
operations, including an AVX2/FMA/F16C canonical-Q2 fallback without VNNI.
Core workers and native-pack format parsing can bind these operations once.
Source ownership transfer and a build without all bundled CPU arithmetic are
separate gates; registering this table alone does not establish either one.

Core owns tensor buffers and scratch; plugins may not retain borrowed pointers
beyond their declared lifetime or free Core allocations. Device launches use
the supplied stream. Models hold the package while bound pointers/graphs exist;
unloading synchronizes work before releasing executable code.

The current registry still has conservative process-wide session checks and
raw select-then-lease APIs. Atomic concurrent selection/replacement across both
registries is an outstanding Phase 2 gate; the existing concurrency test alone
does not prove all replacement races safe.
## CUDA execution extensions (SDK 1.4)

`guild/abi/cuda_dense_abi.h` adds the `CDD1` size-checked extension without
changing Kernel ABI v1, PFKE or the frozen seven-pointer prefill table. It carries
BF16/F32 projection, native Q8_1 quantization and native GGUF MMVQ callbacks,
explicit geometry/strides/stream, caller-owned buffer lifetimes, per-call exact
multi-row selection and C status/error records. It exposes no private C++ class
or CUDA SDK type.

`guild/abi/cuda_attn_abi.h` adds the `CDA1` size-checked extension for CUDA
attention, recurrent (GDN), RoPE, KV cache/streaming and QSA selection
operations. It replaces private C++ records (`QsaShapes`, `GdnShapes`,
`QsaAttnPools`, `KvHostPools`, `KvStreamMap`, `QsaIndexerBuffers`, `RopeScaling`)
and process-global native toggles with public C POD descriptors and explicit
configuration callbacks.

`guild/abi/cuda_moe_abi.h` adds the `CDM1` size-checked extension for CUDA MoE
routing/combine/shared/grouped experts, S-family quantized GEMV/dequant,
activation quantization, elementwise/doorbell operations, token sampling,
speculative verify kernels, and an isolated Qwen specialization (`gr`, `fused_gr`,
`ple`, `cvec`). Discovery/registration is not an execution guarantee;
the CUDA ownership document tracks implementation and real-model gates.
