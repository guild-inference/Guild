# Guild Kernel ABI v1 Specification

Header: [`include/guild/abi/kernel_abi.h`](../../include/guild/abi/kernel_abi.h)

The Guild Kernel ABI v1 is a strict C99/C11/C++17-compatible binary interface separating **Guild Core**, **Model Architecture Runners**, and **Hardware Kernels**. It avoids passing any C++ standard library types (`std::vector`, `std::string`, `std::function`, or exceptions) across the boundary.

---

## 1. ABI Versioning & Negotiation

- **Packed Version Macro**: `GUILD_MAKE_ABI_VERSION(major, minor, patch)`
  - `major`: 16-bit integer (`GUILD_ABI_VERSION_MAJOR(v)`). Incompatible layout or signature changes increment `major`.
  - `minor`: 8-bit integer (`GUILD_ABI_VERSION_MINOR(v)`). Backwards-compatible additions increment `minor`.
  - `patch`: 8-bit integer (`GUILD_ABI_VERSION_PATCH(v)`). Bug-fix clarifications increment `patch`.
- **Current Version**: `GUILD_KERNEL_ABI_VERSION_V1 = GUILD_MAKE_ABI_VERSION(1, 0, 0)`
- **Compatibility Rule (`guild_abi_is_compatible(host_abi, plugin_abi_min, plugin_abi_max)`)**:
  1. `major(host_abi)` must equal `major(plugin_abi_min)` and `major(plugin_abi_max)`.
  2. `plugin_abi_min <= plugin_abi_max`.
  3. `host_abi >= plugin_abi_min`.
- Every extensible struct begins with `uint32_t struct_size` initialized to `sizeof(struct_type)`. Receivers validate `struct_size >= sizeof(...)` before reading fields.

---

## 2. Status Codes & Structured Error Reporting

All ABI functions return `guild_status_t` (`0 = GUILD_STATUS_OK`, negative values on error) and accept an optional `guild_error_info_t*` out-parameter:

| Status Code | Value | Meaning |
|---|---:|---|
| `GUILD_STATUS_OK` | `0` | Success |
| `GUILD_STATUS_ERR_INVALID_ARGUMENT` | `-1` | Null pointer, zero `struct_size`, or invalid parameter |
| `GUILD_STATUS_ERR_ABI_MISMATCH` | `-2` | Incompatible ABI major/minor version range |
| `GUILD_STATUS_ERR_UNSUPPORTED_OP` | `-3` | Operation kind is not implemented |
| `GUILD_STATUS_ERR_UNSUPPORTED_DTYPE` | `-4` | Unsupported input/weight/output data type |
| `GUILD_STATUS_ERR_UNSUPPORTED_SHAPE` | `-5` | Zero dimension, batch out of range, or unaligned stride |
| `GUILD_STATUS_ERR_UNSUPPORTED_HARDWARE` | `-6` | Missing CPU ISA, GPU vendor, or SM version |
| `GUILD_STATUS_ERR_OUT_OF_MEMORY` | `-7` | Scratch workspace or allocation exceeds budget |
| `GUILD_STATUS_ERR_MISSING_DEPENDENCY` | `-8` | Required package or operation is not registered |
| `GUILD_STATUS_ERR_CORRUPT_METADATA` | `-9` | Malformed `.gkmetadata` or invalid field |
| `GUILD_STATUS_ERR_UNTRUSTED_PACKAGE` | `-10` | Untrusted/revoked publisher, hash mismatch, or forbidden script |
| `GUILD_STATUS_ERR_EXECUTION_FAILED` | `-11` | Kernel launch or execution failure |
| `GUILD_STATUS_ERR_NOT_FOUND` | `-12` | Requested runner or kernel does not exist |
| `GUILD_STATUS_ERR_ALREADY_EXISTS` | `-13` | Duplicate `impl_id` or `runner_id` |
| `GUILD_STATUS_ERR_ACTIVE_SESSION_BUSY` | `-14` | Cannot unregister/unload while a session lease is active |
| `GUILD_STATUS_ERR_INTERNAL` | `-15` | Unexpected internal error |

---

## 3. Memory Spaces, Ownership, and Lifetime Rules

Defined in `guild_buffer_desc_t` and `guild_tensor_desc_t`:

- **Memory Space (`guild_memory_space_t`)**:
  - `GUILD_MEM_HOST_PAGEABLE`: Standard pageable host RAM.
  - `GUILD_MEM_HOST_PINNED`: Page-locked host RAM accessible by DMA/device.
  - `GUILD_MEM_HOST_MMAP`: Memory-mapped file backing (read-only).
  - `GUILD_MEM_DEVICE_VRAM`: Device HBM/GDDR memory.
  - `GUILD_MEM_UNIFIED`: Managed/unified memory.
- **Ownership (`guild_buffer_ownership_t`)**:
  - `GUILD_BUFFER_OWNED_BY_RUNTIME`: Allocated and freed by Guild Core. Kernels **must never** free, reallocate, or retain pointers past their declared lifetime.
  - `GUILD_BUFFER_BORROWED_VIEW`: Non-owning slice/view into a larger buffer.
  - `GUILD_BUFFER_SCRATCH_TRANSIENT`: Temporary workspace slice valid only for a single `execute()` call.
- **Lifetime (`guild_buffer_lifetime_t`)**:
  - `GUILD_LIFETIME_CALL`: Valid only until `execute()` returns (or the enqueued stream operation completes).
  - `GUILD_LIFETIME_STEP`: Valid for the current prefill chunk or decode step.
  - `GUILD_LIFETIME_SESSION`: Valid for the lifetime of `GuildSession`.
  - `GUILD_LIFETIME_MODEL`: Static weights valid for the lifetime of `GuildModel`.

`KernelRegistry::validate_buffer_desc()` and `KernelRegistry::validate_tensor_desc()` enforce alignment (power-of-two, pointer divisibility), non-zero dimensions, and bounds checking against `byte_size`.

---

## 4. Execution Context & Scratch Workspace

`guild_exec_context_v1_t` provides the runtime execution environment to every kernel dispatch:
- `device_kind` and `device_index`
- `stream`: opaque `cudaStream_t` or `hipStream_t` (`NULL` for synchronous CPU execution)
- `num_cpu_threads`: worker count for CPU parallel execution
- `scratch_buffer`: pre-allocated workspace buffer (`GUILD_BUFFER_SCRATCH_TRANSIENT`, `GUILD_LIFETIME_CALL`)
- `capabilities`: pointer to discovered `guild_device_capabilities_v1_t`

Before execution, callers may query `query_scratch(op_params, inputs, num_inputs, outputs, num_outputs, &scratch_req, &err)` to determine required temporary workspace size and alignment without dynamic allocation inside the kernel.

---

## 5. Operation Categories (`guild_op_kind_t`)

ABI v1 standardizes 26 operation categories across dense, recurrent, attention, MoE, speculative, and sampling stages:

- **Norms & Elementwise**: `GUILD_OP_RMSNORM`, `GUILD_OP_ADD_RMSNORM`, `GUILD_OP_SILU_MUL`, `GUILD_OP_RESIDUAL_ADD`, `GUILD_OP_SCALE`
- **Embeddings & Projections**: `GUILD_OP_EMBED_LOOKUP`, `GUILD_OP_GEMV_BF16`, `GUILD_OP_GEMM_BF16`, `GUILD_OP_GEMV_QUANT`, `GUILD_OP_GEMM_QUANT`, `GUILD_OP_QUANTIZE_ACT`
- **Positional & Attention**: `GUILD_OP_ROPE_2D`, `GUILD_OP_ATTN_DECODE`, `GUILD_OP_ATTN_PREFILL`, `GUILD_OP_KV_WRITE`, `GUILD_OP_KV_QUANTIZE`
- **Recurrent / Gated DeltaNet**: `GUILD_OP_GDN_CONV_SILU`, `GUILD_OP_GDN_RECURRENCE`
- **Hyper-Connections & PLE**: `GUILD_OP_HYPER_CONNECTION`, `GUILD_OP_PLE_PROJECT`
- **Mixture-of-Experts**: `GUILD_OP_MOE_ROUTER_TOPK`, `GUILD_OP_MOE_EXPERT_DISPATCH`, `GUILD_OP_MOE_SHARED_GATE_ADD`
- **Head, Sampling & Speculative**: `GUILD_OP_LM_HEAD`, `GUILD_OP_SAMPLER_ARGMAX`, `GUILD_OP_SPEC_VERIFY`
