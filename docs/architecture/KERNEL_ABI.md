# Guild Kernel ABI v1 Specification & Plugin SDK

Headers & CMake SDK:
- [`include/guild/abi/kernel_abi.h`](../../include/guild/abi/kernel_abi.h)
- [`cmake/GuildKernelSDKConfig.cmake.in`](../../cmake/GuildKernelSDKConfig.cmake.in)
- Reference External Plugins:
  - [`plugins/ornith_prefill_ext/ornith_prefill_plugin.cu`](../../plugins/ornith_prefill_ext/ornith_prefill_plugin.cu)
  - [`plugins/runner_smoke_ext/runner_smoke_plugin.cpp`](../../plugins/runner_smoke_ext/runner_smoke_plugin.cpp)

The Guild Kernel ABI v1 is a strict C99/C11/C++17-compatible binary interface separating **Guild Core**, **Model Architecture Runners**, and **Hardware Kernels**. No C++ standard library types (`std::vector`, `std::string`, `std::function`) or C++ exceptions cross the ABI boundary.

---

## 1. ABI Versioning & 8/12/12 Bit-Packing

- **Packed Version Macro**: `GUILD_KERNEL_ABI_MAKE_VERSION(major, minor, patch)` (aliased as `GUILD_MAKE_ABI_VERSION(major, minor, patch)`):
  - `major`: 8-bit unsigned integer (`bits 24..31`, extracted via `GUILD_KERNEL_ABI_VERSION_MAJOR_OF(v)` / `GUILD_ABI_VERSION_MAJOR(v)`). Incompatible struct layout or signature changes increment `major`.
  - `minor`: 12-bit unsigned integer (`bits 12..23`, extracted via `GUILD_KERNEL_ABI_VERSION_MINOR_OF(v)` / `GUILD_ABI_VERSION_MINOR(v)`). Backwards-compatible extensions increment `minor`.
  - `patch`: 12-bit unsigned integer (`bits 0..11`, extracted via `GUILD_KERNEL_ABI_VERSION_PATCH_OF(v)` / `GUILD_ABI_VERSION_PATCH(v)`). Bug-fix clarifications increment `patch`.
- **Current Version**: `GUILD_KERNEL_ABI_VERSION_V1 = GUILD_KERNEL_ABI_MAKE_VERSION(1, 0, 0)` (`0x01000000u`).
- **Compatibility Rule (`guild_abi_is_compatible(host_abi, plugin_abi_min, plugin_abi_max)`)**:
  1. `major(host_abi)` must equal `major(plugin_abi_min)` and `major(plugin_abi_max)`.
  2. `plugin_abi_min <= plugin_abi_max`.
  3. `host_abi >= plugin_abi_min`.
- Every extensible struct begins with `uint32_t struct_size` initialized to `sizeof(struct_type)`. Guild Core validates `struct_size >= sizeof(...)` before reading fields and respects `struct_size` prefix bounds when deep-copying extension structs.

---

## 2. Status Codes & Structured Error Reporting

All ABI functions return `guild_status_t` (`0 = GUILD_STATUS_OK`, negative values on error) and accept an optional `guild_error_info_t*` out-parameter:

| Status Code | Value | Meaning |
|---|---:|---|
| `GUILD_STATUS_OK` | `0` | Success |
| `GUILD_STATUS_ERR_INVALID_ARGUMENT` | `-1` | Null pointer, zero `struct_size`, or invalid parameter |
| `GUILD_STATUS_ERR_ABI_MISMATCH` | `-2` | Incompatible ABI major/minor version range |
| `GUILD_STATUS_ERR_UNSUPPORTED_BACKEND` | `-3` | Backend is not supported by the plugin or host |
| `GUILD_STATUS_ERR_UNSUPPORTED_HARDWARE` | `-4` | Missing CPU ISA, GPU capabilities, or SM version |
| `GUILD_STATUS_ERR_UNSUPPORTED_DTYPE` | `-5` | Unsupported input/weight/output data type |
| `GUILD_STATUS_ERR_UNSUPPORTED_SHAPE` | `-6` | Invalid rank, alignment, or batch size |
| `GUILD_STATUS_ERR_CAPABILITY_MISMATCH` | `-7` | Missing required core/runner capability or GEXEC lock mismatch |
| `GUILD_STATUS_ERR_OUT_OF_MEMORY` | `-8` | Scratch workspace or allocation exceeds budget |
| `GUILD_STATUS_ERR_MISSING_KERNEL` | `-9` | Required kernel operation is not registered |
| `GUILD_STATUS_ERR_MISSING_RUNNER` | `-10` | Requested model runner is not registered |
| `GUILD_STATUS_ERR_DUPLICATE_REGISTRATION` | `-11` | Duplicate `impl_id`, `runner_id`, or `package_id` |
| `GUILD_STATUS_ERR_DEPENDENCY_MISSING` | `-12` | Missing package/operation dependency or dependency cycle |
| `GUILD_STATUS_ERR_SECURITY_VIOLATION` | `-13` | Untrusted/revoked publisher, Ed25519/SHA-256 mismatch, or unsafe path |
| `GUILD_STATUS_ERR_NUMERICAL_FAILURE` | `-14` | Non-finite (`NaN`/`Inf`) tensor detected |
| `GUILD_STATUS_ERR_EXECUTION_FAILED` | `-15` | Kernel launch, `dlopen`, or runtime execution failure |
| `GUILD_STATUS_ERR_ACTIVE_SESSION_BUSY` | `-16` | Cannot disable/remove/unload while a model or session lease is active |
| `GUILD_STATUS_ERR_INTERNAL` | `-99` | Unexpected internal error |

---

## 3. Plugin Entrypoint, Transactional Registration & Lifetime Safety

1. **Canonical Entrypoint Symbol**:
   - External `.gkernel` shared libraries export `GUILD_PLUGIN_ENTRY_SYMBOL_V1` (`"guild_plugin_entry_v1"`) with signature:
     ```c
     GUILD_PLUGIN_EXPORT guild_status_t guild_plugin_entry_v1(
         uint32_t host_abi_version,
         guild_plugin_desc_v1_t* out_plugin_desc,
         guild_error_info_t* out_error);
     ```
2. **Transactional Registration (`KernelRegistry::register_plugin_transactional`)**:
   - When a plugin is loaded, Guild Core stages all `register_kernel_op` and `register_runner` calls made inside `plugin_desc.init()` into a temporary staging context:
     - Verifies that every registered kernel/runner's `package_id` matches the `.gkmetadata` `package_id`.
     - Verifies that every `impl_id` is declared in `.gkmetadata`'s `provided_ops` and every `runner_id` is declared in `provided_runners`.
     - Verifies that no duplicate `impl_id`, `runner_id`, or `package_id` exists.
   - If `init()` returns an error or any validation check fails, **zero** kernel operations or runners are committed to `KernelRegistry` or `ModelRunnerRegistry`.
3. **Deep-Copied Descriptors & Extension Payloads**:
   - `KernelRegistry` deep-copies all string fields (`op_name`, `impl_id`, `package_id`), any `direct_fn_table` (`direct_fn_table_size` up to 4096 bytes), and any known `next_ext` extension structs (such as `guild_prefill_kernel_ext_v1_t`, respecting `struct_size` prefix bounds) into registry-owned storage so no dangling pointers into plugin stack/static memory can occur.
4. **Per-Package Reference-Counted Leases & CUDA Stream Synchronization**:
   - `GuildModel` and `GuildSession` acquire `SessionKernelLease` and `SessionRunnerLease` locks on their resolved `package_id`s and register the model's `cudaStream_t` (`main_stream`).
   - While any lease is active on `package_id`, `disable_package`, `remove_package`, and `unregister_package` fail deterministically with `GUILD_STATUS_ERR_ACTIVE_SESSION_BUSY`.
   - When a lease is released or a package is unloaded, Guild Core synchronizes the bound CUDA streams (`cudaStreamSynchronize` / `cudaDeviceSynchronize`), invokes `plugin_desc.shutdown()`, and only then releases the `SharedLibHandle` (`dlclose`).

---

## 4. Zero-Overhead Prefill Direct-Call Extension (`guild_prefill_kernel_ext_v1_t`)

To allow external `.gkernel` plugins to override hot-path `BatchedPrefill` CUDA kernels without per-token ABI marshaling overhead, a kernel descriptor may attach `guild_prefill_kernel_ext_v1_t` (`ext_type = GUILD_EXT_TYPE_PREFILL_KERNEL_V1 = 0x1001u`) via `guild_kernel_op_desc_v1_t::next_ext`:

```c
typedef struct guild_prefill_kernel_ext_v1_s {
    uint32_t struct_size; /* sizeof(guild_prefill_kernel_ext_v1_t) */
    uint32_t ext_type;    /* GUILD_EXT_TYPE_PREFILL_KERNEL_V1 (0x1001u) */
    const void* next_ext;
    void (*rms_norm_weighted)(const float* x, const float* w, float* y, int64_t cols, int64_t rows, float eps, void* stream);
    void (*bf16_gemm)(const uint16_t* w_bf16, const float* x, float* y, int m, int n, int k, void* stream);
    void (*quantize_q8_1)(const float* x, int rows, int cols, void* xq, void* stream);
    void (*quantized_mmvq)(int type, const void* w, const void* xq, float* y, int out_cols, int in_cols, int rows, void* stream);
    void (*router_projection)(const float* x, const uint16_t* w_bf16, float* logits, int64_t in_cols, int64_t n_expert, int rows, void* stream);
    void (*router_topk_softmax)(const float* logits, int64_t n_expert, int64_t k, float* weights, int32_t* ids, int64_t rows, void* stream);
    void (*shared_expert)(const float* x, const void* gate_w, const void* up_w, const void* down_w, const float* gate_inp,
                          float* gate_buf, float* up_buf, float* out_buf, float* scalar_buf, void* xq_buf,
                          int gate_type, int up_type, int down_type, int64_t n_embd, int64_t n_ff, int64_t rows, void* stream);
    void (*moe_combine)(const float* parts, const float* weights, const float* shared, float* y, int64_t n_embd, int64_t k, int64_t rows, void* stream);
} guild_prefill_kernel_ext_v1_t;
```

During `BatchedPrefill::init()`, `KernelRegistry::resolve_prefill_kernels()` selects the highest-priority compatible kernel for each stage and binds any non-NULL function pointers from `guild_prefill_kernel_ext_v1_t` directly into `ResolvedPrefillKernels`, recording the contributing `impl_id` and `package_id` in `bound_impl_ids` and `bound_package_ids`.

---

## 5. Building External `.gkernel` Plugins with `Guild::KernelSDK`

Guild exports a standalone header-only CMake target `Guild::KernelSDK` (`cmake/GuildKernelSDKConfig.cmake.in`) and helper function `guild_configure_gkernel_plugin(<target>)`:

```cmake
find_package(GuildKernelSDK REQUIRED)

add_library(my_custom_kernels SHARED my_custom_kernels.cu)
guild_configure_gkernel_plugin(my_custom_kernels)
target_link_libraries(my_custom_kernels PRIVATE CUDA::cudart)
```

`guild_configure_gkernel_plugin` automatically sets `POSITION_INDEPENDENT_CODE ON`, links `Guild::KernelSDK`, and sets `CUDA_RUNTIME_LIBRARY Shared` so dynamically unloading (`dlclose`) a CUDA `.gkernel` plugin never triggers static `cudart` primary-context teardown in the host process.
