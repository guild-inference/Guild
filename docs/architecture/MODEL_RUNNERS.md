# Model Architecture Runners

Headers & Implementation:
- [`include/guild/model/runner.hpp`](../../include/guild/model/runner.hpp)
- [`src/model/runner.cpp`](../../src/model/runner.cpp)
- [`include/guild/model/descriptor.hpp`](../../include/guild/model/descriptor.hpp)

Model Architecture Runners decouple model-family topology and invariants from Guild Core's memory management, scheduler, and server.

---

## 1. Built-In Runners

| Runner ID | Package Name | Architecture Name | Family | Key Architectural Features |
|---|---|---|---|---|
| `runner.qwen4exp.v1` | `builtin.guild-kernels-qwen` | `qwen4exp` | `ModelFamily::Qwen4Exp` | 48 layers (36 Gated DeltaNet + 12 Full Attention, interval 4), 4-stream Hyper-Connections, 512 routed experts (`top_k=10`) + 1 shared expert, mandatory Per-Layer Embeddings (`GUILD_OP_PLE_PROJECT`), Coupled-Draft MTP |
| `runner.qwen35moe.v1` | `builtin.guild-kernels-ornith` | `qwen35moe` | `ModelFamily::Qwen35MoE` | 40 layers (30 Gated DeltaNet + 10 Full Attention, interval 4), standard single-stream residual connections (`n_streams=1`), 256 routed experts (`top_k=8`) + 1 shared expert, no PLE, native layer-wise batched prefill (`BatchedPrefill`) |

---

## 2. `IModelRunner` Interface

Every model runner implements `guild::model::IModelRunner` (either directly in C++ or via `AbiModelRunnerAdapter` wrapping a C ABI `guild_runner_desc_v1_t`):

- `runner_id()`: Stable versioned identifier (e.g., `"runner.qwen35moe.v1"`).
- `architecture_name()`: GGUF `general.architecture` string (e.g., `"qwen35moe"`).
- `package_name()`: Owning `.gkernel` or built-in package name.
- `runner_version()`: Monotonic integer version.
- `family()`: `ModelFamily` enum tag.
- `inspect_gguf(const GGUFFile&, ModelDescriptor*, guild_error_info_t*)`: Populates `ModelDescriptor` from GGUF metadata and tensor headers.
- `validate_model(const ModelDescriptor&, guild_error_info_t*)`: Enforces architecture-specific invariants before weight allocation (for example, `Qwen4ExpRunner` rejects any descriptor without `requires_ple == true`, while `Qwen35MoERunner` enforces `n_streams == 1` and `requires_ple == false`).
- `required_ops(const ModelDescriptor&, bool has_gpu_backend)`: Returns the list of `guild_op_kind_t` operations required to run the model on the target backend.

---

## 3. Adding a New Model Architecture Runner

To add support for a new architecture (for example, `deepseek2` or `glm4moe`):

1. **Define or Reuse `guild_op_kind_t` Primitives**:
   - Check whether all layer operations are covered by existing `guild_op_kind_t` categories in `include/guild/abi/kernel_abi.h`.
2. **Implement `IModelRunner` or `guild_runner_desc_v1_t`**:
   - Implement `inspect_gguf` to parse the model's layer count, head dimensions, KV/recurrent layer pattern, and MoE routing parameters into `ModelDescriptor`.
   - Implement `validate_model` to check all mandatory dimensions and invariants.
   - Implement `required_ops` to declare the exact set of kernel operations needed for prefill and decode.
3. **Register the Runner**:
   - For built-in runners, register in `ModelRunnerRegistry::register_builtin_runners()`.
   - For external `.gkernel` plugins, export `guild_plugin_get_descriptor_v1` and call `host_api->register_runner(&runner_desc, &err)` inside the plugin's `init` callback.
4. **Add Tests**:
   - Add unit tests in `tests/core/modular_abi_test.cpp` verifying `find_by_architecture`, `validate_model` (both valid and invalid descriptors), and `verify_runner_kernel_requirements`.
