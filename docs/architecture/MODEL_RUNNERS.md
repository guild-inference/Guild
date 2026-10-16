# Model Architecture Runners

SDK 1.1 now supplies the `RMD1` model-description extension in
`include/guild/abi/runner_abi.h`. The independent
[guild-runners-qwen](https://github.com/guild-inference/guild-runners-qwen)
repository uses typed metadata/tensor callbacks and copied C geometry values
instead of delegating description to Core's archetype registry. Real Ornith
inference has passed parity with built-in runner registration disabled.
Qwen4Exp external inference and complete transfer of tensor/topology policy
remain open; see [EXTERNAL_BACKENDS.md](EXTERNAL_BACKENDS.md).

Headers & Implementation:
- [`include/guild/model/runner.hpp`](../../include/guild/model/runner.hpp)
- [`src/model/runner.cpp`](../../src/model/runner.cpp)
- [`include/guild/model/model_descriptor.hpp`](../../include/guild/model/model_descriptor.hpp)
- External Runner Plugin Example: [`plugins/runner_smoke_ext/runner_smoke_plugin.cpp`](../../plugins/runner_smoke_ext/runner_smoke_plugin.cpp)

Model Architecture Runners decouple model-family topology, GGUF metadata validation, and kernel requirements from Guild Core's memory management, scheduler, and server.

---

## 1. Built-In Runners

| Runner ID | Package ID | Architecture ID | Family | Key Architectural Features |
|---|---|---|---|---|
| `runner.qwen4exp.v1` | `builtin.guild-kernels-qwen` | `qwen4exp` | `ModelFamily::Qwen4Exp` | 48 layers (36 Gated DeltaNet + 12 Quad-Sparse Attention, interval 4), 4-stream Hyper-Connections (`hc=4`), 512 routed experts (`top_k=10`) + 1 shared expert, mandatory Per-Layer Embeddings (`ple_compute`), Coupled-Draft MTP |
| `runner.qwen35moe.v1` | `builtin.guild-kernels-ornith` | `qwen35moe` | `ModelFamily::Qwen35MoE` | 40 layers (30 Gated DeltaNet + 10 Full GQA Attention, interval 4), single-stream residual (`hc=1`), 256 routed experts (`top_k=8`) + 1 shared expert, no PLE, native layer-wise batched prefill (`BatchedPrefill`) |

---

## 2. `IModelRunner` Interface & `AbiModelRunnerAdapter`

Every model runner implements `guild::model::IModelRunner` (either directly in C++ for built-in runners or via `AbiModelRunnerAdapter`, which wraps an external C ABI `guild_runner_desc_v1_t` registered from a `.gkernel` plugin):

- `runner_id()`: Stable versioned identifier (e.g., `"runner.qwen35moe.v1"`).
- `arch_id()`: Canonical architecture identifier (`"qwen4exp"`, `"qwen35moe"`, or custom architecture ID).
- `package_id()`: Owning `.gkernel` or built-in package identifier.
- `family()`: `ModelFamily` enum tag.
- `capabilities()`: Runner capability bitmask (`GUILD_RUNNER_CAP_*`).
- `matches_gguf(const GgufMetadata&)`: Inspects GGUF header keys (`general.architecture`, tensor names, layer counts) to determine if this runner supports the artifact.
- `validate_and_describe(const GgufModel&, ModelDescriptor&, std::string& err)`: Populates and validates `ModelDescriptor` from GGUF metadata and tensor headers.
- `configure_geometry(const ModelDescriptor&, core::ModelGeometry&, int64_t& top_k, std::string& err)`: Populates runtime geometry and enforces architectural invariants before device memory allocation.
- `required_operations(const ModelDescriptor&, const core::ModelGeometry&, const memory::ExecutionPlan&)`: Returns the list of `RunnerOpRequirement` entries (`op_name`, `category`, `backend`, `weight_dtype`, `activation_dtype`, `mandatory`) validated by `ModelRunnerRegistry::validate_runner_requirements()`.

---

## 3. Lifecycle & Per-Package Lease Safety

- `ModelRunnerRegistry::register_abi_runner(const guild_runner_desc_v1_t&, std::string& err)` validates `struct_size`, `abi_version` compatibility (`guild_abi_is_compatible`), and non-empty `runner_id`/`arch_id`, and deep-copies strings into `AbiModelRunnerAdapter`.
- When `GuildModel::load()` resolves a runner, it acquires a per-package `SessionRunnerLease(&ModelRunnerRegistry::instance(), runner->package_id())` and stores all resolved kernel `package_id`s in `RunnerExecutionPlan::resolved_package_ids` (held by `SessionKernelLease`).
- While any `GuildModel` or `GuildSession` holds a lease on a runner's `package_id`, `ModelRunnerRegistry::unregister_package()` and `GKernelPackageManager::disable_package()` / `remove_package()` refuse to unload the package with `GUILD_STATUS_ERR_ACTIVE_SESSION_BUSY`.

---

## 4. Building an External Runner `.gkernel` Plugin

See [`plugins/runner_smoke_ext/runner_smoke_plugin.cpp`](../../plugins/runner_smoke_ext/runner_smoke_plugin.cpp) for a complete external runner plugin compiled against `Guild::KernelSDK`:

1. Include `"guild/abi/kernel_abi.h"`.
2. Define a `guild_runner_desc_v1_t` with `struct_size = sizeof(guild_runner_desc_v1_t)`, `abi_version = GUILD_KERNEL_ABI_VERSION_V1`, `runner_id`, `arch_id`, `package_id`, and `probe_gguf` / `validate_model` / `build_execution_plan` callbacks.
3. Export `GUILD_PLUGIN_EXPORT guild_status_t guild_plugin_entry_v1(...)` and call `host_api->register_runner(host, &runner_desc, out_error)` inside the plugin's `init` callback.
4. Package with a `.gkmetadata` manifest declaring `"scope": "runner"` (or `"hybrid"`) and `"provided_runners": ["<runner_id>"]`, sign with `guild kernel sign`, and install with `guild kernel install`.
