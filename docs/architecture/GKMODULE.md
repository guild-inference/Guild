# Runtime modules and namespaced operations (Phase 3)

SDK **1.5.0** adds a C99/C++17 operation service in
[`module_abi.h`](../../include/guild/abi/module_abi.h). Runtime modules use the
existing `GKernelPackageManager`, `KernelRegistry`, `KernelBinding`, trust store,
and Phase 2E publication/recovery path. There is no second module registry.

## Responsibilities

Core owns scheduling, device/memory allocation, package authentication, provider
selection and lifetimes. Runners own architecture semantics and decide which
operations and versions to request. Providers implement arithmetic and validate
their tensor contracts. A module groups reusable operations independently of a
model family or hardware backend. No model-name dispatch is added to Core.

The existing Qwen RMD1 description interface and pre-bound execution tables remain
in use. The new service is available to native runners through the optional host
API tail and to in-process runners through `KernelRegistry::operation_api()`.
It supplies operation building blocks; it does not add a general graph executor
or extend RMD1's periodic recurrent/attention topology to arbitrary models.

## Package format and compatibility

`.gkmodule` is a **GKPK v1 archive with a `.gkmetadata` v1 manifest**, exactly like
`.gkernel`. Directory and tar sources use the same hardened extraction path.
There is no new archive encoding or signature algorithm.

| Package | Runtime role |
|---|---|
| `gkernel` | Existing kernel and/or model-runner plugin; may also supply namespaced operations |
| `gkmodule`, `module_abi: 1` | Reusable operation-only plugin; `scope: "module"`, explicit `library_path`, nonempty `provided_ops`, no `provided_runners` |
| `gkmodule` without `module_abi` | Historical build-time component; still rejected for runtime installation/loading |

A runtime module exports `guild_plugin_entry_v1`, and uses its normal
`init`/`shutdown` callbacks. `package_id`, semantic `version`, publisher,
Kernel ABI range, OS/architecture, backends, artifact hashes and provided IDs
use the existing manifest fields. Descriptor capabilities are authenticated
as part of the signed library artifact. Registered implementation IDs must
exactly match `provided_ops`; failed or duplicate registration aborts publication.

The new opt-in fields are omitted from canonical payloads when absent. Existing
signed `.gkernel` payloads remain valid. Kernel/runner descriptor layouts and
Kernel ABI **1.0.0** remain unchanged. `guild_host_api_v1_t` gains an optional
tail pointer: check `struct_size` before reading `operations`, then check the
service's size and version. Old binaries can use the original prefix. Old hosts
reject runtime modules rather than interpreting them as legacy plugins.

## Dependencies

Use the existing `module_dependencies` array for runtime edges:

```json
{
  "module_id": "example.scan",
  "min_version": "1.0.0",
  "max_version": "1.9.99",
  "abi_tag": "guild.module.v1",
  "publisher_id": "org.illatio.guild",
  "sha256": "<SHA-256 of dependency canonical unsigned manifest>"
}
```

`max_version`, publisher pin and digest pin are optional. Bounds are inclusive;
use an upper bound to prevent a dependency major upgrade. The digest pins the
manifest **payload**, not the archive or signature. A runtime edge must target
an enabled `gkmodule` with module ABI 1. Missing, disabled, incompatible, cyclic,
publisher-mismatched and digest-mismatched dependencies fail before `dlopen`.
Both ordinary `dependencies` and runtime module edges participate in graph
validation, topological load, rollback validation and disable/remove protection.

In historical `.gkernel` metadata, module edges with other ABI tags remain
build-time provenance records. Runtime modules require `guild.module.v1` for
every module dependency. This preserves existing signed build-time declarations
without silently reinterpreting them as runtime dependencies.

## Operation identity and selection

Set `op_kind = GUILD_OP_CUSTOM`, use a namespaced `op_name` such as
`example.sequence.prefix_sum`, and attach a `guild_operation_ext_v1_t` (`GOP1`)
through `next_ext`. `impl_id` remains globally unique. New semantic operation
names require no Core enum changes. Modules may register multiple operations.

GOP1 declares:

- An operation semantic version, independent of package/implementation version.
- Operation-specific capability bits and numerical guarantees.
- Contiguous/outer-strided layouts, permitted memory spaces and alignment.
- Rank bounds; descriptor dimension and batch bounds also apply.
- Explicit sync-mode bits (`GUILD_SYNC_BIT(mode)`, not the legacy enum mask).
- Required `validate`, `query_scratch` and `execute` callbacks.

The operation query specifies an inclusive same-major version interval,
permitted backends, required capabilities/numerics, sync mode, concrete tensor
descriptors, RAM/VRAM budgets and an optional exact implementation ID. Device
ISA/SM bounds come from the existing descriptor and capabilities object.
Dtype IDs include existing quantized formats; heterogeneous input contracts
use `GUILD_DTYPE_ANY` plus the provider's `validate` callback. In GOP1,
`input_dtype`/`output_dtype` constrain every input/output tensor respectively.
The first tensor axis is contiguous (one block stride for quantized formats);
outer axes may have explicit strides. Zero strides denote packed layout.

Resolution checks actual tensor shapes and scratch requests, rather than calling
scratch negotiation with null tensors. Scratch-query failure rejects a candidate.
Rank/layout/dtype/device/capability mismatches and memory budget failures provide
diagnostic errors. Compatible candidates sort by descending priority, then
lexicographically ascending `impl_id`, independent of registration order.
`exact_impl_id` disables fallback. Otherwise another compatible implementation
may be chosen at resolution. Execute never reselects or falls back after a launch.

## C service, ownership and concurrency

1. Copy `*host_api->operations` during plugin `init`. The registration `host` and
   host table are temporary; do not retain them. The copied service's `user`
   belongs to its registry and is valid until that registry is destroyed.
   Device-capability extension tails are not supported by this service version;
   pass `caps.next_ext = NULL`.
2. `resolve` copies the query's tensor descriptors and returns an opaque handle
   owning a Phase 2E `KernelBinding`. Tensor payloads are **not copied or owned**.
3. `describe` borrows the copied descriptor for the handle's lifetime. `scratch`
   reports required buffers. Core/caller allocates them; providers never free them.
4. `execute` checks the context, negotiated contract, scratch, buffer lifetime and
   device before calling the provider. CPU execution is host-blocking. GPU work
   must use the supplied stream; the caller selects `device_id` first.
5. `release` waits for recorded GPU streams before dropping its binding. Keep
   streams and buffers alive until release. For graph capture, retain the handle
   until all graph executions complete and the graph objects are destroyed.

Each handle serializes `scratch`/`execute` calls. Independent handles may execute
concurrently; providers must synchronize their own shared state. Callers must
not release a handle concurrently with a call using it. Handles must be released
before their registry is destroyed. An active handle prevents package unload,
replacement and rollback. Existing model/session ownership and graph teardown
remain under the Phase 2E rules.

Validation/scratch callbacks are pure negotiation callbacks: no registry or
package-manager reentry, retained query pointers, device launches or allocation.
They may be called under the registry lock. Execution callbacks may consume
caller-owned persistent state through tensors or versioned operation parameters;
no C++ types or exceptions cross the ABI. `init` registers descriptors and may
allocate plugin state; `shutdown` cleans it up, including after failed init.
Registration callbacks must run synchronously on the init thread. Concurrent
calls to registry registration/lookup APIs from separate callers are serialized
by the lifecycle transaction; callbacks cannot retain the temporary staging host.
Providers must make partial-init cleanup safe and quiesce their own background
work before shutdown returns.

Package publication now holds the existing lifecycle transaction through the
durable registry write. A failed commit cannot expose a replacement binding to
another thread and thereby prevent rollback. Enable/disable persistence failure
restores the prior registration state as well.

## Development example and future runners

[`plugins/scan_module`](../../plugins/scan_module) builds against only the installed
SDK and supplies inclusive/exclusive sequence scans in C99. These operations are
absent from Qwen. [`gkmodule_test`](../../tests/core/gkmodule_test.cpp) signs and
archives the module, loads competing providers, resolves through the C service,
executes both operations, checks numerical outputs and exercises unloading,
dependency failures, concurrent calls and crash recovery. A dependent module also
calls its base module through the copied host C service.

[`gkmodule_cuda_test`](../../tests/core/gkmodule_cuda_test.cpp) loads a signed CUDA
variant, executes on a non-default stream and inside a captured CUDA graph,
checks output and verifies safe release. Its single-thread kernel is a lifecycle
fixture, not a performance implementation.

See the [example workflow](../../plugins/scan_module/README.md). Future runners
can use the same service for acoustic-token transforms, specialized MTP,
attention/MoE variants, streaming state updates or codec integration. Such
contracts belong in the extension's versioned headers and namespaces. Phase 3
does not implement those models, codecs, GPACK or GEXEC generation/autotuning.

## Security and platform scope

All Phase 2E signature, revocation, hashing, archive validation, immutable-version
and crash-recovery checks apply. Native code executes only after authentication;
signatures do not sandbox a publisher's native library. Do not load untrusted
native code by disabling signature enforcement.

The integration implementation uses the existing POSIX dynamic loader. Windows
SDK compilation is supported, but Windows dynamic loading remains unsupported.
CPU and CUDA module execution are tested on Linux; HIP and macOS runtime module
execution require platform validation. Multi-device capability discovery remains
the existing device-0-oriented interface and has not been validated here.
