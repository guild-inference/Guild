# Phase 5.5 cleanup and maintainability audit

Phase 5.5 was a readability, formatting, documentation and maintainability pass over the first-party C++/CUDA
sources. It changed no statement, expression, layout, ABI or build flag. This document is the record of what was
done, how it was checked, and - in the second half - the follow-up work the pass surfaced but deliberately did
not do, in priority order.

Everything below was measured on this machine: dual-socket Intel Xeon Gold 6130 (32 physical cores, 64 threads, 2
NUMA nodes) + NVIDIA GeForce GTX 1070 (8 GiB, SM 6.1, CUDA 12.0.140, driver 580.178.04), GCC 13.3, clang-format
18.1.8.

## 1. What changed

| Area | Change |
| --- | --- |
| Formatting | `.clang-format` + `.clang-format-ignore` + `tools/format_code.py` + CI check; 443 of 461 eligible files reformatted mechanically (include/, src/, tests/, tools/, plugins/, bench/ and sycl/) |
| Conventions | [CODE_STYLE.md](CODE_STYLE.md): 120-column limit and why, the two justified exceptions, exclusions |
| Comments | The load lifecycle, the memory planner, the GPACK reader, the package manager, GEXEC identity/optimizer, the kernel registry, runner resolution, the server and the CLI dispatcher |
| Documentation | [architecture/README.md](../architecture/README.md) (component map and index), [architecture/EXECUTION_LIFECYCLE.md](../architecture/EXECUTION_LIFECYCLE.md) (load to decode, with diagrams), [CONTRIBUTING.md](../CONTRIBUTING.md); stale links in AGENTS.md repaired |

Not machine-formatted, on purpose: Python, CMake, Markdown and JSON. They are conventions in CODE_STYLE.md, not
enforced formatting, because the mechanical passes that exist for C/C++ (clang-format, a pinned version, a
whitespace-only diff) do not exist for them, and a half-formatted Python tree is worse than a consistent one.

## 2. How it was checked

* **Formatting is idempotent.** `python tools/format_code.py --check` is clean twice in a row over all 461
  files, and `tools/format_code.py --check <path>` was extended to accept file paths as well as directories.
* **The C++ test suite is unchanged.** 100 tests before and after: 95 pass, 4 skipped by design
  (`prefill_fused_moe_test`, `prefill_fused_iq_test`, `expert_multi_test`, `pool_stress` - each needs a fixture
  that is not in git), and the same pre-existing fixture failure (`ple_parity`, missing legacy
  `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0` shard). Total time 318.27 s against a 317.96 s baseline.
* **The machine code is equivalent.** Every translation unit was recompiled from pristine `92f0600` with the
  same `compile_commands.json` commands and compared per function: 172 of 188 objects are instruction-for-
  instruction identical after normalisation; 5 `.cu` objects differ only in nvcc's anonymous-namespace name
  hashes; 5 test executables differ only in cold error-path code, because the test `CHECK` macro stringizes the
  expressions whose spacing the formatter rewrote. No object under `src/**` or `include/**` differs.
* **The CPU-only configuration builds and tests.** Reported in the validation log below.
* **Performance is unchanged.** Counterbalanced A/B against the pre-format binaries, same GPACKs and settings;
  see the validation log below.

## 3. Follow-ups, in priority order

These are proposals, each with the reason it is not trivial. None of them was attempted in Phase 5.5: the phase
explicitly excluded refactors, and every item here changes code rather than comments.

### P1 - Functions that are too large to review

Measured with a brace-matching pass over first-party sources (approximate line counts; parity drivers excluded
because their long `main`s are linear checklists by design):

| Lines | Max brace depth | Function |
| ---: | ---: | --- |
| 1,852 | 12 | `Prefill::run_impl` (`src/prefill/prefill.cpp:1978`) |
| 828 | 7 | `Verifier::record_window` (`src/core/verify.cpp:698`) |
| 520 | 5 | `GPackReader::open` (`src/artifact/gpack.cpp:354`) |
| 425 | 6 | `cmd_kernel` (`src/cli/guild_main.cpp:1283`) |
| 303 | 7 | `ExpertPool::drain` (`src/kernels/cpu/pool.cpp:690`) |
| 261 | 7 | `detect_cpu_topology` (`src/kernels/cpu/pool.cpp:88`) |

`Prefill::run_impl` and `Verifier::record_window` are the two that matter most, and both are hazardous to split:
they are the code paths that build and capture the CUDA graphs, so a refactor has to preserve the exact
allocation and capture order, the stream identity and the scratch reuse that Phase 3 verified. A safe first step
would be extract-the-body-only helpers (no ownership changes), each with its own parity test.

`GPackReader::open` is the best-value split: it is a linear sequence of section parses (metadata KV, tensor
index, dense index, native dense index, expert layout), each of which can become a private method with its own
bounds-checked unit test, and it has no device state to preserve.

### P1 - Process-wide kernel settings are globals (Resolved in [PHASE5.6_VALIDATION.md](PHASE5.6_VALIDATION.md))

`core::layer_set_shared_early()` and `core::qsa_set_kv_resident()` (`include/guild/core/layer.hpp`) configure
behaviour for the whole process, and `GuildModel::load` sets them while loading. A second `GuildModel` in the
same process - a second model, or a test that loads two - would silently run with the first model's settings.
Today the CLI, the server and the tests all load one model per process, so this is latent rather than live, but
it is exactly the kind of coupling that is invisible until someone adds a second load. Either move these to a
per-load context object threaded through the kernels, or make the setters refuse a second, conflicting call so
the limitation is enforced rather than assumed. *(Resolved in Phase 5.6 via `core::ModelExecutionState` and
`core::ScopedModelExecution`; see [PHASE5.6_VALIDATION.md](PHASE5.6_VALIDATION.md).)*

### P1 - `Server::stop()` races the listener on the listening fd (Resolved in [PHASE5.6_VALIDATION.md](PHASE5.6_VALIDATION.md))

ThreadSanitizer reports a data race on `listen_fd_` in `src/server/server.cpp`: `stop()` closes the socket and
writes `listen_fd_ = -1` (line 179) while `listener_loop()` reads it to fill in `poll` (line 203). `running_` and
`stopped_` are atomics, but the fd is a plain `int`, so the read is unsynchronised. In practice `shutdown()` wakes
the `poll` and the loop exits; between those two events the listener can poll a descriptor another thread has
closed, and a descriptor number can be reused by an unrelated `open` in the same process. The fix is to give the
listener thread ownership of the descriptor (or make it atomic) and signal shutdown through an eventfd/pipe that
`poll` watches, so no thread reads a descriptor another thread may close. Once it is fixed, add `server_test` to
the TSan CI gate - today the gate is `modular_abi_test` + `gkmodule_test` only, which is why this has never been
caught there. *(Resolved in Phase 5.6 via `lifecycle_mutex_`, `wake_pipe_`, `active_clients_`, and post-join
`listen_fd_` close; see [PHASE5.6_VALIDATION.md](PHASE5.6_VALIDATION.md).)*

This was found while validating this phase, and the race is pre-existing: `src/server/server.cpp` changed here by
a 16-line comment block only, and a per-function comparison of the compiled object against pristine `92f0600`
finds 74 symbols and 0 instruction differences.

### P2 - Duplicated model descriptor tables

`src/cli/guild_main.cpp` and `src/program/generate.cpp` carry the same hardcoded descriptor defaults (at
`guild_main.cpp:100` and `generate.cpp:85`: `n_embd = 2560`, `n_layers = 48`, `n_heads = 24`, `n_kv_heads = 2`,
`head_dim = 256`, `context_length = 262144`, and more below them). A descriptor that the container can supply
should be derived from the container metadata in one place; a hardcoded fallback that differs between two
front ends is a bug waiting for the first architecture variant. `src/core/concurrent_main.cpp` and
`src/core/overlap_main.cpp` share a second block: the two test drivers' argument parsing and arena setup.

### P2 - One generation per process

`GuildSession::generate` takes a model-wide `generation_mutex`, and the model owns one KV staging area and one
verifier. That is a deliberate simplification - it is what makes the shared scratch and the captured graphs
sound - but it means "multi-session" today means "several sessions, queued", not concurrent decode. The
concurrency document ([PHASE2E_CONCURRENCY.md](../architecture/PHASE2E_CONCURRENCY.md)) describes the current
guarantees; a real per-session KV and stream split would be a project with its own benchmark, not a cleanup.

### P2 - Coverage signals

The suite is broad (57 files under `tests/` plus the unit tests under `src/`), and a stem-based scan finds no
test text mentioning any of: `src/runtime/batched_prefill.cpp`, `src/runtime/chat_template.cpp`,
`src/runtime/performance.cpp`, `src/models/{downloader,manifest,registry,sha256,store}.cpp`,
`src/core/{device,pinned}.cu`, `src/kernels/cpu/pool.cpp` (`pool_test` needs the absent `pack/full` fixture) and
`src/abi/ed25519.hpp`. Some of these are exercised indirectly - `runtime_load_failure_test`,
`qwen_runtime_parity_test` and the server tests reach through `GuildModel::load` - but "reached indirectly" is
not the same as a unit test that fails with a specific message. The model store (`src/models/`) is the most
valuable gap: it is the code that writes into the user's data directory and downloads over the network, and it
has no test here at all.

### P3 - TODO markers

Three remain, none of them describing unimplemented functionality in this repository:
`src/kernels/iq_parity.cpp:5` (and its SYCL copy) explains where the generated fixtures come from, and
`sycl/src/program/generate.cpp:2526` records what a past issue number was about. They can be dropped or turned
into a sentence without a tracking number when those files are next touched.

### P3 - Formatting coverage not yet enforced

* `sycl/` is formatted but was not compiled here (no `icpx`, no oneMKL on this machine). Its formatting is
  tool-verified only, and the pass is recorded as formatting-only in the history.
* `bench/results/**` and generated tables (`*_oracle_vectors.inc`, `*_host_tables.inc`) are excluded; they are
  data, not source.
* Python, CMake, Markdown and JSON have conventions but no formatter. If the project wants enforced Python
  formatting, it should be a separate change with its own pinned version and a reviewable whitespace-only diff,
  exactly as this phase did for C/C++.

## 4. Validation log

### 4.1 Builds and test suites

| Build | Result |
| --- | --- |
| CUDA 12 release (`build-cuda12`, SM 6.1, `GUILD_BUILD_TESTS=ON`) | clean rebuild; 100 tests: 95 pass, 4 skipped by design, 1 pre-existing fixture failure (`ple_parity`, missing legacy `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0` shard). Total 318.27 s vs a 317.96 s baseline. |
| CPU-only release (`build-cpu55`, `-DGUILD_ENABLE_CUDA=OFF`, tests ON) | 0 compiler warnings; 31 tests: 31 pass, 2 skipped by design (`expert_multi_test`, `pool_stress` - absent `pack/full` fixture) in 28.68 s. |
| Sanitizers (`build-cuda12/phase3-asan`, `phase3-tsan`, CPU-only) | see 4.4 |

`ple_parity` is unchanged from the baseline and fails for a missing fixture, not a missing feature: PLE coverage
exists in `tests/models/native_dense_ple_key_test.cpp` and in the runtime's PLE binding failure path.

The areas this phase touched are covered by named tests in those runs: `gpack_test` (GPACK container),
`gexec_contract_test` and `gexec_optimizer_test` (GEXEC format, identity, optimization and replay),
`gkmodule_test` and `modular_abi_test` (module ABI and package lifecycle), `server_test` Test 7 (six concurrent
requests through one engine) and `ornith_gkernel_plugin_test` (several live sessions, lease-protected disable and
fallback).

### 4.2 GEXEC artifacts must be rebuilt after any source change

`guild verify build-cuda12/phase5-ornith.gexec --gpack ornith-1.5-35b-ext.gpack` against the post-format binary
answers `{"valid": false, "reason": "bound kernel or module package digest mismatch"}`. That is the contract
working as designed: the built-in provider identity includes a configure-time SHA-256 of the runtime, kernel and
runner source bytes, so a whitespace-only pass invalidates an artifact exactly as a semantic change would.
Rebuilding costs one optimization pass, measured again here at 269.95 ms (Ornith) and 600.28 ms (Qwen),
including the model load.

For a user this is visible as one portable run after an upgrade - the reason is in the startup profile, and
`guild optimize` restores it. Making artifacts survive non-semantic edits would require hashing a normalised
source form or a semantic version instead; that is a deliberate design change, not part of this phase.

### 4.3 Performance A/B, formatted vs pre-format binary

Two counterbalanced runs per configuration (`new, old, old, new, gexec, gexec`), same GPACKs, same settings as
the Phase 5 table (Ornith: 512 prompt / 128 generated / `--spec 0` / `--batch 512`; Qwen: 64 generated /
`--spec 1`), quiet machine. Means of the two runs:

| Model | Configuration | Prefill tok/s | Decode tok/s | Load ms | Host RSS GiB | VRAM GiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Ornith | pre-format binary, portable | 225.75 | 47.14 | 6,236 | 22.38 | 2.41 |
| Ornith | formatted binary, portable | 225.66 | 47.03 | 6,157 | 22.49 | 2.41 |
| Ornith | formatted binary, GEXEC | 226.10 | **49.90** | 6,265 | 22.11 | 2.41 |
| Ornith | *Phase 5 table (GEXEC)* | *229.1* | *49.84* | *6,190* | *21.7* | *2.41* |
| Qwen | pre-format binary, portable | 91.49 | 22.02 | 9,258 | 67.43 | 7.14 |
| Qwen | formatted binary, portable | 91.74 | 22.10 | 9,259 | 67.01 | 7.14 |
| Qwen | formatted binary, GEXEC | 90.99 | **22.69** | 9,264 | 67.19 | 7.14 |
| Qwen | *Phase 5 table (GEXEC)* | *95.39* | *22.66* | *9,428* | *65.3* | *7.14* |

* Portable decode is within 0.2% (Ornith) and 0.4% (Qwen) of the pre-format binary, and within the same range of
  the Phase 5 numbers; prefill, load time, host RSS and VRAM are all within the run-to-run spread of this machine
  (about ±1.5% on throughput, and identical VRAM to two decimals everywhere).
* The GEXEC path still selects `cpu.expert_pool.gexec_fused_avx512` for both models and reproduces the Phase 5
  gains: 49.90 tok/s decode on Ornith (+6.1% over portable in this session) and 22.69 on Qwen (+2.6%).
* Qwen's prefill reads 91 tok/s here against the 95.39 tok/s in the Phase 5 table. The pre-format binary
  measured 91.49 tok/s in this same session, on this same machine, so the difference belongs to the measurement
  session (or the harness settings used for that table), not to this phase. What matters here is the
  like-for-like binary comparison: 91.49 vs 91.74.
* **Token parity**: every run of both models produced byte-identical token id dumps - across the formatted and
  pre-format binaries, and with and without GEXEC.

### 4.4 Sanitizers

CPU-only sanitizer builds (`build-cuda12/phase3-asan`, `phase3-tsan`), all 31 tests in each:

| Configuration | Result |
| --- | --- |
| ASan + UBSan | 31/31 pass in 69.36 s |
| TSan | 29/31 pass in 152.83 s; `gexec_optimizer_test` passes in 16.46 s with a 600 s timeout (the default 90 s is too tight under TSan, so that was a timeout, not a defect); `server_test` fails deterministically on the pre-existing `listen_fd_` race described in P1 above |

The CI sanitizer gate (`modular_abi_test`, `gkmodule_test`) is green under both configurations. The
`server_test` failure is reported as a finding, not fixed: correcting it changes synchronisation, which this
phase excludes.

No ASan report, UBSan report or TSan report was produced by any other test in either configuration.
