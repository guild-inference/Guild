# Phase 5.6 Validation: Server Listener Lifecycle & Per-Model Execution Isolation

Phase 5.6 resolves the two P1 findings documented in [PHASE5.5_CLEANUP.md](PHASE5.5_CLEANUP.md):

1. **HTTP server listener and worker shutdown race** (`include/guild/server/server.hpp`, `src/server/server.cpp`,
   `include/guild/server/engine.hpp`, `src/server/engine.cpp`).
2. **Process-wide mutable model/kernel execution globals** across `src/runtime/`, `src/core/`, `src/kernels/`, and
   `src/prefill/` that previously prevented multiple independently configured `GuildModel` instances from safely
   coexisting or executing in one process.

---

## 1. Root-Cause Analysis & Architectural Fixes

### 1.1 HTTP `Server` Listener and Worker Lifecycle (`src/server/server.cpp`)

**Root causes identified in the pre-5.6 implementation:**

* `Server::stop()` called `shutdown(listen_fd_, SHUT_RDWR)`, `close(listen_fd_)`, and `listen_fd_ = -1` on the
  caller thread while `listener_thread_` was concurrently executing `Server::listener_loop()` and reading
  `listen_fd_` to populate `pollfd` and call `accept(listen_fd_, ...)`.
* Closing `listen_fd_` before `listener_thread_.join()` created both a ThreadSanitizer data race on `listen_fd_`
  and a file-descriptor reuse hazard: any concurrent `open()`/`socket()`/`pipe()` in the same process could reuse
  the closed descriptor number before `listener_loop()` exited `poll()`/`accept()`.
* `request_stop()` only set `running_ = false` without waking `poll()`, forcing `listener_loop()` to wait for the
  100 ms poll timeout.
* `stop()` called `engine_->stop()` *after* joining `workers_`, which could deadlock or delay shutdown if a worker
  was blocked inside `engine_->generate_stream()`.
* Client sockets sitting in `client_queue_` or actively blocked in `handle_client()` (`recv()` with a 30 s
  `SO_RCVTIMEO`) were neither shut down nor drained on `stop()`, allowing `stop()` to stall for up to 30 seconds
  on half-open connections or leak unpopped descriptors.
* `MockInferenceEngine::delay_ms_` (`include/guild/server/engine.hpp`) was a plain `int` mutated by
  `set_delay_ms()` while a disconnected streaming worker thread could still be reading it.

**Phase 5.6 lifecycle protocol:**

* `lifecycle_mutex_` serializes `start()` and `stop()` transitions.
* A non-blocking `wake_pipe_[2]` (`O_NONBLOCK`, `FD_CLOEXEC`) is created in `Server::Server()` and closed only in
  `Server::~Server()`. `request_stop()` stores `running_ = false` (`memory_order_release`) and writes one byte to
  `wake_pipe_[1]`, waking `poll()` in `listener_loop()` immediately without polling sleeps.
* `start()` passes `listen_fd` and `wake_pipe_[0]` by value into `listener_loop(int listen_fd, int wake_fd)`. The
  listener thread never reads mutable member fields for its file descriptors.
* `stop()` calls `request_stop()`, acquires `lifecycle_mutex_`, calls `shutdown(listen_fd_, SHUT_RDWR)` (without
  closing the descriptor), cancels in-flight generation via `engine_->stop()` *before* joining workers, closes
  unpopped descriptors in `client_queue_`, and calls `shutdown(fd, SHUT_RDWR)` on all sockets in
  `active_clients_` so workers blocked in `recv()`/`send()` wake immediately while retaining exclusive ownership of
  `close(client_fd)`.
* `close(listen_fd_)` is executed strictly *after* `listener_thread_.join()`, eliminating both the TSan data race
  and any possibility of file-descriptor number reuse while `poll()`/`accept()` is active.

### 1.2 Per-Model Configuration Isolation (`include/guild/core/execution_state.hpp`)

**Complete inventory of process-wide mutable model/kernel state isolated in Phase 5.6:**

| Subsystem / File | Pre-5.6 Mutable Global State | Phase 5.6 Ownership & Isolation Mechanism |
| --- | --- | --- |
| `src/core/layer.cpp` | `g_shared_early`, `g_fused_gr`, `g_fast_attn`, `g_publish_kernel`, `g_fused_gdn`, `g_fast_select`, `native_bf16_projections`, `native_flash_attn_short` | Stored in `core::ModelExecutionState` owned by `GuildModel::Impl::exec_state` and shared with `core::SessionState::exec_state`; bound via `core::ScopedModelExecution`. |
| `src/core/layer.cpp` | `g_kv_resident`, `g_kv_host_only`, `g_kv_int8`, `g_kv_int8_rot`, `g_kv_q4`, `g_kv_hybrid`, non-atomic `g_kv_host_bytes` | KV residency/format fields stored in `core::ModelExecutionState`; `g_kv_host_bytes` made `std::atomic<uint64_t>` and per-model `sp.kv_host_bytes` computed from `impl->ss.qsa_states`. |
| `src/core/session.cpp` | `g_kv_share_staging_mode` (`session_set_kv_share_staging`) | Stored per model in `core::ModelExecutionState::kv_share_staging_mode`; `SessionState` retains `std::shared_ptr<ModelExecutionState> exec_state` and enters `ScopedModelExecution` on `session_init`, `session_release`, `session_zero`, `session_capture`, `session_loop`, `session_token`, `session_capture_token`, and `session_run_token`. |
| `src/core/native_head.cpp` | `g_embed` (`set_native_embed` / `native_embed`) | Stored per model in `core::ModelExecutionState::native_embed`, preventing a second model load or teardown from overwriting or clearing another model's `NativeEmbed*`. |
| `src/kernels/cpu/pack_layout.cpp` | Process-wide `ExpertLayout layout` singleton (`expert_layout()`) | Scoped via `kernels::cpu::ScopedExpertLayout` to `ModelExecutionState::expert_layout`, and captured by value in `core::ExpertSource::layout_` when `FileExpertSource` or `ArenaExpertSource` opens. |
| `src/core/expert_source.cpp` | `FileExpertSource`, `ArenaExpertSource`, and `expert_pool_dispatch*` reading global `expert_layout()` | Reads `this->layout()` / `d.src->layout()` directly from the model's `ExpertSource` instance, including background prefetch/unbuffered I/O threads. |
| `src/kernels/cuda_attn_bridge.cpp` | Global `g_native_qsa`, `g_native_rope`, `g_native_gdn`, `g_native_qsa_indexer`, `g_native_qsa_score`, `g_rope_scaling`, `g_mrope_tab[]`, `g_rope_tab[]` | Stored per model in `kernels::CudaAttnConfigState` (`ModelExecutionState::cuda_attn`), bound via `ScopedCudaAttnConfig` and synchronized to the active backend table on scope entry/exit. |
| `src/kernels/cuda_moe_bridge.cpp` | Global `flags`, `grouped_old`, `gr_fast`, `expert_mode`, `expert_phase` | Stored per model in `kernels::CudaMoEConfigState` (`ModelExecutionState::cuda_moe`), bound via `ScopedCudaMoEConfig` and synchronized to the active backend table on scope entry/exit. |
| `src/prefill/prefill.cpp` | `static const bool all` in `fused_ring()` (line 191) and `static const MmqPlan plan` in `mmq_plan()` (line 839); non-TLS `g_pinned_share`, `g_ring_override`, `g_ring_budget`, `g_ring_small_max` | Replaced process-once `static const` caches that permanently latched the first loaded model's `expert_layout().fmt` with per-layout evaluation; made ring tuning globals `thread_local`. |
| `src/runtime/model_impl.hpp` | Sticky CUDA thread-local error (`cudaErrorMemoryAllocation`) left uncleared after a failed `cudaMalloc` during `GuildModel::load` | Cleared `(void) cudaGetLastError();` in `GuildModel::Impl::~Impl()` so a failed model load due to VRAM exhaustion never poisons subsequent kernel launches on already-loaded models on the same thread. |

**Zero hot-path overhead:** `ScopedModelExecution` checks `if (!raw || raw == tl_active_raw) return;` before doing
any work. Entering `GuildSession::generate` binds the model's `ModelExecutionState` once on the request thread;
nested `Verifier`, `BatchedPrefill`, `Prefill`, `MtpDrafter`, and `SessionState` calls see `raw == tl_active_raw`
and execute a single pointer comparison with zero heap allocations and zero mutex locks.

---

## 2. Build, CTest, and Sanitizer Validation

Measured on dual Intel Xeon Gold 6130 (32 physical cores / 64 threads, 376 GiB RAM across 2 NUMA nodes) + NVIDIA
GeForce GTX 1070 (8 GiB VRAM, SM 6.1, CUDA 12.0, driver `580.178.04`):

| Build / Gate | Command | Result |
| --- | --- | --- |
| CPU-only release (`build-cpu55`) | `ctest --test-dir build-cpu55 --output-on-failure -j4` | **30/30 executed tests pass** (2 skipped by design: `expert_multi_test`, `pool_stress`), 11.74 s |
| ASan + UBSan (`build-cuda12/phase3-asan`) | `ctest --test-dir build-cuda12/phase3-asan --output-on-failure -j4` | **30/30 executed tests pass** (2 skipped), 0 ASan/UBSan findings, 23.97 s |
| ThreadSanitizer (`build-cuda12/phase3-tsan`) | `setarch $(uname -m) -R ctest --test-dir build-cuda12/phase3-tsan --output-on-failure -j4` | **30/30 executed tests pass** (2 skipped), **0 TSan warnings** (including `server_test`, `multi_model_isolation_test`, `gexec_optimizer_test`, `gkmodule_test`, `modular_abi_test`), 27.14 s |
| CUDA 12 release (`build-cuda12`) | `ctest --test-dir build-cuda12 --output-on-failure` | **96/96 tests with present fixtures pass** (4 skipped, 1 pre-existing missing-fixture test `ple_parity`), 351.56 s |

### 2.1 Server Lifecycle Stress (`server_test`)

`tests/server/server_test.cpp` (Tests 1–15) verifies under Release, ASan+UBSan, and TSan:

* Normal startup, `/health`, `/v1/models`, non-streaming and streaming `/v1/chat/completions`, error handling, and
  clean shutdown (Tests 1–10).
* `stop()` before `start()` and repeated `stop()` after failed socket initialization (Test 11).
* 5 repeated `start()`/`stop()` cycles on the same `Server` instance with idempotent repeat `stop()` calls (Test 12).
* 15 iterations of 4 concurrent threads calling `stop()` simultaneously while the listener is waiting in `poll()`,
  verifying prompt wake-up via `wake_pipe_` (<500 ms total) and zero TSan races (Test 13).
* `stop()` while 6 half-open client connections are blocked in `recv()` inside `handle_client()`, completing in
  `<5 ms` instead of blocking on the 30,000 ms `SO_RCVTIMEO` timeout (Test 14).
* 10 rounds of 4 client threads continuously connecting and sending requests concurrently with `Server::stop()` under
  ThreadSanitizer with zero data races or descriptor errors (Test 15).

### 2.2 Multi-Model Isolation & Coexistence (`multi_model_isolation_test`)

`tests/core/multi_model_isolation_test.cpp` verifies:

1. **Concurrent `ScopedExpertLayout` thread isolation**: 6 threads x 500 iterations mutating and reading thread-scoped
   `ExpertLayout` instances with distinct layer counts, expert counts, blob sizes, and `gate_up_interleaved` flags.
2. **Concurrent `ModelExecutionState` scope isolation**: 6 threads x 300 iterations toggling `shared_early`,
   `fused_gr`, `native_bf16`, `kv_resident`, `kv_host_only`, `kv_int8`, `kv_q4`, `kv_hybrid`, `kv_share_staging`,
   `native_embed`, `gate_up_interleaved`, `rope_scaling`, and CUDA attention/MoE bridge flags, verifying both nested
   scopes and non-mutation of process defaults.
3. **Real two-model coexistence, concurrency, failed-load recovery, GEXEC isolation, and asymmetric teardown**:
   - **Model A**: `ornith:35b` loaded from GGUF + pack directory, 2,048 context, `Resident` `FP16` KV,
     `shared_early = true`, `RopeScalingType::None`, portable contiguous expert layout (`gexec_active = false`,
     `gate_up_interleaved = false`). Isolated reference tokens: `271 248068 198 760 1156 369`.
   - **Model B**: `ornith:35b` loaded simultaneously in the same process from `ornith-1.5-35b-ext.gpack` + a GEXEC
     artifact selecting `cpu.expert_pool.gexec_interleaved_avx2` (`gexec_active = true`,
     `gate_up_interleaved = true`), 4,096 context, `HostOnly` `INT8` KV, `shared_early = false`,
     `RopeScalingType::Linear` (`freq_base = 1.05`, `factor = 1.0`). Reference tokens:
     `781 513 5904 3808 660 14145` (`refA != refB`).
   - **Sequential A -> B -> A**: re-running Model A after Model B loads and executes produces the exact `refA` token
     sequence (`271 248068 198 760 1156 369`).
   - **Failed load & VRAM exhaustion recovery**: attempting an invalid `KvMode::Streaming` load and a 256K-context
     `Resident` `FP16` load (`failed to allocate session buffer (5325 MiB)`) while Model A and Model B are resident
     returns clean errors, clears the sticky CUDA thread error, and leaves both Model A and Model B bit-exact on
     subsequent generation.
   - **Concurrent execution**: running `sessionA->generate()` and `sessionB->generate()` simultaneously on two
     threads produces exact `refA` and `refB` token sequences.
   - **Asymmetric teardown**: resetting `modelA` while `sessionA` is still held causes `sessionA->generate()` to fail
     closed cleanly; destroying `sessionA` tears down Model A's `Impl` while Model B remains resident and continues
     producing exact `refB` tokens.

---

## 3. Counterbalanced End-to-End Performance & Numerical Parity

Measured with `ornith_prefill_bench` (`--prompt 512 --batch 512 --context 32768 --ignore-eos`, 128 generated tokens
with `--spec 0` on Ornith, 64 generated tokens with `--spec 1` on Qwen), two portable and two GEXEC runs per model
(`build-cuda12/phase56-bench/summary.json`) compared against the Phase 5.5 baseline
(`build-cuda12/phase55-bench/summary.json`):

| Model | Mode | Phase 5.5 Prefill tok/s | Phase 5.6 Prefill tok/s | Phase 5.5 Decode tok/s | Phase 5.6 Decode tok/s | Phase 5.6 Load ms | Host RSS GiB | VRAM GiB | Token & Logit Parity |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Ornith-1.5-35B | Portable | 225.66 | **226.71** (+0.5%) | 47.03 | **46.48** (-1.2%) | 6,261 | 21.73 | 2.41 | Bit-identical `.ids` & `.f32` |
| Ornith-1.5-35B | GEXEC (`gexec_fused_avx512`) | 226.10 | **227.56** (+0.6%) | 49.90 | **49.71** (-0.4%) | 6,319 | 21.90 | 2.41 | Bit-identical `.ids` & `.f32` |
| Qwen3.8-Flash-Next | Portable | 91.74 | **95.26** (+3.8%) | 22.10 | **21.84** (-1.2%) | 9,537 | 64.61 | 7.14 | Bit-identical `.ids` & `.f32` |
| Qwen3.8-Flash-Next | GEXEC (`gexec_fused_avx2`) | 90.99 | **95.15** (+4.6%) | 22.69 | **22.57** (-0.5%) | 9,569 | 65.15 | 7.14 | Bit-identical `.ids` & `.f32` |

All portable and GEXEC runs stay well within the machine's normal ±1.5% run-to-run variance, preserve identical VRAM
footprint (`2.41 GiB` Ornith, `7.14 GiB` Qwen), and produce byte-for-byte identical output token IDs (`.ids`) and
first-token full-vocabulary float32 logits (`.f32`) compared to Phase 5.5.
