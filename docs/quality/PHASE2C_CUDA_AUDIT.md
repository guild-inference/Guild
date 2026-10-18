# Phase 2C CUDA Migration Bug Audit

## Scope

This records the Phase 2C source review and local tests of the migrated groups in
`guild-kernels-cuda` 1.3.0, arithmetic commit
`b8b675b5fd7605366c04facd7ced079de1ac6f69`, and their Core adapters. The inventory
below defines the review scope; it is not a claim that every mode was executed on
every device. Legacy `src/prefill/` GEMM/MMQ and Qwen4Exp prefill remain in Core.

- **SDK ABI v1 contracts (`include/guild/abi/`)**:
  - `external.cuda.router.bf16.v1`: existing PFKE router projection callback in `kernel_abi.h`.
  - `external.cuda.dense.v1`: `guild_cuda_dense_ext_v1_t` (`CDD1`, `cuda_dense_abi.h`).
  - `external.cuda.attn.v1`: `guild_cuda_attn_ext_v1_t` (`CDA1`, `cuda_attn_abi.h`).
  - `external.cuda.moe.v1`: `guild_cuda_moe_ext_v1_t` (`CDM1`, `cuda_moe_abi.h`).
- **`guild-kernels-cuda` implementation (`src/`, `include/guild/cuda/`)**:
  - Router: `src/router.cu`
  - Dense/MMVQ/IQ math: `src/dense_backend.cpp`, `src/math/bf16_gemv.cu`, `src/math/native_bf16.cu`, `src/math/native_mmvq.cu`, `src/math/iq_kernels.cu`
  - Attention/GDN/RoPE/KV/QSA: `src/attn_backend.cpp`, `src/attention/{gdn,gdn_multi,fused_gdn,native_gdn,native_gdn_preprocess,rope,native_rope,kv_q8,kv_q4,kv_stream,qsa,qsa_select,qsa_decode_attn,qsa_prompt_attn,native_qsa,native_qsa_indexer,native_qsa_score,native_flash_attn}.cu`
  - MoE/S-family/Elementwise/Sampler/Verify/Qwen4Exp: `src/moe_backend.cpp`, `src/moe/{dequant_bf16,dequant_s2,elementwise,native_moe,native_router,quantize_act,remote_expert_kernels,router_top10,s2_expert_grouped,s2_gemv,s2_gemv_fast,s2_gemv_q8,s2_gemv_quads,s_gemv,sampler,shared_expert,verify_kernels}.cu`, `src/qwen/{cvec,fused_gr,gr,native_gr_norm,native_gr_postops,native_ple_postops,ple}.cu`
- **Guild Core bridges and runtime lifecycle**:
  - `src/kernels/cuda_dense_bridge.cpp`, `src/kernels/cuda_attn_bridge.cpp`, `src/kernels/cuda_moe_bridge.cpp`, `src/kernels/cuda_moe_layout.cpp`, `src/kernels/cvec.cpp`
  - `src/core/remote_expert_opt.cpp`, `src/core/verify.cpp`, `src/core/mtp.cpp`
  - `src/runtime/model.cpp`, `src/runtime/model_impl.hpp`, `src/runtime/batched_prefill.cpp`

---

## Audit Checklist & Findings

### 1. ABI v1 Contract & Descriptor Validation

- **Finding 1 (fixed in `src/moe_backend.cpp`)**: CDM1 needed a negative-device
  check and size checks for nested S-form, shared-expert, PLE, fused-GR, sampler,
  GR-shape/workspace and PLE-postops records before consuming their fields.
  `checked_stream()` now rejects a negative device ID, alongside its existing
  context-size, ABI-major and CUDA-backend checks. `check_nested()` validates the
  consumed records. The existing zero-size S-form compatibility case remains.
  `tests/moe_test.cpp` exercises negative `device_id` and truncated sampler/GR
  records. Core validates extension size, tag, version and callbacks when binding.

### 2. Stream Propagation & CUDA Graph Capture Compatibility

- The adapters forward the caller's stream through the public execution context.
  Core owns allocation, graph capture/replay and synchronization. The standalone
  dense and attention tests exercise capture/replay and malformed requests;
  real Ornith tests exercise verifier capture and recurrent-state commit.
- RoPE/cvec pointers and numerical toggles must be configured before capture.
  `fused_gr_check()` stays in `Verifier::init()`, before capture, including for
  `hc == 1`. The proposed load-time optimization to skip that check was not
  applied, so the paired experiment retains the reference initialization path.
- Host configuration/query operations are distinct from asynchronous arithmetic.
  Capture/replay coverage of Ornith does not validate every Qwen4Exp mode.

### 3. Multi-Device Context Hygiene & Sticky CUDA Error Prevention

- **Finding 2 (Fixed in `guild-kernels-cuda` `src/attention/native_rope.cu` and `src/qwen/fused_gr.cu`)**:
  - `mrope_dev()`, `fused_gr_variant()`, and `fused_gr_check()` called `cudaGetDevice(&d)` and fell back to `d = 0` when no device was active (e.g., `CUDA_VISIBLE_DEVICES=-1`), leaving `cudaErrorNoDevice` as a sticky thread-local CUDA runtime error.
  - **Resolution**: Added `cudaGetLastError()` cleanup whenever `cudaGetDevice(&d) != cudaSuccess` in `native_rope.cu` and `fused_gr.cu`.
- **Finding 3 (Fixed in Core `src/kernels/cuda_dense_bridge.cpp`, `src/kernels/cuda_attn_bridge.cpp`, `src/kernels/cuda_moe_bridge.cpp`, `src/runtime/batched_prefill.cpp`)**:
  - `cuda_dense_bridge.cpp`, `cuda_attn_bridge.cpp`, and `batched_prefill.cpp` hardcoded `c.device_id = 0` instead of querying `cudaGetDevice(&c.device_id)`, while `cuda_moe_bridge.cpp` threw `std::runtime_error("CUDA MoE: cannot query current device")` even for host-only queries (`ple_scratch_bytes`, `gr_workspace_floats`) when `CUDA_VISIBLE_DEVICES=-1`.
  - **Resolution**: Unified `context()` across all bridges and `batched_prefill.cpp` to query `cudaGetDevice(&c.device_id)` and fall back cleanly to `c.device_id = 0` after `cudaGetLastError()` when no device is present.
- **Finding 4 (Fixed in Core `src/core/verify.cpp` and `src/core/mtp.cpp`)**:
  - `Verifier::~Verifier()` and `MtpDrafter::~MtpDrafter()` synchronized streams, destroyed CUDA graphs, and freed device allocations without first activating `const OnDevice on_device(device_);`.
  - **Resolution**: Added `const OnDevice on_device(device_);` at the start of `Verifier::~Verifier()` and `MtpDrafter::~MtpDrafter()`.

### 4. Exception Safety, Resource Leaks & DSO Teardown Order

- **Finding 5 (Fixed in Core `src/kernels/cuda_attn_bridge.cpp`)**:
  - `ScopedCudaAttn::ScopedCudaAttn` assigned `active = &binding.table;` before calling `sync_state(*active);`. If `sync_state` threw an exception, `ScopedCudaAttn`'s destructor would not run, leaving the thread-local `active` pointer dangling.
  - **Resolution**: Invoked `sync_state(binding.table);` before assigning `active = &binding.table;` (matching `ScopedCudaMoE`).
- **Finding 6 (Fixed in Core `src/kernels/cvec.cpp`)**:
  - In `upload_here()`, if the second or third `cudaMalloc` or any `cudaMemcpy` failed, previously allocated `t.dir`, `t.s`, or `t.on` device buffers were overwritten by `t = DevTables{};` without `cudaFree`.
  - **Resolution**: Explicitly freed non-null `t.dir`, `t.s`, and `t.on` before resetting `t = DevTables{};`.
- **Finding 7 (Fixed in Core `src/core/remote_expert_opt.cpp`)**:
   - In `RemoteExpertOpt::reduce`, `std::find_if` was dereferenced without checking `it == peers_.end()`. A rejected CDM1 call could also throw instead of returning the method's `bool`/`err` failure result. This method is not declared `noexcept`.
  - **Resolution**: Added `it == peers_.end()` check and wrapped `guild::kernels::remote_expert_reduce` in `try/catch` to populate `err` and return `false`.
- **Finding 8 (Fixed in Core `src/runtime/model_impl.hpp`)**:
  - `GuildModel::Impl::~Impl()` bound `ScopedCudaAttn` during teardown of `prefill`, `batched_prefill`, `ver`, `mtp`, and `pool`, but did not bind `ScopedCudaDense` or `ScopedCudaMoE`, nor explicitly reset those scopes before `kernel_lease` released external `.gkernel` DSO handles.
  - **Resolution**: Bound `ScopedCudaDense`, `ScopedCudaAttn`, and `ScopedCudaMoE` across engine component teardown and explicitly `.reset()` all three scopes after `cudaStreamSynchronize(main_stream)` and before member destruction reaches `kernel_lease`.

---

## Verification Summary

- **`guild-kernels-cuda` standalone CTest (`1.3.0`, commit `b8b675b5fd7605366c04facd7ced079de1ac6f69`)**:
  - Zero compiler warnings across `nvcc` (CUDA 12.0, `sm_61`) and `g++-12`.
  - `100% tests passed, 0 tests failed out of 4` (`cuda_router`, `cuda_dense`, `cuda_attn`, `cuda_moe`).
  - `compute-sanitizer` did not produce a usable result. The installed tool first
    could not locate `libsanitizer-collection.so`; placing the executable beside
    its injection libraries then reported “Target application terminated before
    first instrumented API call”. The cause is unconfirmed. The actual driver
    was 580.178.04. Finite-output and numerical-oracle passes are separate checks,
    not substitutes for a memory-sanitizer pass.
- **Guild Core CTest (`build-cuda12`)**:
  - 93 tests selected: 89 passed, four skipped, zero failed. `prefill_fused_moe_test`
    and `prefill_fused_iq_test` require SM80+; `expert_multi_test` and `pool_stress`
    skip on this CPU's missing VNNI/VBMI. `ple_parity` was excluded for its absent
    external GGUF. `server_test` was also excluded from this particular run.
    The absent-pack `expert_parity` and `pool_test` are not registered.
  - `tools/test_version_report.py --build build-cuda12`: passed with `"pinned": true`, `"dense_bundled": true`, `"attn_bundled": true`, `"moe_bundled": true`.
  - `tools/test_package_registration.py`: passed for `org.guild.kernels.cpu@1.1.1`, `org.guild.kernels.cuda@1.3.0`, and `org.guild.runners.qwen@1.0.0`.
  - `tools/test_external_runner.py`: bitwise Ornith-1.5-35B parity (248,320 first-token logits, 8 greedy tokens, GDN recurrent state, QSA host-only KV cache) and bundled-fallback verification passed.

The device was a single GTX 1070 (SM61). Multi-device state transfer and unload,
HIP/Windows inference, and full external Qwen4Exp PLE/MTP inference remain
unverified. See [CUDA ownership](../architecture/CUDA_BACKEND_OWNERSHIP.md) for
the remaining source boundary.
