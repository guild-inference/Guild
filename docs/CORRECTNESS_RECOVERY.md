# Correctness recovery

Starting checkout: `f8d0de7`. This campaign restores target execution and error reporting before tuning performance.
No model formats, kernels, or Ornith performance paths are being redesigned.

## Findings rechecked against source and history

| Finding | Verification |
| --- | --- |
| Automatic mock inference | Confirmed in both `cmd_run` and `cmd_serve`. Missing assets or failed native/process initialization could select `MockInferenceEngine` without `--mock`. |
| Successful HTTP responses after failure | Confirmed. Non-streaming handlers ignored the engine return value. Streaming handlers sent 200 before inference and recorded failed generation as `ok`. |
| Process EOF treated as completion | Confirmed. `GuildProcessEngine` could return true without receiving `DONE`. |
| Release-disabled tests | Confirmed in `server_test` and `parity_test`: `assert()` removed both checks and calls such as JSON parsing and socket setup under `-DNDEBUG`. |
| Expert failure ignored | Confirmed in the runtime session: the dispatch failure latch was reset before each window and never checked. |
| Missing PLE integration | Confirmed. The reusable loader never opened its `PleTable` or bound `ss.ple`. The driver at `028e3e6^` did both and refused missing PLE assets. |
| Incorrect profile reader / unfilled GPU slots | Confirmed. The loader treated `STRP` as a pair count and published residency without filling slots. |
| Qwen vocabulary discrepancy | The installed UD-IQ4_XS tokenizer declares **248,320** entries. The loader's constant happens to match this artifact; the **151,936** descriptor default is stale. Neither constant is an acceptable artifact-validation rule. |
| Native tokenizer is not BPE | Confirmed: only vocabulary longest-match encoding is implemented. Supplied token IDs are required for target parity until encoding is independently validated. |
| Ornith long-context attention cap | Source still retains the Qwen selection width without an indexer. This is outside this task's execution-parity scope. |

## CLI and HTTP failure reporting

- Mock execution requires `--mock`, including on CPU-only builds, and is labeled explicitly.
- Native initialization failure is reported and exits nonzero; it does not silently try another execution engine.
- Legacy process-engine code remains. Its generation path now requires a valid `DONE` and propagates `ERR`/EOF.
- `GenerationResult` carries an error message and `finish_reason=error`; rejected results cannot supply a completion body.
- Non-streaming inference failure returns HTTP 500 with an OpenAI error object.
- Streaming failure before the first token returns HTTP 500. After streaming starts, failure emits an error event and closes without a successful finish chunk or `[DONE]`. HTTP status cannot be changed after headers have been sent.
- Engine exceptions are converted to inference errors. Failed requests are recorded as errors, rather than successful requests.
- The native REPL returns nonzero on inference failure and no longer prints a fixed throughput claim.

Release-active regressions cover both HTTP endpoint families, false returns, exceptions, inconsistent success/error results,
and failures after a partial stream. CLI subprocess tests cover missing assets and explicit mock selection. Ornith diagnostic
targets are GPU-only so a fresh CPU test build does not try to include GPU-only implementation headers.

Validated on Ubuntu 24.04, dual Xeon Gold 6130, GTX 1070: CPU Release build and
`ctest --test-dir build-guild -R '^(server_test|cli_failure_test)$' --output-on-failure` passed. The same tests passed in
CUDA Release (`CMAKE_CUDA_ARCHITECTURES=61`, `GUILD_EXPERIMENTAL_SM60=ON`).

## Remaining work in this campaign

Runtime failure latching, finite-output checks, release-active runtime tests, PLE reconnection, and actual Qwen parity are
tracked separately. Passing HTTP tests with an explicitly constructed test double does not prove model inference.

## Runtime failure and release-build recovery

- The native loader catches load exceptions, rejects missing/partial artifacts, validates tensor shapes and propagates
  `NativeDense` failure instead of continuing with skipped weights. Execution metadata and vocabulary are read from the GGUF,
  so stale manifest geometry and default RoPE values do not control the loaded target.
- The established `STRP` reader is used; profile header counts are bounded. GPU slots are admitted, filled, checked byte for
  byte, and only then published as resident. Missing profiles cannot create a nonempty uninitialized tier.
- KV precision is applied using the enum. Unimplemented streaming configurations are rejected rather than mislabeled.
- Expert failures remain latched. The runtime callback rejects missing expert blobs and nonfinite inputs/outputs; failed
  windows drain GPU waits, refuse commit, and invalidate the model. Later requests require a reload.
- Every head-logit entry is checked before a window's token IDs are returned. This is a host readback/check, intentionally
  prioritizing correctness; no kernel has been redesigned to hide its cost.
- Invalid prompt IDs/lengths/sampling are rejected before GPU embedding access. MTP failures are checked, an absent/disabled
  drafter cannot be dereferenced, and draft IDs are only used when a drafter exists.
- No-indexer requests beyond the existing attention selection capacity are refused. This prevents silently truncated
  attention while the generic dense-attention implementation remains a later correctness task.
- Main-session pinned KV and embedding/RoPE registrations are released on model destruction and failed loads.
- `parity_test` now uses release-active checks. Its previously asserted unloaded-tokenizer round trip was false in the source:
  byte IDs encoded but decoded to empty strings. Unloaded encoding is now unavailable. The test checks that failure contract,
  rather than claiming BPE parity. Loaded native text encoding remains unvalidated and generation refuses text-only requests;
  callers can supply token IDs. Full native BPE/template support is a remaining product blocker.

CPU Release: 26 registered tests, 22 passed and four VNNI/VBMI tests explicitly skipped on Xeon 6130. CUDA sm_61 targeted
release regressions passed. `runtime_inference_failure_test` loads a small synthetic model and runs the actual CUDA verifier:
missing expert blobs and NaN output-head scales produce no tokens, cannot commit, invalidate readiness, and release pinned KV.
`runtime_dispatch_test` exercises real native CPU expert rows, source failure latching, and nonfinite outputs on this non-VNNI CPU.

Still to validate: Qwen PLE integration and actual supplied-token parity. The wider planner, multi-model globals, tokenizer,
and Ornith long-context issues need later campaigns; successful synthetic tests do not establish full-model quality.
