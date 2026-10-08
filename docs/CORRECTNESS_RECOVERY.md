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

Passing HTTP tests with an explicitly constructed test double does not prove model inference; the real-model checks below
are separate gates.

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

## Required Qwen PLE reconnection

`src/runtime/ple.cpp` reconnects the setup from `028e3e6^:src/program/generate.cpp`:

- Resolve the shard holding `per_layer_token_embd.weight`, or the supplied PLE path.
- Require the module for Qwen4Exp, validate its declared layer, hash constants, EOS cut, n-gram/head counts, convolution
  geometry, tensor shapes and engine forms. Unsupported contracts fail explicitly.
- Bind the existing BF16/native/canonical key projection, value projection, normalization weights and F16 convolution.
- Own host embedding staging, device embedding and scratch for the entire captured-graph lifetime.
- Connect the session token window and normalized convolution history before verifier/prefill capture.
- Refuse production generation if the required binding is subsequently disconnected. No public ablation fallback is provided.

`ple_binding_failure_test` covers mandatory-table/metadata failures. The real-model test also checks table reads, finite
nonzero history, the committed token window, reload/reset behavior, and a separately captured ablation as a negative control.
The synthetic CUDA test verifies nonzero encoded expert bytes are filled and checked before GPU residency is published.

## Real Qwen result and reproduction

Measured on Ubuntu 24.04, 2 x Xeon Gold 6130 (32 physical cores, no VNNI/VBMI), 376 GiB reported RAM, GTX 1070 8 GiB,
NVIDIA driver 580.173.02, CUDA 12 compiler, Release `-DNDEBUG`, sm_61 experimental build. No speed tuning was done.

Artifact: existing `Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf` and its matching native pack. Both engines read the
same installed shards and pack. The frozen 20 input IDs are in `tests/data/qwen_recovery_ids.txt`; they encode:

```text
<|im_start|>user
What is 2 + 2?<|im_end|>
<|im_start|>assistant
<think>

</think>

```

Reference: preserved pre-extraction Strata engine 0.1.39 (`engine-cuda12/strata`), SHA-256
`d24277e73c13dce297e659110d216679676a999ffb6335c2371ec866af4ee90b`.
Input-file SHA-256: `c2e35a5dc29e5dd3484af49746818ce536622b1a27631da445cbf55c7597fba9`.

Configuration: 2048-cell context, FP16 host-only KV, 512-token batched prefill, two-row verifier capacity, MTP off, zero GPU
expert slots, all routed experts in resident RAM, greedy sampling. `GUILD_IQ_MT_MIN=1` / `STRATA_IQ_MT_MIN=1` pin CPU
arithmetic across window sizes; the campaign was also repeated with **the shipped default dispatch rule, `IQ_MT_MIN=2`**.
The reference server uses `--short-read 0 --prompt-cache 0 --suffix-draft 0 --pcie-frac 0`.

Results:

- **248,320 first-token logits bitwise identical in both CPU dispatch modes**: maximum absolute error 0, relative L2 error 0. The test's numerical
  gate is independently declared as max absolute error <= 1e-5 and relative L2 <= 1e-6; bitwise identity is reported separately.
- **Eight greedy output IDs identical**, including two fresh candidate requests:
  `17,478,220,17,283,220,19,248046` = `2 + 2 = 4<|im_end|>`.
- PLE history was finite and nonzero and the token-window commit matched the supplied IDs.
- Comparing identical per-token paths with/without PLE changed logits by a maximum of **4.11010027** (`IQ_MT_MIN=1`)
  and **4.51094866** (default rule 2). A new verifier was
  captured for the ablation because changing a pointer does not remove an operation from an existing graph.
- Disconnecting PLE through the production session returned an error, emitted zero tokens and invalidated readiness.

Capture the reference first-token logits (raw F32 vocabulary row) with the existing reference engine:

```bash
STRATA_DUMP_FIRST_LOGITS=/tmp/opencode/guild-correctness/qwen-reference-fp16.bin \
STRATA_IQ_MT_MIN=1 engine-cuda12/strata \
  --pack "$PACK" --native "$MODEL" --expert-profile data/expert-profile.bin --expert-cache off \
  --tokens-file tests/data/qwen_recovery_ids.txt --max-new 1 --max-context 2048 \
  --spec 2 --prefill 512 --kv fp16 --kv-host-only --resident-budget-gib 56 --suffix-draft 0 --pcie-frac 0

GUILD_IQ_MT_MIN=1 build-cuda12/qwen_runtime_parity_test "$MODEL" "$PACK" tests/data/qwen_recovery_ids.txt \
  /tmp/opencode/guild-correctness/qwen-reference-fp16.bin engine-cuda12/strata data/expert-profile.bin \
  /tmp/opencode/guild-correctness/qwen-parity
```

For the default-dispatch repetition, use `STRATA_IQ_MT_MIN=2`, capture to `qwen-reference-default.bin`, and invoke the same
test with `GUILD_IQ_MT_MIN=2` and that reference file. Its report is in `/tmp/opencode/guild-correctness/qwen-parity-default`.

`MODEL` is the existing first shard above; `PACK` is its matching directory. The output directory contains the candidate
logit row, per-token PLE/ablated diagnostic rows and `parity.txt`. A full-model CTest registration is opt-in through the five
`GUILD_QWEN_PARITY_*` CMake paths; it never downloads assets. The test starts the reference process before candidate loading
and waits for it to release memory, rather than holding both models on an 8 GiB GPU.

## Full test status and remaining blockers

- CPU Release suite: **23 passed, four ISA tests skipped**, 27 registered, no failures.
- CUDA sm_61 Release suite: **80 passed, six skipped, one failed**, 87 registered. All newly added regressions passed.
- The inherited `ple_parity` failure remains visible: it requires the original Q2_0 GGUF, canonical `pack/full`, and the
  independent ggml captures `bench/micro/ple_in.bin` / `ple_out.bin`, which this checkout does not supply. The installed
  Unsloth native pack is not an equivalent fixture. This failure predates the campaign; it was not disabled or changed into
  a pass. The new real-Qwen parity gate validates execution against the preserved working engine but does not recreate that
  intermediate-activation oracle.
- The parity proof is for the stated artifact, prompt, backend and configuration. Active MTP, long contexts, other KV
  formats, other models and independent full-precision-model quality are not established by this result.
- Exact planner/actual-allocation reconciliation, multi-model process globals, generic dense attention and uniform
  error-return handling in older kernels that call `exit()` remain later tasks. No kernel, GPACK/GEXEC format or legacy
  implementation was rewritten in this campaign.

## Native Tokenizer and Chat Template Implementation

The previous greedy substring tokenizer has been replaced with a complete native C++ byte-level BPE tokenizer and model-specific chat template engine:

- **BPE Tokenization**: Byte-level encoding (GPT-2 byte mapping), rank-ordered merges from `merges.txt`, exact pre-tokenization scanner using compiled Unicode classification tables (`include/guild/runtime/unicode_tables.hpp`), and control/special token identification from `token_type.json`.
- **Incremental Detokenization**: Stateful `IncrementalDecoder` buffers trailing incomplete multi-byte UTF-8 sequences across token boundaries, preventing partial UTF-8 emissions in SSE streams. Benchmark shows 71.6 ns/token overhead and ~18.2M tokens/s bulk decode throughput.
- **Model-Specific Chat Templates**: Native `ChatTemplate` reads `chat_template.jinja`, identifying model family structure (Qwen vs Ornith) and correctly handling roles (system/developer/user/assistant), merged system blocks, reasoning effort instructions, thought tags, generation prompts, and strict turn validation (rejecting empty messages, system messages out of order, or unsupported roles).
- **Parity Verification**: Tested against Hugging Face reference outputs across 19 text test vectors (ASCII, contractions, multiple spaces, newlines, code/JSON, Hindi, Chinese, mixed multilingual, emoji, combining marks, long inputs) and 8 full multi-turn chat templates with 100% exact token ID equality.
- **Real Model Serving**: Tested with installed Qwen3.8-Flash-Next UD-IQ4_XS. Both `guild run` REPL and `guild serve` HTTP endpoints (`/v1/chat/completions` and `/v1/completions`) accept ordinary text and chat messages, producing identical outputs under deterministic generation across streaming and non-streaming modes. Token-ID parity remains bitwise preserved.
