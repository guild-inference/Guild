# Qwen decode regression: Strata versus Guild

Follow-up: [controlled generic runtime results](runtime-generic-2026-10-14.md)
correct the cache-policy attribution below. Matching actual cache counts and the
offload fraction eliminates the residual gap; adaptation was not a measured win.

Measured 2026-10-14 on this PC: Ubuntu, two Xeon Gold 6130 CPUs (32 physical cores),
376 GiB reported RAM, GTX 1070 8 GiB, driver 580.173.02, CUDA 12 / sm_61 Release build.
The preserved reference is `engine-cuda12/strata`, SHA-256
`d24277e73c13dce297e659110d216679676a999ffb6335c2371ec866af4ee90b`.

Both engines used the same installed Qwen3.8-Flash-Next UD-IQ4_XS shards, native pack,
expert profile, and MTP draft. Every performance run retained 262144-token FP16
host-only KV capacity, 512-token prefill chunks, a four-token verification window,
and a 56 GiB expert RAM budget. Prompt IDs were frozen and identical between engines.
Thinking was disabled and sampling was greedy. Each main result is the median of
three fresh, uncached 256-token generations after a 16-token warmup.

## Results

| Engine/configuration | Prose prompt (35 tokens) | Coding prompt (40 tokens) |
| --- | ---: | ---: |
| Old Strata, automatic/adaptive GPU cache and PCIe offload | 20.02 tok/s | 25.12 tok/s |
| Guild before fixes, no GPU cache and no PCIe offload | 17.51 tok/s | 21.83 tok/s |
| Guild: asynchronous commits restored, offload still disabled | 17.53 tok/s | 21.94 tok/s |
| Guild: PCIe offload restored, original scalar finite scans | 18.73 tok/s | 23.37 tok/s |
| Guild: PCIe offload and optimized finite scans, no GPU cache | **19.38 tok/s** | **23.95 tok/s** |
| Guild: same fixes with a static 48-slot GPU cache | 19.02 tok/s | 22.97 tok/s |

The recommended configuration keeps zero persistent GPU expert slots on this PC.
It gains 10.7% on the prose fixture and 9.7% on the coding fixture over the previous
Guild build. The remaining difference from Strata's default configuration is 3.2%
and 4.7% respectively; this is not a claim of identical default-cache performance.

MTP acceptance varies with the text. The fixed default configuration produced about
2.13 tokens per verification window on prose and 3.01 on code. Thus the coding rate
does not describe every request.

## Causes and fixes

### PCIe expert offload was tied to persistent GPU cache slots

`src/runtime/model.cpp` enabled RAM-to-GPU expert execution only when
`plan.routed_experts_in_gpu > 0`. The 8 GiB planner and this PC's launch configuration
selected zero slots, so the GPU processed no routed experts over PCIe, despite the
whole expert complement being in mapped, pinned RAM.

The old driver used a bandwidth-scaled PCIe share independently of cache occupancy.
It also used a fixed denominator of 256. The extracted initializer incorrectly
multiplied the fraction by the model's expert count (512).

Guild now probes the link before allocation, enables supported pinned-RAM offload
independently of cache slots, and uses the dispatcher's correct 256 denominator.
The measured probe was about 7.9 GB/s, selecting 56/256. `--pcie-frac 0` explicitly
disables it for matched CPU-only controls and numerical parity checks.

### Finite-value checks became a scalar decode-path cost

The correctness-recovery code checks CPU expert inputs/results and every head logit.
`include/guild/core/validation.hpp` scanned them using a scalar early-return loop.
The checks remain mandatory, including logits the sampler would not select. They
now use vectorizable 64-value exponent-bit reductions and locate the first failing
element only when a block contains NaN or infinity.

`src/runtime/model_impl.hpp` also checked stale host rows owned by GPU experts.
Those rows are zeroed by the captured GPU copy/combine path, not written by the CPU
pool. Validation now checks the CPU-owned result rows; full-logit validation still
checks the final GPU computation. A test injects a stale NaN into a GPU-owned host
row and verifies that the healthy CPU result is accepted.

### Commit/draft overlap was lost during extraction

The old single-device driver enabled `Verifier::set_commit_async(true)`. The new
runtime omitted it. This is restored with waits before state resets and successful
request completion. `GUILD_COMMIT_SYNC=1` retains the synchronous control. Its
isolated throughput improvement on this PC was small, as the table shows.

### The old small GPU cache is adaptive

Strata auto-sized about 58 slots and exchanged experts as requests ran. Guild's
current cache is static. Enabling 48 static slots slowed both fixtures, so it is not
enabled in `guild-qwen-256k-fp16.json`. The old and new default configurations therefore
still have different GPU cache policies.

## Correctness checks

- Eight targeted CTests passed: planner, finite outputs, load failures, dispatch,
  inference failures, prefill failures, CLI behavior, and CLI failures.
- Fully matched CPU-only execution with MTP enabled produced **all 256 identical
  generated token IDs on each of the two performance prompts**.
- Matched CPU-only Strata medians: 18.52 / 23.17 tok/s. Fixed Guild's single parity
  runs: 18.32 / 23.13 tok/s, about 1% / 0.2% apart.
- The finite-check optimization preserved all six 256-token offloaded outputs.
- The real-Qwen parity gate compared **all 248320 first-token logits bitwise**:
  maximum absolute error 0, relative L2 error 0. This gate uses 2048-token capacity,
  a two-row verifier, CPU-only routed experts, and MTP disabled.
- The same gate matched repeated greedy continuations, required PLE state, and
  rejection of disconnected PLE. GPU offload without a persistent cache, pending
  asynchronous commits, reset, and subsequent generation were exercised by the
  CUDA fixture test. NaN/Inf rejection remains covered.

No full 262144-token prompt was generated for this regression campaign; capacity
remained fully allocated/configured throughout the performance runs.

## Reproduce

From `/home/ubuntu/Guild`, with the API server stopped so the GPU is free:

```bash
.venv/bin/python scratch/qwen_regression_bench.py --exe engine-cuda12/strata --legacy --label strata-old-default --expert-cache auto --out scratch/qwen-regression-strata-repeat.json
.venv/bin/python scratch/qwen_regression_bench.py --exe build-cuda12/guild-generate --label guild-fixed --out scratch/qwen-regression-guild-repeat.json
```

For the fully matched CPU-only control, pass `--expert-cache off --pcie-frac 0` to
Strata and `--expert-cache 0 --pcie-frac 0` to Guild. Strata interprets numeric cache
size zero as auto-sizing; the literal `off` is required to disable its cache.

Raw results and diagnostic logs are `scratch/qwen-regression-*.json` / `.log`.
`scratch/summarize_qwen_regression.py` checks token-ID comparisons. Full vocabulary
parity results are in `scratch/qwen-regression-parity/parity.txt`.

The rebuilt binaries are `build-cuda12/guild-generate` and `build-cuda12/guild`.
The existing launcher uses the rebuilt engine and automatic PCIe offload:

```bash
sh /home/ubuntu/Guild/run-qwen-256k-fp16.sh --host 192.168.4.22 --api-key 'REPLACE_WITH_YOUR_SECRET_KEY'
```

The benchmark processes exit after testing; the server was left stopped.
