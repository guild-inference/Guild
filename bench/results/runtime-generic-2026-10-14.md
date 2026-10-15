# Generic runtime performance: controlled causes and implementation

Measured on 2026-10-14 on Ubuntu, two Xeon Gold 6130 CPUs (32 physical cores),
376 GiB reported RAM, one GTX 1070 8 GiB, NVIDIA driver 580.173.02, CUDA 12,
sm_61 Release builds. These measurements describe this hardware, not other GPUs.

## Corrected attribution

The earlier comparison did not isolate cache policy. Strata converts a numeric
cache request into a largest-blob byte allowance, then fits variable-sized slots
inside it. Its `--expert-cache 48` actually produced **67 slots**. Guild had used
48 uniform, largest-sized slots. Automatic PCIe probing also selected different
fixed-point fractions in different starts.

With **67 actual slots, the same profile and the explicit fraction 0.218**, old
Strata and Guild produced all six identical 256-token outputs:

| Matched execution | Prose | Coding |
| --- | ---: | ---: |
| Strata, static variable-sized cache | 19.288 tok/s | 24.620 tok/s |
| Guild, 67 uniform slots before generic changes | 19.356 tok/s | 24.609 tok/s |

The coding difference is about 0.05%. A controlled Strata run with adaptation
enabled measured 19.216 / 24.408 tok/s, versus 19.288 / 24.620 with it disabled.
**These runs do not establish an adaptive-cache speed advantage.** The former
claim that adaptation explained the remaining gap is superseded by this control.

The confirmed losses were disabled/mis-scaled PCIe offload, scalar finite checks,
and full-logit CPU readbacks/allocations. Cache count, actual byte size, offload
fraction, and MTP acceptance must be matched before attributing a kernel regression.

## Measured automatic policy

Each row is a median of three fresh, uncached 256-token generations after a
16-token warmup. Frozen prompt IDs match the earlier experiment. Thinking is off
and sampling is greedy. Qwen retained **262144-token FP16 host-only KV capacity
and MTP with a four-token verifier**. Ornith used 4096-token capacity, FP16
host-only KV, and no MTP.

| Model/configuration | Prose | Coding |
| --- | ---: | ---: |
| Qwen3.8-Flash-Next UD-IQ4_XS, Guild before fixes | 17.512 tok/s | 21.826 tok/s |
| Qwen, new generic automatic policy | **20.251 tok/s** | **24.485 tok/s** |
| Ornith-1.5-35B Q4_K_M, CPU-routed-expert control | 39.252 tok/s | 38.984 tok/s |
| Ornith, new generic automatic policy | **39.259 tok/s** | **39.077 tok/s** |

Qwen gains 15.6% / 12.2% over the earlier Guild build on these prompts. The selected
Qwen split ranged from 26/256 to 39/256 across its three supported format pairs;
complete-window calibration measured 136.25 ms with CPU experts and selected
126.87 ms. Automatic cache sizing chose zero slots after accounting for fixed
allocations, captured graphs, and the reserve. No model-name condition made that
choice.

Ornith's grouped expert down formats (Q4_K and Q6_K) are not supported by the
existing grouped GPU-expert kernels. The policy correctly left those experts on
the CPU. Dense operations and attention still use the GPU. All six 256-token
Ornith outputs exactly matched its CPU-expert control. This is a capability-based
fallback, not a claim of a new GPU kernel for these formats.

These short-prompt tests retain the stated context capacities; they do not measure
a completely filled 262144-token prompt. Startup calibration is measured on short
scratch windows and does not guarantee an optimal split for every later workload.

### The automatic split changes output, measurably

Automatic calibration is a timing measurement, so its selected split can differ
between starts. On this PC, restarting with identical weights, prompts, and flags
selected 26..39/256 in one start and 25..45/256 in the next. Because a GPU-computed
expert and a CPU-computed one round differently, that difference changes generated
text. Measured on the same frozen prompts with greedy sampling:

| Restart with identical configuration | Prose | Coding |
| --- | ---: | ---: |
| Tokens identical to the first start's 256 | 208/256 | 233/256 |
| First divergent token | 208 | 233 |

Both outputs remain sound: the divergence appears late and then persists because
greedy decoding continues from a different token. This is the same known property
the GPU expert cache already has (`bench/results/2026-09-27-cache-parity`: 95-98%
same top-1, equal perplexity). It is not new numerical instability.

Concretely, that means:

- The **fully matched CPU-expert control is bitwise reproducible**, which is what
  the parity gate asserts (all 248320 logits bitwise, zero error).
- The **automatic split is not**. Use explicit `--pcie-frac <f>` for reproducible
  generation; automatic is for throughput and picks its own value.
- Automatic cache sizing is deterministic given the same free VRAM; the factor
  that varies is the measured CPU-vs-GPU-plus-copy timing per format pair.

Throughput is unaffected by which split is chosen: the verification run measured
19.906 / 24.206 tok/s against the published 20.251 / 24.485, within normal variation.
Its draft acceptance and RAM backing (59,519,795,200 bytes) were identical.

## Generic implementation

### Expert policy and sizing

`include/guild/runtime/performance.hpp` and `src/runtime/performance.cpp` implement:

- Per-layer support checks using actual quantization types and geometry, through
  the backend's `native_expert_supported` contract.
- Native-format timing with the actual expert source, CPU worker pool, GPU
  kernels, and host-to-device copies. Unsupported/unmapped layers remain on CPU.
- Complete-verification-window trials against a CPU-expert baseline. The trials
  emit no tokens, reset session state, and select a measured improvement rather
  than a Qwen-specific bandwidth constant. Explicit `--pcie-frac` values bypass
  tuning and retain the dispatcher's fixed denominator of 256.
- Exact counts for explicit cache requests. Automatic caches use each ranked
  pair's actual aligned byte size and live free VRAM after weights, KV, prompt
  scratch, verifier, draft/head, and verifier graph allocations.
- A partial cache requires a validated ranking. A complete GPU-capable tier can
  be populated without a profile when it fits. Cached slots are copied and
  verified before publishing residency. Captured graphs are reset before a late
  cache binding; arena addresses used by MTP remain stable.
- RAM backing budgets derived from the plan and actual expert layout, removing
  the runtime's former forced 56 GiB minimum. File-tier cache population uses one
  reusable blob staging buffer instead of retaining thousands of transient blobs.

`ExpertDispatch::pcie_layer_numerators` makes the policy apply independently to
each supported layer. A single unsupported layer no longer disables offload for
all other layers. The canonical fixed-layout pack retains its existing backend
geometry contract and format default.

### Full-distribution validation without full readback

`finite_first_bad` checks **every F32 logit**, including unselected vocabulary
entries, on the device. It reports the first NaN/Inf index through an eight-byte
status word captured in the existing verifier stream. No extra synchronization
or complete vocabulary allocation/readback is required on the normal path.
The same validation covers single-sequence and batched verifier paths.

The matched 67-slot experiment compared this against
`GUILD_LOGITS_VALIDATION_HOST=1`, which reuses a host buffer and checks the full
distribution. All six 256-token outputs matched:

| Validation path, same binary/configuration | Prose | Coding |
| --- | ---: | ---: |
| Full vocabulary readback to reused host buffer | 18.938 tok/s | 24.027 tok/s |
| Device check and compact status | 19.183 tok/s | 24.393 tok/s |

The coding request read back **295004160 bytes** on the host path versus **680
bytes** for 85 device-validated windows. Host validation work fell from about
1.3 ms/window to 0.002 ms/window. Device scan cost remains part of GPU execution;
these timings do not subtract that cost or imply the same gain on another GPU.

CPU expert-result validation remains mandatory. GPU-owned mapped host rows are
excluded from the CPU scan because GPU copy/combine kernels own those rows; the
complete final distribution is still checked.

### Per-instance state, controls and reporting

- Commit/draft overlap is a verifier-instance setting; initializing one model
  does not change another verifier's setting. Successful requests and resets
  drain pending commits. The existing synchronous control remains available.
- `--pool-workers` and `--spec-min-p` are carried through planner/runtime settings
  instead of silently discarded. Confidence gating also applies with sampling;
  draft sampling receives the request's actual sampler parameters.
- Cache/RAM/file/offload counters are request deltas, excluding startup calibration
  and preceding requests. Fully GPU-resident graphs report routed hits even when
  they bypass the CPU dispatcher. The engine pipe exports these counters.
- Startup diagnostics record measured costs, chosen split, actual cache bytes,
  CPU workers, MTP state/confidence, and resident expert backing bytes.
- Default geometry now consistently describes its indexed-sparse layout. The KV
  staging helper takes attention mode and optional context length into account,
  matching actual dense/sparse allocations. Tests cover 64, 4096, and 262144-cell
  dense capacities plus full-context sparse FP16/INT8/Q4 attention and graph replay.

## Validation and reproducibility

- Rebuilt the CUDA suite: **86 passed, six hardware-dependent skips, zero failures**
  out of 92 registered tests run with `ple_parity` excluded for its absent fixtures.
- The remaining KV-staging failure found during this campaign was fixed and the
  complete run repeated successfully. Final load/failure/performance checks also
  passed after the last diagnostic and bounded-fallback cleanup.
- The final real-Qwen gate matched **all 248320 first-token logits bitwise**, with
  maximum absolute error 0 and relative L2 error 0, plus repeated greedy outputs
  and PLE state/failure checks. This parity control uses CPU-routed experts,
  2048-token capacity, a two-row verifier and MTP disabled.
- The device-validation and host-validation controls matched all six 256-token
  outputs; the normalized Strata/Guild cache controls matched all six outputs;
  Ornith automatic/CPU-expert controls matched all six outputs.
- A second automatic-policy start with otherwise identical settings reproduced
  208/256 and 233/256 tokens on the two prompts. This is the documented split
  sensitivity above, not a correctness failure; see
  "The automatic split changes output, measurably".

The generic policy tests cover multiple dimensions/formats, unsupported GPU
formats, exact slot limits, variable byte budgets, unmapped sources, and explicit
fractions independent of expert count. CUDA runtime fixtures cover late automatic
cache binding, CPU fallback, per-instance commit settings, honored controls,
rejected post-capture rebinding, and failures emitting no tokens.

`tools/bench_runtime.py` is the reusable runner. It accepts executable, model,
pack, tokenizer, profile, MTP, context, cache, worker, confidence, and offload
settings. A JSON prompt list accepts explicit token IDs for exact comparisons.
Reports contain executable hashes, frozen IDs, effective command, selected
performance environment variables, per-request counters, timings, and stderr logs.

Example on this PC, from `/home/ubuntu/Guild` with the server stopped:

```bash
.venv/bin/python tools/bench_runtime.py \
  --exe build-cuda12/guild-generate \
  --model /mnt/models-ssd/Strata-data/models/unsloth-UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf \
  --pack /mnt/models-ssd/Strata-data/packs/unsloth-ud-iq4_xs \
  --profile data/expert-profile.bin --mtp /mnt/models-ssd/Strata-data/mtp/rt \
  --context 262144 --resident-budget-gib 56 --out scratch/qwen-policy-repeat.json
```

Raw measurements: `scratch/qwen-generic-*.json`, `scratch/ornith-generic-*.json`
and their `.log` files. `scratch/check_generic_runtime_results.py` checks frozen
prompts and output-ID comparisons.Test logs are
`scratch/generic-runtime-ctest.log` / `.xml`; the real-Qwen parity gate has its
artifacts in `scratch/generic-runtime-qwen-parity/`. The legacy `ple_parity` CTest requires absent external
fixtures and is excluded from the otherwise complete suite.

The local Qwen launcher now requests automatic expert caching and automatic
offload, retaining full FP16 KV and MTP. The server is left stopped:

```bash
sh /home/ubuntu/Guild/run-qwen-256k-fp16.sh --host 192.168.4.22 --api-key 'REPLACE_WITH_YOUR_SECRET_KEY'
```
