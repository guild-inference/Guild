# Phase 2 continuation: paired regression investigation

The preserved executable still hashes to
`71dca3597ba64f135517e0d32f5dd585171d4913b18a1919a537f926c2decc7c`.
The original separate-run medians are historical measurements, not a paired
estimate of a software regression.

## Method and reproducibility

`tools/bench_paired_plugins.py` runs fresh processes serially, one full-shaped
warmup followed by one measured repetition. Ten counterbalanced rounds yield ten
samples per configuration. Forward/reverse ordering rotates between round pairs.
Model, prompt construction, context 32768, batch 256, 128 generated tokens,
host-only KV, expert RAM backing and VRAM reserve match the earlier experiment.
Binary and plugin hashes, inherited CUDA/OpenMP settings, commands, UTC times,
CPU governor/frequency snapshots, GPU clocks/temperature/power and raw CSVs are
recorded under `bench/results/phase2-continuation/`. No build runs concurrently.

Reported deltas are the **median of within-round ratios**, not ratios of separate
medians. Deterministic paired bootstrap intervals resample the ten round pairs.
The intervals describe this finite experiment; they do not establish universal
equivalence or eliminate shared-machine/time effects. Frequency snapshots are
before/after samples, not continuous per-core effective-frequency measurements.

```sh
python3 tools/bench_paired_plugins.py --prompts 512 4096 --rounds 10 \
  --output bench/results/phase2-continuation/paired-initial
CUDA_MODULE_LOADING=EAGER python3 tools/bench_paired_plugins.py \
  --cases preserved builtin runner all --prompts 512 --rounds 10 \
  --output bench/results/phase2-continuation/paired-eager
```

The initial experiment measures preserved, rebuilt built-in, CPU-only override,
CUDA-only override, runner-only override and all overrides. The long initial
command hit the shell timeout before its last sample and resumed from receipts;
it did not silently count the interrupted process as a result.

## Findings before the fix

At 512 and 4096 tokens, rebuilt built-in prefill differed from preserved by
-0.20% [-1.26%, +0.29%] and -0.68% [-1.41%, +0.10%] respectively. The initial
1–2.5% separate-run drop is not reproduced as a statistically isolated built-in
regression. The preserved binary itself moved from 240.9 / 221.9 tok/s in the
earlier experiment to 237.0 / 212.4 tok/s in these runs; comparison to stale
absolute numbers would misattribute system/run-order drift to source changes.

Runner-only decode **did** have a repeatable difference: -1.80% versus built-in
at 512 tokens and -1.45% at 4096 tokens. Package validation discovered the GPU
before `GuildModel::load` set `CUDA_MODULE_LOADING=EAGER`. The runtime had already
initialized using its default loading policy. Plugin processes showed about
20 MiB less device residency than the built-in process despite identical tensor,
KV and scratch allocations. This affects runtime/module initialization, not the
runner's per-token function dispatch (there is no runner lookup in that loop).

Setting EAGER in the process environment before startup restored the device
footprint and reduced the 512-token all-package decode delta versus built-in to
-0.08% [-0.24%, -0.00%]. It did not prove every smaller difference nonexistent;
runner-only decode still differed by -0.67% in that experiment.

## Fix and full-context paired verification

`platform::configure_device_runtime_policy()` now applies the default before
hardware discovery or package capability validation can initialize CUDA. An
explicit embedding application's `CUDA_MODULE_LOADING` choice is preserved.
Setting the variable late in model load has been removed. No kernel arithmetic,
clock lock, affinity rule, quantization or synchronization shortcut was changed.

```sh
python3 tools/bench_paired_plugins.py --cases preserved builtin all \
  --prompts 512 4096 8192 16384 --rounds 10 \
  --output bench/results/phase2-continuation/paired-policy-fix
```

Median throughput after the fix (prefill / decode, tok/s):

| Prompt | Preserved | Rebuilt built-in | All external overrides |
|---:|---:|---:|---:|
| 512 | 236.632 / 43.245 | 237.314 / 43.254 | 237.268 / 43.208 |
| 4096 | 213.054 / 32.937 | 213.360 / 32.972 | 212.398 / 32.875 |
| 8192 | 195.298 / 26.072 | 195.273 / 25.988 | 195.368 / 25.996 |
| 16384 | 170.078 / 17.974 | 169.782 / 17.882 | 170.059 / 17.934 |

All-package paired prefill deltas versus preserved are -0.01%, -0.69%, +0.15%,
and -0.09%; all four bootstrap intervals include zero. Decode deltas are -0.19%,
-0.31%, -0.11%, and +0.07%, also with intervals including zero. At 4096 tokens,
all-package prefill remains -0.57% versus rebuilt built-in, with interval
[-1.08%, -0.003%]. Keep that smaller remaining comparison open rather than
claiming exact performance equality. The previous large initialization-related
decode loss is materially reduced; no performance improvement is claimed.

`tools/audit_binary_codegen.py` records disassembly, instruction/opcode counts
and normalized fingerprints for native and SIMD arithmetic and the worker drain.
The inspected native quantizers and K-quant/AVX-512 arithmetic retain their
instruction counts; the worker drain changed (1290 to 1287 instructions). Raw
assembly remains available to distinguish relocation changes from code changes.
This is targeted code inspection, not evidence that every generated function is
byte-identical. There was no experimental LTO, visibility or ISA-flag change.

After moving policy setup, modular ABI tests and the combined independently
built CPU/CUDA/runner real-model test passed full-vocabulary first-token logits,
eight greedy IDs, recurrent-state and tested KV-prefix parity. Further extraction
must repeat these gates; this investigation does not complete Phase 2.
