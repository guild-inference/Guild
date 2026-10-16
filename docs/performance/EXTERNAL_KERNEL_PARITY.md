# Independent migration-slice verification

Status: partial Phase 2 validation, not complete backend extraction or a passed
no-regression gate. Raw CSVs are in `bench/results/external-kernels/`.

## Configuration

Dual Xeon Gold 6130 (32 physical cores, 64 threads; AVX-512F/BW/DQ/VL, no VNNI),
GTX 1070 8 GiB SM 6.1, CUDA 12.0, driver 580.178.04; Ornith-1.5-35B Q4_K_M;
context capacity 32768, prefill batch 256, 128 generated tokens. Each case used
one full-shaped warmup and ten repetitions, serially, with no concurrent build
or inference workload. The 512 MiB VRAM reserve and host-only KV were retained.

The preserved Phase 1 binary is from `42018db`; SHA-256 is
`71dca3597ba64f135517e0d32f5dd585171d4913b18a1919a537f926c2decc7c`.
The rebuilt Core supports the new contracts at `4b4ded0`. External execution
means native CPU experts + CUDA router + Qwen description runner, **not** all
CUDA kernels. Other CUDA arithmetic remains built-in in both rebuilt cases.

## Ten-run medians (prefill / decode, tok/s)

| Prompt | Preserved Phase 1 | Rebuilt built-in fallback | External slices together |
|---:|---:|---:|---:|
| 512 | 240.874 / 43.553 | 237.499 / 43.574 | 238.354 / 43.357 |
| 4096 | 221.878 / 32.997 | 216.379 / 33.217 | 217.980 / 33.039 |
| 8192 | 200.260 / 26.191 | 197.576 / 26.184 | 196.940 / 25.955 |
| 16384 | 173.265 / 18.102 | 170.342 / 17.913 | 171.151 / 17.854 |

External-slice prefill is approximately 1.0–1.8% below the preserved binary;
decode ranges from about +0.1% to -1.4%. The rebuilt fallback also shows a
prefill drop (about 1.3–2.5%). These are separate serial measurements, not a
paired statistical attribution. The data does not justify an improvement claim
or establish that the new dispatch accounts for all of the difference. A paired
re-run and profiling of the bound CPU calls remain open performance gates.

## Correctness and lifecycle evidence

- Core Release build: 89 CTest passes, six existing skips, with `ple_parity`
  excluded because its required external GGUF fixture is absent. The excluded
  test is not counted as passing.
- Independent runner: C99 header compilation and C++ model-reader test passed.
- Independent CPU: GGML-reference activation and single-expert arithmetic test
  passed. SIMD translation units are isolated from baseline dispatch.
- Independent CUDA: real-GPU router test passed on SM 6.1.
- Hosted CPU and runner test workflows, and the hosted CUDA compile workflow,
  passed at the published repository commits recorded in the verification receipt.
- Real Ornith: all first-token vocabulary logits, eight greedy IDs, all captured
  recurrent-state bytes and the tested first 512-token FP16 host KV prefixes
  matched bit-for-bit across separate built-in/external processes.
- External CPU and CUDA calls were verified by bound package IDs and DSO
  counters. Active runner leases reject disable while a model is alive.
- Real Ornith passed with built-in Qwen registration and native CPU fallback
  disabled. This does not prove an external-only CUDA backend.
- Independently built CPU 1.0.0 -> 1.0.1 -> rollback 1.0.0 passed real-model
  comparisons and version/counter checks; Core's executable SHA-256 was unchanged.

The first independent CPU test failed because the DSO's private GGML conversion
tables were uninitialized. Core's initialization cannot initialize another DSO's
private tables. Calling `ggml_cpu_init()` before publishing callbacks fixed the
single-expert mismatch and the real-model comparison. The earlier failure was
not accepted as numerical tolerance or hidden by fallback.

Qwen4Exp external inference, full CUDA external-only execution, all concurrent
replacement races, multi-device unload, production-signed hosted releases and
Windows/HIP integration remain unverified. See `../architecture/EXTERNAL_BACKENDS.md`.
