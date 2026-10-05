# KV and expert tiers on a RAM-rich, small-VRAM PC

The engine can keep the main model's full KV in pinned host RAM with `--kv-host-only`. Its attention operations
stage only their selected blocks into one shared GPU buffer per session. FP16, INT8 and Q4_0 work. K8V4 still
requires resident KV; it is rejected with either host-streaming option. WSL's pinned-memory limit still applies.

Example additions to an existing single-GPU engine command:

```text
--max-context 262144 --kv fp16 --kv-host-only --resident-budget-gib 55
```

Remove `--kv-resident` when using `--kv-host-only`: supplying both is an error, including an explicit
`--kv-resident 0`. The meaning of `--kv-resident 0` has not changed: the KV is fully in VRAM.
`--expert-cache off` explicitly selects zero persistent GPU expert slots. Existing `auto` and the legacy
`--expert-cache 0` alias with a profile remain unchanged.

Setup accepts `--context 262144 --kv fp16`, keeps that precision in the config, and offers FP16 interactively
above 8K. The ordinary defaults remain FP16 through 8K and INT8 above it. Start scripts launch the saved config;
they do not substitute another precision.

## Root causes and fixes

- Setup excluded FP16 from long-context selection and config recovery, classified every non-INT8 choice as Q4,
  and priced every non-Q4 choice as INT8. One format table now drives labels, selection and byte estimates.
- RAM complement planning required an allocated GPU cache, and the verifier required its device arena even
  for all misses. Empty-tier planning and an independent residency table now let RAM/file experts execute
  without persistent GPU expert storage. Partial profiles no longer leave spare RAM budgets unused.
- KV streaming assumed persistent per-layer residency; prefill also staged a whole context-sized layer.
  Transient shared staging reuses the resolver and attention kernels, with bounded query tiles for prefill.
- Cache/MTP planning could spend the memory needed to process even the smallest prompt. Host-heavy plans
  reserve the actual minimum prompt layout, and setup's automatic MTP can release its allocations when needed.
- Download percentages used exact bytes while both displayed values rounded to tiny decimal-GB numbers.
  Shared adaptive units now keep small files readable.

## Storage costs

Each main-model QSA cell holds two KV heads, each with 256 values for K and V. These are storage bytes, not just
the number of quantization bits:

| Format | Setup label | Bytes per main-layer cell | Bytes per MTP cell | Full KV at 262144 cells, 12 main layers + MTP |
| --- | --- | ---: | ---: | ---: |
| `fp16` | FP16 | 2048 | 2048 | 6.500 GiB |
| `int8` | 8-bit | 1056 | 1056 | 3.352 GiB |
| `q4_0` | 4-bit (Hadamard-rotated) | 576 | 576 | 1.828 GiB |
| `k8v4` | INT8 K + 4-bit V | 816 | 1056 | 2.648 GiB |

FP16 is `2 heads × K/V × 256 × 2 bytes`. INT8 adds one FP16 scale per 64 codes: 1024 code bytes + 32 scale
bytes. Q4_0 uses 18 bytes per 32 values, or 144 bytes per head/side. K8V4 combines a 528-byte INT8 K half and a
288-byte Q4_0 V half. The MTP layer uses plain INT8 under K8V4.

These follow `kv_pool_bytes` in `src/core/layer.cpp`, `kv_block_bytes` in `src/kernels/cuda/kv_stream.cu`, and
`kv_q8.hpp` / `kv_q4.hpp`. Setup can run before there is a compiled engine or downloaded model, so it mirrors
these layouts in one `KV_FORMATS` table. Its helpers accept layer counts; the default 12 main + 1 draft matches
the supported Qwen3.8 geometry. Guard/alignment bytes and page rounding add a small overhead. KV estimates do
not include weights, recurrent state, indexer keys, RoPE or prompt buffers.

At 262K FP16, the full main KV costs 6.0 GiB. With MTP's full host copy, RAM costs 6.5 GiB, or 6.98 decimal GB.
If MTP is disabled, its copy is not allocated. The previously used INT8 formula understated FP16 by 3.38 GB.

| Main KV allocation at 262K FP16 | KV pool VRAM |
| --- | ---: |
| Fully resident (`--kv-resident 0`) | 6144 MiB |
| Existing streaming (`--kv-resident 32768`) | 768 MiB, across 12 layers |
| Host-only (`--kv-host-only`) | 32.125 MiB, shared by 12 layers |

Host-only is not zero-VRAM attention: the indexer retains its exact FP32 pooled keys (384 MiB at 262K across
12 layers), and shared RoPE takes 64 MiB. Page tables, resolver metadata, query/output scratch and recurrent
state also remain on the GPU. Batch slots have their own sessions and staging buffers.

## Host-only attention

The existing append kernels write the authoritative full host KV. The existing block resolver then gathers a
selection into GPU slots. Host-only uses the same resolver, but invalidates a layer's transient map before
each attention operation, preserving cumulative traffic counters. This is necessary because another layer
may have used the shared storage since the previous operation. No previous operation's residency is assumed.

The staging bound comes from the engine's selection and verification constants, not the context:

```text
8 maximum verify queries × (2051 / 4 + 2 pages) × 4 cells = 16448 staging cells
```

The extra pages cover the threshold and tail blocks. This is 32.125 MiB with FP16, 16.564 MiB with INT8 and
9.035 MiB with Q4_0. The primary QSA state owns the pool; other layers borrow it alongside the already shared
RoPE tables. Each layer owns its full host KV and its map. Attention operations are ordered on their session's
stream, including captured graphs; independent sessions do not share staging.

The prompt path no longer allocates an identity-layout KV pool for an entire context in this mode. It resolves
tiles of at most eight queries, then calls the same prompt-attention kernel or its existing decode-attention
fallback. Only storage and query tiling change; selection, masking, quantization and attention arithmetic do not.
Conversation snapshots store the authoritative host KV and invalidate transient maps on restore.

MTP is optional and retains its separate bounded dense-attention ring when enabled. That draft layer is not
sparse: it reads its draft window, so it cannot borrow the much smaller sparse main-model staging pool. An
unbounded `--mtp-window 0` retains the existing independent, fully resident draft KV allocation. Host-only does
not change the draft window's semantics. Disabling MTP removes its weights, head, ring and host KV copy.

### PCIe tradeoff

A long-context single-token decode may stage roughly 48 MiB across 12 FP16 QSA layers when selections share no
pages. INT8/Q4 transfer less; a verify/prompt tile deduplicates overlapping selections. Unlike persistent
streaming, host-only cannot reuse blocks from an earlier attention operation. Its traffic competes with expert
and PLE transfers. It frees VRAM, but is not universally faster. Short prompts do not measure the long-context
transfer cost; run a workload-specific benchmark before choosing it for speed.

## An empty GPU expert tier is valid

`FileExpertSource::pin_cache_complement` treats an unopened zero-slot `ExpertCache` as an empty set. Its
complement is every expert. A nonempty cache must still be filled and synchronized before planning RAM residency.
RAM selects profile-ranked pairs first, then unranked pairs in stable layer/expert order. Partial profiles therefore
cannot prevent a large enough budget from holding the entire model.

There is no dummy GPU cache. A small all-miss residency table supplies the verifier and PCIe planner with the
model's expert layout. GPU-only hit planning and adaptive swaps are inactive. There are no GPU lend slots or
RAM/GPU exchange buffers. Prompt processing allocates its own buffers; CPU and temporary PCIe execution obtain
expert bytes from the RAM tier. Only experts outside that tier use the file fallback. Logs report actual resident
bytes and counts, file fallback counts and whether adaptive swaps are active.

The installed Unsloth UD-IQ4_XS tested here has 55.43 GiB of experts. A **55 GiB** budget is valid but does not
hold every expert when the GPU holds none. A budget at least as large as the expert bytes (and enough available
RAM after headroom) is required for zero steady-state expert file reads. The integration test uses 64 GiB; it
allocates only the 55.43 GiB actually needed, not the whole allowance. RAM-budget clamping/headroom rules remain.
Existing restrictions on resident budgets with layer splits/helper expert caches remain.

## Small-card planning

For an 8 GB-class card, setup keeps the requested precision and context. With sufficient RAM and a compatible
engine, it selects host-only KV at long contexts first. Otherwise resident streaming can use its legal 20480-cell
minimum. An unset draft vocabulary may use the smaller English/code subset; explicit or saved choices stay.
Prompt chunks start at 512 tokens. Setup explains these choices.

Setup marks automatic MTP with `--mtp-optional`. Before sizing the GPU expert tier, the engine compares measured
free VRAM with the real `Prefill::bytes_needed(..., 512)` layout, draft-head binding cost and the reserve. If those
do not fit, it releases MTP and serves through single-token verification. Explicit engine `--mtp DIR` without
`--mtp-optional` remains mandatory. `INFO` reports `spec=1`, no lookup drafts, when MTP is off.

With a RAM expert budget, setup prices that RAM tier plus headroom and full host KV, not all experts as if no
file tier existed. Automatic budgets are adjusted for host KV; an explicit budget is preserved with a warning
when it leaves too little headroom.

Host-heavy auto-cache sizing leaves room for the minimum own prompt buffers and verifier, even if that leaves
zero expert slots. Own prompt buffers are reserved **in addition to** the draft head and existing headroom:
a small nonzero cache may still be unable to lend buffers, and lazy GEMM/staging allocations need that headroom.
The existing prompt allocator still steps down legal chunks as needed. Setup refuses an
obviously impossible full-VRAM long-FP16 plan, without rewriting its precision/context. It probes the binary's
CLI capabilities and will not emit new flags for an older engine; build/update from these sources if required.
This is not a guarantee against another process allocating VRAM after planning, or every possible model/driver
allocation failure. Startup allocation failures remain explicit errors, not a supposedly ready server.

## Verification

On Linux, GTX 1070 8 GiB (Pascal/sm_61), CUDA 12, about 377 GiB system RAM, on 2026-10-05:

- `kv_host_only_test`: bitwise resident/host-only attention at 262144 cells for FP16, INT8 and Q4_0; one through
  eight queries, scattered full-width selections, two layers sharing staging, CUDA graph replay, and counters.
- `file_expert_source_test`: an actually unopened zero-slot cache, nonzero budgets, a valid partial profile,
  resident expert bytes, repeated RAM reads with zero file reads, and file fallback outside a smaller budget.
- Existing streaming, INT8, hybrid, conversation snapshot/validation, cache/memory and profile tests. Hybrid's
  tensor-core-only subcase is skipped on Pascal; its append/gather/decode checks still run.
- Setup tests cover 262K explicit and interactive FP16, labels/layouts, configs/start scripts, 8 GB planning,
  old engines, explicit choices, rejected impossible plans and adaptive download units. Existing golden configs
  remain byte-identical. The platform-mocked test harnesses do not invoke real ROCm builds or alter log paths.

Results:

- Full CUDA Release/sm_61 build: passed (`cmake --build build-cuda12 -j 8`).
- Setup discovery: 271 tests, OK, one skipped
  (`python -m unittest discover -s tools -p 'test_setup_*.py'`).
- Relevant engine subset: 10/10 passed. Full registered CTest suite: 48/49 passed. `ple_parity` cannot run because
  its required original Q2_0 shard and `bench/micro/ple_in.bin` / `ple_out.bin` oracle fixtures are absent;
  its fixture check still fails explicitly (`ctest --test-dir build-cuda12 --output-on-failure -j 1`).
- Server discovery: 268 tests, OK, seven skipped, in a temporary environment with the documented
  `jsonschema>=4.23,<5` dependency range. The system Python has 4.10.3 and no `referencing`, which caused four
  failures and one error in structured-output tests before using the temporary environment. No server code or
  installed Python packages were changed (`python -m unittest discover -s serve -p 'test_*.py'`).
- Python compilation and `git diff --check`: passed. No repository formatter/linter configuration was found.
- Real-model cases below: passed, including zero-cache RAM-backed PCIe expert execution.

Reproduce the real-model test without modifying its saved config:

```sh
python tools/test_host_heavy_engine.py --config strata-unsloth-ud-iq4_xs.json \
    --engine build-cuda12/strata --output /tmp/strata-host-heavy-test
```

It requires the installed model/tokenizer and enough RAM for all experts. It compares real greedy continuations
and main-model state fingerprints in resident 8K, host-only 8K and host-only 262K configurations; exercises
prompt prefill and rewind; checks decode RAM/file counters; and forces optional MTP off with extra headroom.
It uses zero persistent GPU expert slots and `--pcie-frac 0` to avoid changing CPU/GPU expert rounding between
parity runs. A separate `--pcie-frac 1` case checks RAM-backed temporary GPU expert execution with zero cache
slots and zero file expert reads; it does not require CPU/GPU expert arithmetic to be bitwise identical.
Another case uses setup's automatic cache, the exact 55 GiB budget, and optional MTP with the ordinary reserve;
it must reach READY and process prompts whether MTP fits or is disabled. A 55 GiB budget need not hold all
experts when a smaller GPU cache leaves more than 55 GiB in its complement, so this case allows file fallback.
These are short prompts at a 262K allocation capacity, not a full 262K-token prefill benchmark.
Tensor-core hardware and HIP host-only parity need separate hardware validation.

### Short-prompt measurements

The installed UD-IQ4_XS model, the same 132-token prompt and four greedy output tokens, FP16, 31 pool workers,
zero GPU expert slots, all 24,576 experts in 55.43 GiB of pinned RAM, MTP off:

| KV mode and allocation capacity | Free VRAM at READY | Prefill | Four-token decode |
| --- | ---: | ---: | ---: |
| Resident, 8K | 1958 MiB | 4780 ms | 275 ms |
| Host-only, 8K | 2116 MiB | 4788 ms | 278 ms |
| Host-only, 262K | 1630 MiB | 4789 ms | 279 ms |

These are single smoke-test runs, not averaged throughput benchmarks. Tokens and main-state fingerprints
matched across all three. Each four-token decode read 1,920 expert blobs from RAM and zero from files. Rewind
on the second request reused 125 prompt tokens and processed eight fresh tokens, also with matching output.
The separate all-PCIe expert run took 633 ms for four tokens and read zero expert blobs from files; this is why
host-only KV and GPU expert offload should not be sold as universal speed improvements.

The forced optional-MTP test used a 1500 MiB reserve: with MTP loaded, only 1480 MiB were free against the
2148 MiB prompt/draft-head/reserve requirement. Releasing MTP reached READY with 1620 MiB free, accepted the
same prompt, and matched the resident greedy output. The ordinary reserve can retain MTP when it fits.

The setup-equivalent automatic-cache case kept the installed CJK draft vocabulary, MTP, the 700 MiB ordinary
headroom, 262K FP16 and the exact 55 GiB RAM budget. Reserving 1347 MiB for own prompt buffers, draft head and
headroom yielded 55 native expert slots (129 MiB), 55.00 GiB in RAM and 136 file-fallback experts. It reached
READY with 396 MiB free and served both prompts, including rewind. The first four-token decode took 388 ms
and read seven expert blobs from files; the second read six. This budget is runnable, not an all-experts-in-RAM
claim. Before the reservation fix, the same case allocated 243 GPU slots and failed to bind the 213 MiB draft
head with only 178 MiB left after its own prompt buffers and verifier had allocated.

## Changed files

- Setup/accounting/progress/planning: `setup.py`.
- Expert tier and verifier: `include/guild/core/expert_source.hpp`, `src/core/expert_source.cpp`,
  `src/core/verify.cpp`, and the tier/planning paths in `src/program/generate.cpp`.
- KV staging/lifetime/prefill: `include/guild/core/layer.hpp`, `include/guild/kernels/kv_stream.hpp`,
  `src/core/layer.cpp`, `src/kernels/cuda/kv_stream.cu`, `src/prefill/prefill.cpp`, `src/core/mtp.cpp`,
  `src/core/conversation_snapshot.cpp`, and the CLI/serving paths in `src/program/generate.cpp`.
- Engine tests/build: `CMakeLists.txt`, `tests/core/kv_host_only_test.cpp`,
  `tests/core/file_expert_source_test.cpp`, `src/core/conversation_snapshot_test.cpp`,
  `src/kernels/kv_hybrid_parity.cpp`.
- Python tests: `tools/test_setup_kv.py`, `tools/test_setup_download_progress.py`,
  `tools/test_host_heavy_engine.py`; platform-mocked harness corrections in `tools/test_setup_amd.py`,
  `tools/test_setup_golden.py`, `tools/test_setup_unsloth.py`.
- Documentation: this file and `docs/DETAILS.md`.

## Suggested commits

1. `setup: expose and correctly account for fp16 KV` — format metadata, selection/reporting, RAM/VRAM helpers,
   config preservation and precision tests.
2. `setup: fix adaptive download progress units` — formatting helpers, downloader and boundary tests.
3. `experts: allow resident RAM tier without GPU cache` — complement planning, residency tables, verifier/PCIe
   integration, adaptive/lend no-ops, tier logging and zero-slot tests.
4. `kv: add host-only streaming mode` — shared staging, resolver invalidation, bounded prompt tiles, snapshot
   support, CLI and attention parity tests.
5. `setup: make low-VRAM planning degrade optional features safely` — capability probing, small-card choices,
   optional MTP lifetime/single-token serving and minimum-buffer reservation.
6. `tests: cover host-heavy low-VRAM configurations` — real-model integration, portable setup/Pascal test
   harness fixes and this documentation. Some engine CLI edits share `generate.cpp`; stage them by hunk.

No commits are created by this change. Existing local model configs and installed engine binaries are left alone.
