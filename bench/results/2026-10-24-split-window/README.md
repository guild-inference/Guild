# Qwen decode and turn latency: the split verify window, short prompt tails, and ideas that did not help

Measured 2026-10-24 on this PC: Ubuntu, two Xeon Gold 6130 (32 physical cores, 2 NUMA nodes, AVX-512 without
VNNI), 376 GiB RAM, GTX 1070 8 GiB, driver 580.178.04, CUDA 12 / sm_61 Release build of `guild-generate` at
`5902dcd` plus this change (`-DGUILD_EXPERIMENTAL_SM60=ON`, the pinned CPU and CUDA backends of `backends.lock.json`).

Every row is the median of three fresh 256-token greedy generations after a 16-token warmup, measured with
`tools/bench_runtime.py` on its two default prompts (prose 35 tokens, code 40 tokens), thinking off. Model:
Qwen3.8-Flash-Next UD-IQ4_XS shards with the native pack, `data/expert-profile.bin`, the MTP draft, `--spec 4`,
`--spec-min-p 0.5`, FP16 host-only KV, automatic expert cache, 56 GiB expert RAM budget, 31 pool workers + host.
Context capacity is 32768 unless a row says 262144. Run-to-run spread within one configuration was about 0.5%.

## Result

The verifier can cut a window of T >= 2 tokens into two groups and order the stream so the CPU pool computes one
group's routed experts while the GPU runs the other group's mixer and router (`Verifier::record_window`). Every
token's arithmetic is unchanged. The setter existed but nothing called it after `guild-generate` became a runtime
wrapper (`4530824`); the older engine had measured it "exact, ~7% slower". The runtime now turns it on;
`GUILD_SPEC_SPLIT=0` is the one-group control. It stays off when every expert is VRAM resident.

Same binary, alternating control and new default:

| Configuration | Prose | Code |
| --- | ---: | ---: |
| One group (`GUILD_SPEC_SPLIT=0`), run 1 | 21.23 tok/s | 26.85 tok/s |
| Split (default), run 1 | **25.97 tok/s** | **33.73 tok/s** |
| One group, run 2 | 20.76 tok/s | 26.56 tok/s |
| Split, run 2 | **25.86 tok/s** | **33.66 tok/s** |
| One group, 262144 context | 21.00 tok/s | 26.24 tok/s |
| Split, 262144 context | **25.60 tok/s** | **34.40 tok/s** |

That is +22% to +31%. Before the change, a GUILD_VERIFY_PROFILE run spent about 37 of 90 ms per window with the GPU
waiting on the CPU pool, and a `perf` sample showed the pool's threads parked for most of the GPU's share of the
window: the two devices took turns. Ornith and other models without a drafter verify one token per window, where
the split does not apply.

## Exactness

The one-group control reproduced all six outputs of the unchanged binary, and the split's outputs were identical
from run to run. With the default rules the split's prose text differs from the one-group text from character
812, and the GPU expert cache's hit counts differ. Both come from rounding that already depends on how tokens are
grouped, not from the split:

- below `GUILD_IQ_MT_MIN` tokens (default 2) an expert's rows use ggml's per-token dot, which rounds differently
  from the multi-token kernels (#152, `guild-kernels-cpu` `native_expert.cpp`);
- VRAM-cached experts run on the GPU's grouped kernel per group.

With both removed (`GUILD_IQ_MT_MIN=1 --expert-cache 0`) the split matched the one-group control on **all six
256-token outputs and on every draft-acceptance count** (138/241 prose, 169/213 code), at 25.89 / 33.14 tok/s
against 21.13 / 25.84. The same grouping dependence already makes default output vary with MTP acceptance.

The runtime load-failure, performance, dispatch and inference-failure CTests passed on this build.

## Short prompt tails through the verifier

The Qwen prefill engine streams every selected routed expert to the GPU before it computes a chunk, which has a
fixed cost even for a few tokens. An agentic turn after a reused prefix usually adds only a few tokens. A request
that restores a 299-token checkpoint, prefills a new tail of k tokens and emits one token, through
`guild-generate --serve` (`tail_cost.py`, median of three):

| Tail tokens | Prefill engine | Verifier windows (4 tokens) |
| ---: | ---: | ---: |
| 2 | 582 ms | **161 ms** |
| 4 | 836 ms | **181 ms** |
| 8 | 1087 ms | **265 ms** |
| 16 | 1323 ms | **434 ms** |
| 32 | 1753 ms | **775 ms** |
| 64 | 2071 ms | **1437 ms** |
| 128 | **2555 ms** | 2793 ms |

They cross near 110 tokens here. Tails up to 64 tokens now use the verifier's windows when the model has the
prefill engine (`GUILD_PREFILL_VERIFY_MAX` moves the bound, 0 turns it off); the batched prefill, which computes
CPU experts itself, is unchanged. The first generated token matched in all 21 requests. The two paths round
differently (GPU against CPU experts), so later text can differ: the prose benchmark diverged at character 812, as
the split did. Decode rates were unchanged (25.68 / 34.26 tok/s). Time to first token of the cold 40-token code
prompt fell from 2347 to 932 ms, and of the cached ones from about 90-200 to 76 ms. In `prefix_seq.py`, a
continuation turn took 744 ms instead of 1582 and a branch with a checkpoint restore 788 ms instead of 1559.

## Measured and rejected

| Idea | Measurement | Verdict |
| --- | --- | --- |
| Pin the expert RAM so automatic PCIe offload runs (`GUILD_PIN_EXPERTS=1`; automatic mode pins only at <= 24 GiB on one NUMA node) | Offload chose 26/256; 21.21 / 26.44 tok/s against 21.53 / 27.30; load 42 s against 11 s | Slower: NUMA interleaving is worth more than the offload on this PC. The gate stays. |
| Use the GEXEC fused AVX2 CPU kernels without a `.gexec` (`iq256_row_dot_1row` instead of ggml's `ggml_vec_dot_iq3_s_q8_K`) | 21.39 / 27.22 tok/s, identical output; the IQ3_S kernel's share of `perf` samples fell from 18.3% to 16.6% | No measurable gain. Not changed. |
| 63 pool workers (the SMT siblings, unpinned) | 19.20 / 25.58 tok/s, identical output | Slower. |
| AVX2 `quantize_row_q8_K` for the host thread's activation quantization (ggml's x86 path is the scalar reference) | Bit-identical to the reference on 400000 rows; 0.92 against 4.11 us per 2560-float row, about 0.3 ms per window | Exact but below this benchmark's noise; not integrated. The test is in `q8k/`. |

## Reproduce

From `/home/ubuntu/Guild`, with the API server stopped:

```bash
.venv/bin/python tools/bench_runtime.py --exe <build>/guild-generate \
  --model /mnt/models-ssd/Strata-data/models/unsloth-UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf \
  --pack /mnt/models-ssd/Strata-data/packs/unsloth-ud-iq4_xs --profile data/expert-profile.bin \
  --mtp /mnt/models-ssd/Strata-data/mtp/rt --context 32768 --resident-budget-gib 56 \
  --runs 3 --max-new 256 --out split-on.json
GUILD_SPEC_SPLIT=0 .venv/bin/python tools/bench_runtime.py ... --out split-off.json
```

Add `GUILD_IQ_MT_MIN=1` and `--cache 0` to both for the exactness control. Raw reports, including generated
text and per-request counters, are in `raw/`.
