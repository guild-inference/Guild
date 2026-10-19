# Phase 2E multi-session, memory-pressure, and regression validation

All measurements in this report were collected on the same host and GPU as Phase 2D:

- **CPU**: Dual Intel Xeon Gold 6130 (2 sockets, 32 physical cores / 64 threads, AVX-512F/BW/DQ/VL, no VNNI)
- **RAM**: 377 GiB system memory
- **GPU**: 1x NVIDIA GeForce GTX 1070 8 GiB (`SM 6.1`, CUDA 12.0.140, driver `580.178.04`)
- **Model**: Ornith-1.5-35B (`ornith:35b`, 40 layers: 30 GDN + 10 full-attention QSA layers, 256 routed experts, top-10 active, `context = 32,768`, `batch = 256`, greedy `temperature = 0`)
- **Pinned repositories**:
  - `Guild`: `257642139acd325e3771b0e11d0d0b5384ad672b` + Phase 2E lifecycle/concurrency/staging changes
  - `guild-kernels-cpu`: `264227c74d0151b914d826b0fa35a0f576a117b8` (`org.guild.kernels.cpu@1.1.1`)
  - `guild-kernels-cuda`: `fe4c0808dab38e72d31b9b6bfcc0943ee6e3c79a` (`org.guild.kernels.cuda@1.4.0`)
  - `guild-runners-qwen`: `650b09cef3a1dec184a96605111fa6d44622ebe6` (`org.guild.runners.qwen@1.1.0`)

Raw artifacts are stored in `bench/results/phase2e-multisession/`.

## 1. Single-session counterbalanced parity vs Phase 1 and Phase 2D

Measured with `tools/bench_paired_plugins.py` (`--cases preserved builtin cpu cuda runner all --prompts 512 4096 16384 --rounds 2 --output bench/results/phase2e-multisession/paired-plugins`, `generate = 128`, `--full-warmup`):

| Prompt | Case | Prefill median (tok/s) | Decode median (tok/s) | Prefill vs `builtin` (%) | Decode vs `builtin` (%) | Decode vs Phase 1 `preserved` (%) | Phase 2D ref decode (tok/s) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 512 | `preserved` (Phase 1) | 238.939 | 43.369 | +0.25% | -6.98% | — | 43.479 |
| 512 | `builtin` | 238.350 | 46.621 | — | — | +7.50% | 46.802 |
| 512 | `cpu` (`.gkernel`) | 237.993 | 46.294 | -0.15% | -0.70% | +6.75% | 46.519 |
| 512 | `cuda` (`.gkernel`) | 238.074 | 46.626 | -0.12% | +0.01% | +7.51% | 46.793 |
| 512 | `runner` (`.gkernel`) | 237.984 | 46.295 | -0.15% | -0.70% | +6.75% | 46.612 |
| 512 | `all` (`.gkernel`) | 238.088 | 45.559 | -0.11% | -2.28% | +5.05% | 46.627 |
| 4096 | `preserved` (Phase 1) | 220.861 | 32.965 | +0.42% | -26.34% | — | 33.056 |
| 4096 | `builtin` | 219.955 | 44.754 | — | — | +35.77% | 44.860 |
| 4096 | `cpu` (`.gkernel`) | 219.874 | 44.759 | -0.03% | +0.01% | +35.78% | 44.828 |
| 4096 | `cuda` (`.gkernel`) | 219.912 | 44.597 | -0.01% | -0.35% | +35.29% | 44.844 |
| 4096 | `runner` (`.gkernel`) | 219.813 | 44.528 | -0.06% | -0.51% | +35.08% | 44.819 |
| 4096 | `all` (`.gkernel`) | 220.421 | 44.240 | +0.22% | -1.15% | +34.20% | 44.811 |
| 16384 | `preserved` (Phase 1) | 171.728 | 17.929 | -1.25% | -54.67% | — | 17.997 |
| 16384 | `builtin` | 173.903 | 39.551 | — | — | +120.60% | 39.731 |
| 16384 | `cpu` (`.gkernel`) | 173.706 | 39.583 | -0.11% | +0.08% | +120.78% | 39.671 |
| 16384 | `cuda` (`.gkernel`) | 174.002 | 39.639 | +0.06% | +0.22% | +121.09% | 39.622 |
| 16384 | `runner` (`.gkernel`) | 173.686 | 39.563 | -0.12% | +0.03% | +120.67% | 39.644 |
| 16384 | `all` (`.gkernel`) | 173.632 | 39.481 | -0.16% | -0.18% | +120.21% | 39.544 |

Single-session `builtin` vs Phase 2D is within `0.45%` on decode and `0.88%` on prefill across all tested prompt lengths, well inside the `1.5%` decode and `3.0%` prefill gates.

## 2. Multi-session concurrency (`1, 2, 4, 8` sessions) under active package operations

Measured with `ornith_prefill_bench --prompt 512 --generate 64 --runs 2 --batch 256 --context 32768 --full-warmup --sessions {1,2,4,8} --concurrent-pkg-ops` (`bench/results/phase2e-multisession/ms-{builtin,all}-s{1,2,4,8}-p512.json`).
Note: `GuildSession` handles created from a single `GuildModel` share the model's `SessionState` and serialize GPU execution on `GuildModel::Impl::generation_mutex`; concurrent sessions test queue serialization, per-session lease tracking, and resistance to concurrent package disable/remove attempts while requests are in flight.

| Mode | Sessions | Requests | Prefill p50 (tok/s) | Decode p50 (tok/s) | TTFT p50 / p95 (ms) | Latency p50 / p95 (ms) | VRAM sample (MiB) | Host RSS peak (GiB) | Max kernel / runner leases | Concurrent list ops | Busy unload rejections | Token parity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `builtin` | 1 | 2 | 236.17 | 45.29 | 2,189.4 / 2,191.2 | 3,580.5 / 3,582.3 | 2,622.19 | 37.71 | 45 / 3 | 6,426,252 | 0 | PASS |
| `builtin` | 2 | 4 | 236.53 | 44.92 | 3,989.7 / 5,782.5 | 5,392.1 / 7,184.2 | 2,622.19 | 37.71 | 47 / 4 | 12,947,365 | 0 | PASS |
| `builtin` | 4 | 8 | 235.44 | 44.92 | 7,594.5 / 12,994.4 | 8,999.5 / 14,396.7 | 2,622.19 | 37.71 | 51 / 6 | 25,779,045 | 0 | PASS |
| `builtin` | 8 | 16 | 234.08 | 46.14 | 14,710.5 / 27,211.0 | 16,077.0 / 28,587.9 | 2,622.19 | 37.71 | 59 / 10 | 59,739,957 | 0 | PASS |
| `all` (`.gkernel`) | 1 | 2 | 234.40 | 45.40 | 2,204.9 / 2,205.0 | 3,593.5 / 3,605.5 | 2,632.19 | 37.75 | 47 / 3 | 788,793 | 4,732,758 | PASS |
| `all` (`.gkernel`) | 2 | 4 | 233.95 | 45.00 | 4,013.1 / 5,822.1 | 5,414.1 / 7,223.6 | 2,632.19 | 37.75 | 50 / 4 | 1,356,637 | 8,139,822 | PASS |
| `all` (`.gkernel`) | 4 | 8 | 234.25 | 44.71 | 7,625.0 / 13,059.6 | 9,037.1 / 14,467.6 | 2,632.19 | 37.75 | 56 / 6 | 2,733,633 | 16,401,798 | PASS |
| `all` (`.gkernel`) | 8 | 16 | 233.04 | 44.92 | 14,822.2 / 27,375.6 | 16,206.1 / 28,782.8 | 2,632.19 | 37.75 | 68 / 10 | 5,884,009 | 35,304,054 | PASS |

## 3. Resource-aware KV staging policy under memory pressure

On Ornith-1.5-35B (10 full-attention QSA layers), per-layer QSA staging (`own_staging = true`) allocates `10 * 64.02 MiB = 640.16 MiB` (`671,252,480` B) of device staging tiles so each layer retains its cached KV pages across decode steps. Shared staging (`own_staging = false`) allocates a single `64.02 MiB` (`67,125,248` B) staging pool shared across all 10 QSA layers, saving `576.00 MiB` (`603,979,776` B) of VRAM per `SessionState` arena at the cost of reloading the staging pool across layers during decode.

Measured at `prompt = 4,096`, `generate = 64`, `runs = 2` with all external `.gkernel` packages (`bench/results/phase2e-multisession/staging-*-p4096.json`):

| Policy | Setting | `kv_shares_staging` | Total KV staging VRAM | Sampled GPU VRAM | Prefill p50 (tok/s) | Decode p50 (tok/s) | Token parity |
|---|---|---|---:|---:|---:|---:|---|
| Per-layer staging (single-session default) | `GUILD_KV_SHARE_STAGING=0` (or unset when VRAM headroom >= 1 GiB) | `false` | 640.16 MiB (`671,252,480` B) | 2,632.19 MiB (`2,760,048,640` B) | 216.14 | 43.21 | PASS |
| Explicit shared staging | `GUILD_KV_SHARE_STAGING=1` | `true` | 64.02 MiB (`67,125,248` B) | 2,056.19 MiB (`2,156,068,864` B) | 214.56 | 32.77 | PASS |
| Automatic multi-session budget (`> 1 GiB` cap) | `GUILD_MAX_CONCURRENT_SESSIONS=4` (or `GUILD_KV_STAGING_BUDGET_MIB=128`) | `true` | 64.02 MiB (`67,125,248` B) | 2,056.19 MiB (`2,156,068,864` B) | 215.12 | 32.97 | PASS |
