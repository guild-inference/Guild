# Fifteenth round: the verifier's streams at the top priority, and Ornith's expert placement

PC: 2x Xeon Gold 6130 (16 cores each, AVX-512 BW without VNNI, 2 NUMA nodes, 376 GiB RAM), GTX 1070 8 GB (sm_61,
15 SMs, 151 W cap, PCIe 3.0 x16), CUDA 12. Branch `perf/next` (from `perf/prompt-path`), the pinned kernel
revisions (guild-kernels-cpu `9b7281a`, guild-kernels-cuda `de65d60`).

## The verifier's streams at the GDN stream's priority (now the default)

The verifier already ran its GDN stream at the device's greatest priority; its main and shared-expert streams were
at the default. Pascal has two levels, so work queued beside a window on default-priority streams competed with
the window on equal terms. That work is Drama 3's tower layers and codec chunks (round 11's `Beside` worker). Now
all three of the window's streams are at the greatest priority. `GUILD_VERIFY_STREAM_PRIO=0` restores the old
behavior. Only the scheduling changes, so the results are the same.

**Drama 3** (Q4_K_M, the server's settings, one 286-character text, `--max-new 2000`). Two seeds, 2 rounds
alternated, frames/s:

| Seed | Default priority | Greatest priority | Change | WAV md5, both arms |
| --- | ---: | ---: | ---: | --- |
| 11 | 45.84 / 45.85 | 47.08 / 46.45 | +2.7% / +1.3% | `6e04c9b1` |
| 12 | 45.91 / 46.05 | 46.73 / 46.35 | +1.8% / +0.7% | `f84288c9` |

The tower's own time grows (1.6-1.8 s to 1.9-2.2 s a text) because it now waits for the window's kernels. The
frames come faster because the backbone no longer waits for the tower.

**Qwen3.8-Flash-Next** (UD-IQ4_XS, setup's flags with the DFlash drafter; prose and code, 2 runs x 256 tokens, 2
rounds alternated), tok/s:

| Arm | prose | code |
| --- | ---: | ---: |
| default priority | 49.05 / 48.89 | 62.07 / 61.51 |
| greatest priority | 49.11 / 49.19 | 61.76 / 61.65 |

**Ornith-1.5-35B** (Q4_K_M, its MTP drafter, `--spec 4`; prose and code), tok/s:

| Arm | prose | code |
| --- | ---: | ---: |
| default priority | 80.34 / 79.61 | 92.82 / 91.95 |
| greatest priority | 79.96 / 79.24 | 92.58 / 91.95 |

For both models the arms are within their run-to-run spread (about 1%), and every run's tokens are the same in
both arms.

## Ornith: where its time goes, and two placements that lost

Ornith decodes GPU-bound. Under nsys the GPU is busy about 79% of a window, and most of the gaps are nsys's own
graph launch time. The dense staged MMVQ products are 47.5% of kernel time. The load leaves VRAM unused (3,645 MiB
in use), so a GPU expert cache looked promising:

- **An expert profile.** `--expert-cache auto` keeps a cache only with a profile. A 16-prompt routing trace
  (`--dump-routing`, `tools/make_profile.py`) shows the hottest 20% of (layer, expert) pairs serving 46.6% of
  lookups.
- **GPU support for its formats.** The cache could not hold Ornith's Q4_K and Q6_K down matrices. A local kernel
  branch (guild-kernels-cuda `perf/q6k-experts`) adds them. Against the CPU's own rows on layers 0-9, the GPU path's
  relative error is 1.0-1.2e-2, the CPU's 1.5-1.7e-2.

With both, the cache kept 1,634 slots (2.96 GiB), and decode was slower:

| Arm | prose | code | story | sql | Chinese | Russian |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| no cache | 79.9 / 79.3 | 92.1 / 91.9 | 71.3 / 71.1 | 90.2 / 89.9 | 75.9 / 75.7 | 59.7 / 59.7 |
| profile, 1,634 slots | 60.8 / 60.3 | 69.2 / 68.4 | 56.9 / 56.7 | 71.6 / 71.0 | 59.8 / 59.5 | 47.2 / 47.0 |

A GPU-bound model only adds GPU work when it moves experts to the GPU; the CPU was already hiding them. The
profile and the kernel branch are not shipped.

**One token group instead of two** (`GUILD_SPEC_SPLIT=0`) reads the dense weights once a window instead of twice.
That puts the CPU experts on the critical path: prose 70.0 / 69.8 against 79.2 / 79.4, code 78.3 / 78.0 against
92.1 / 91.8.

## Files

`runs/`:
- Drama 3: `dp.sh`, `dp.progress`.
- Qwen: `qp.sh`, `qb.sh`, `qp.progress`, `qp*.sum`.
- Ornith: `oprio.*` (priority), `oab2.*` (expert cache; `gq` is the build with the kernel branch), `osplit.*` (token
  groups).

The progress files list each run's tok/s and a hash of its token ids.
