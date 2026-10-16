# Hybrid CPU/GPU Workload Optimization Artifacts (`Ornith-1.5-35B-Q4_K_M`)

Canonical analysis report: **[docs/performance/CPU_GPU_WORKLOAD_ANALYSIS.md](../../../docs/performance/CPU_GPU_WORKLOAD_ANALYSIS.md)**

---

## Provenance Summary

- **Hardware:** 2× Intel Xeon Gold 6130 (32 physical cores / 64 threads, AVX-512F/BW/DQ/VL, no VNNI), 377 GiB DDR4 across 2 NUMA nodes (`Linux 6.8.0-142-generic`); 1× NVIDIA GeForce GTX 1070 8 GiB (Pascal SM 6.1), driver `580.178.04`, CUDA `12.0` (`V12.0.140`).
- **Model:** `/home/ubuntu/models/ornith-1.5-35b/Ornith-1.5-35B-Q4_K_M.gguf` (SHA-256 `42739874cc2ccfdb8523b23fbe52e29b2a7555c8176737ca9ca0b5d59859d41f`) and pack `/mnt/models-ssd/Strata-data/packs/ornith-1.5-35b`.
- **Commits:**
  - Starting public commit: `499b61b` (`499b61be26a8ba7e619e452003ec89f4910b0fee`)
  - Preserved native prefill working tree: `2909102` (`29091029e9284651214240706ddaf27948b137b7`)
  - ABI-safe batched prefill projections, NUMA policy & thread affinity: `c4b0746`
  - Reproducible CPU/CUDA workload diagnostic & benchmark suite: `87fc64f` (`87fc64fc596bb859f6b5f4eca4db992475d115ca`)

---

## Key Artifact Groups

1. **Full-Shaped-Warmup 28-Case Benchmark Matrix (`280` measured rows, 10 runs/case, `context = 32,768`, `generate = 128`):**
   - Directory: [`matrix-owner-20261016T033335Z-2eb5322b/`](matrix-owner-20261016T033335Z-2eb5322b/)
   - Summary reports: [`matrix-owner-20261016T033335Z-2eb5322b/report.md`](matrix-owner-20261016T033335Z-2eb5322b/report.md), [`report.json`](matrix-owner-20261016T033335Z-2eb5322b/report.json), [`report.csv`](matrix-owner-20261016T033335Z-2eb5322b/report.csv)
   - Saved `B=256` baseline CSVs: `baseline-p{512,2048,4096,8192,16384}-b256.csv` and `.command.json`
   - Recursive directory summary (all primary and excluded CSVs): [`summary.json`](summary.json)
2. **CUPTI GPU Activity and Per-Thread CPU Expert Overlap Traces:**
   - Reviewed summaries: [`profile-final-gpu.tooling-review.summary.json`](profile-final-gpu.tooling-review.summary.json), [`profile-serial-gpu.tooling-review.summary.json`](profile-serial-gpu.tooling-review.summary.json)
   - Raw traces (deterministically gzip-compressed in [`archive-manifest.json`](archive-manifest.json)): `profile-final-gpu.tsv.gz`, `profile-serial-gpu.tsv.gz`, `activity-baseline-gpu.tsv.gz`
   - CPU expert call & per-layer profiles: `profile-final-cpu.csv`, `profile-final-layers.csv`, `profile-serial-cpu.csv`, `profile-serial-layers.csv`
3. **CPU Expert Microbenchmarks, Hardware Counters, and Disassembly:**
   - Microbenchmark CSVs: `cpu-avx{512,2}-g{1,256}-c{1,8,16,32}-t{1,8}.csv`
   - Core & uncore IMC DRAM counters / `turbostat`: `counters-avx512-t{1,8}.{perf,turbostat}.txt`, `counters-avx2-t{1,8}.{perf,turbostat}.txt`, `imc-read-write-t{1,8}.{perf,turbostat}.txt`
   - Static disassembly & register/stack analysis: [`assembly-summary.json`](assembly-summary.json), [`assembly-excerpts.txt`](assembly-excerpts.txt)
4. **Controlled `llama.cpp` Reference (`11fe02151f79c41d0d4af7da708755d73b9c0da6`, `context = 32,768`, `batch = 256`, `threads = 32`, `--aligned --host-kv`):**
   - `llama-cpu-only.csv` (`-ngl 0`), `llama-ngl16.csv` (`-ngl 16`), `llama-dense-gpu.csv` (`-ngl 99 --cpu-experts`)
5. **Numerical, Weight Representation, and Parity Validation:**
   - First-token logit & 128-token ID comparison: [`numerical-comparison.json`](numerical-comparison.json)
   - GGUF vs native pack weight audit: [`weight-audit.stdout.txt`](weight-audit.stdout.txt)
   - Real-model parity suites: [`short-parity.json`](short-parity.json) (`1..512`), [`long-parity.json`](long-parity.json) (`2050..16384`), [`validation-projection-widths.stdout.txt`](validation-projection-widths.stdout.txt), [`validation-registry.stdout.txt`](validation-registry.stdout.txt), [`validation-pool-affinity.stdout.txt`](validation-pool-affinity.stdout.txt), [`validation-batched-projections.stdout.txt`](validation-batched-projections.stdout.txt), [`validation-ctest.stdout.txt`](validation-ctest.stdout.txt)
   - Environment & compiler flag inventory: [`environment.json`](environment.json)
