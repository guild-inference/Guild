# Phase 6.6: Adaptive Speculative Decoding, Draft-Free Prediction, and Cost-Aware Inference

**Date**: October 21, 2026  
**Baseline Commit**: `afcdd1d` (Phase 6.5 complete)  
**Target Models**:
- **Ornith 1.5 35B (`qwen35moe`)**: `/mnt/models-ssd/Strata-data/gpacks/ornith-1.5-35b-ext.gpack` (`256` routed experts/layer, `top-8`, `40` layers, no MTP weights)
- **Qwen3.8-Flash-Next (`qwen4_exp`)**: `/mnt/models-ssd/Strata-data/gpacks/qwen3.8-flash-next-ext.gpack` (`512` routed experts/layer, `top-10`, `48` layers, GPU MTP draft layer in `/mnt/models-ssd/Strata-data/mtp/rt`)

---

## 1. Executive Summary

Phase 6.6 designs, implements, validates, and benchmarks an **adaptive, cost-aware speculative decoding system** across Guild's hybrid GPU+CPU Mixture-of-Experts (MoE) inference pipeline. Rather than maximizing draft acceptance rate or draft depth in isolation, the runtime scheduler explicitly minimizes **expected wall-clock milliseconds per committed target-model output token**:

$$C(s, k) = \frac{\mathbb{E}[\text{iteration wall-clock time (ms)}]}{\mathbb{E}[\text{committed output tokens}]}$$

where iteration wall-clock time includes drafting/lookup/retrieval latency, target-model MoE verification across window size $T = 1 + k$, GPU-CPU synchronization, and KV/GDN/QSA commit or rollback, and committed output tokens count actual target-model tokens emitted ($\text{accepted draft prefix} + 1$ target/bonus token).

### Headline Measured Outcomes (2× Xeon Gold 6130 + GTX 1070 8 GiB, GEXEC Active)

1. **Ornith 1.5 35B (`ornith:35b`, no MTP weights — previously limited to `T=1` single-token decode)**:
   - Unlocked multi-token speculative verification (`T=1..5`) for draftless **Prompt-Lookup (`lookup`)**, **Session-Local Retrieval (`retrieval`)**, and **Cost-Aware Adaptive (`adaptive`)** modes without requiring any draft model or extra VRAM weights.
   - Across input-grounded, structured, RAG, multi-turn, and long-context workloads (`W2`–`W6`, `W10`), **`adaptive` delivers `1.395x` to `1.608x` end-to-end decode speedups** over `normal` decoding (`50.0` $\rightarrow$ **`80.65 tok/s`** on `W2` summarization, **`77.75 tok/s`** on `W4` RAG, **`69.79 tok/s`** on `W5` multi-turn, and **`80.03 tok/s`** at `p=512` / **`55.31 tok/s`** at `p=32,704` populated context).
   - On partially repetitive code editing (`W1`) and multi-turn conversation (`W5`), **`adaptive` beats both `normal` AND fixed `lookup`/`retrieval`** (`52.98 tok/s` vs `50.29` `normal` and `50.91` `lookup` on `W1`; `69.79 tok/s` vs `50.02` `normal`, `68.27` `lookup`, and `68.38` `retrieval` on `W5`) by dynamically modulating draft depth and backing off on low-confidence spans.
   - On speculation-hostile creative prose (`W7`) and mathematical reasoning (`W8`), fixed `lookup` regresses to `48.84 tok/s` (`-1.4%`) and `47.91 tok/s` (`-4.5%`), whereas **`adaptive` automatically falls back to `normal` decoding (`49.53 tok/s` on `W7`, `50.01 tok/s` on `W8` — within `0.04%`–`0.28%` of `normal`)**.
   - Under 2-session concurrent execution (`--sessions 2`), `adaptive` scales median decode throughput from `49.75 tok/s` (`normal`) to **`83.68 tok/s` (`1.682x`)**, cutting median round wall time from `7,058.17 ms` to **`5,993.70 ms` (`-15.1%`)** with `100%` deterministic session isolation.

2. **Qwen3.8-Flash-Next (`qwen3.8-flash-next`, hybrid arbitration across `normal`, `mtp`, `lookup`, `retrieval`)**:
   - Arbitrates dynamically between GPU **MTP drafting (`k=1`, `T=2`)**, CPU **Prompt-Lookup (`k=1..4`, `T=2..5`)**, **Session-Local Retrieval (`k=1..4`)**, and **Normal decoding (`k=0`, `T=1`)**.
   - On input-grounded/repetitive workloads (`W2` summarization, `W4` RAG, `W6` boilerplate, `W10` `32K`–`65K` long context), **`adaptive` shifts from `mtp` (`23.3`–`25.7 tok/s`) to deep `lookup` (`k=4`), reaching `32.02`–`33.70 tok/s` (`1.746x`–`1.843x` over `normal`'s `18.3 tok/s`, and `+31.0%`–`+38.3%` faster than `mtp` alone)**.
   - On workloads where prompt-lookup fails or regresses (`W1` code editing where `lookup` drops to `17.85 tok/s`; `W7` creative prose where `lookup` drops to `17.88 tok/s`; `W8` math reasoning where `lookup` drops by **`-13.8%`** to `15.91 tok/s`), **`adaptive` selects `mtp` and achieves `25.39 tok/s` (`W1`), `22.15 tok/s` (`W7`), and `24.31 tok/s` (`W8`, `+52.8%` faster than fixed `lookup` and `+31.7%` faster than `normal`)**.
   - On `W5` multi-turn conversation and `W10` `16,384`-token populated context, **`adaptive` beats every fixed strategy (`normal`, `mtp`, `lookup`, `retrieval`)** by combining `MTP` on novel transitions with `Lookup`/`Retrieval` on repeated spans (`23.38 tok/s` on `W5` vs `18.37` `normal`, `20.58` `lookup`, `19.97` `retrieval`, `22.87` `mtp`; `24.50 tok/s` at `p=16,384` vs `17.92` `normal`, `20.44` `lookup`, `23.94` `mtp`).
   - At **`65,472` populated context tokens (`65K`)**, `adaptive` achieves **`31.78 tok/s` (`31.47 ms/tok`)** vs `17.49 tok/s` (`normal`) and `24.20 tok/s` (`mtp`) — a **`1.817x` decode speedup at 65K context**.

3. **100% Exact Greedy & Target-Distribution Sampling Parity**:
   - Fixed a subtle `#152` reduction-order gate in `src/kernels/cpu/gexec_expert.cpp` (`gexec_iq4nl_down_rows`, `iq256_row_dot_1row`, `gexec_iq4xs_gu_rows_nt`) so GEXEC's AVX-512/AVX2 CPU expert providers honor `GUILD_IQ_MT_MIN=1` for `nt == 1`, achieving **100% bit-for-bit token ID parity across `normal`, `mtp`, `lookup`, `retrieval`, and `adaptive` on every workload (`W1`–`W10`) on both Ornith 35B and Qwen3.8-Flash-Next**.
   - Verified **100% bitwise seed-invariant sampling parity** under non-greedy sampling (`temperature=0.7, top_p=0.9, min_p=0.05, top_k=40, seed=12345`) on real model execution and `25,000`-token Monte Carlo distributional tests (`TVD < 0.01`, zero mass outside `top_k`/`top_p`/`min_p` support).

---

## 2. External Research & Systems Survey (Through October 2026)

| Technique / System | Core Mechanism | Applicability to Guild's Hybrid GPU+CPU MoE Runtime |
| :--- | :--- | :--- |
| **Prompt Lookup Decoding (PLD)** (Saxena, 2023; vLLM `ngram` / SGLang `lookahead`, 2024–2026) | Matches suffix n-grams of the current sequence against earlier positions in the prompt and generated output; proposes verbatim continuation tokens with zero model weights. | **Directly adopted & enhanced (`SuffixDrafter`)**. Costs `< 0.005 ms` on CPU, requires `0 B` of VRAM, and unlocks speculative decoding on models without MTP weights (Ornith 35B) as well as deeper windows ($k=4$) on Qwen3.8-Flash-Next. |
| **REST: Retrieval-Based Speculative Decoding** (He et al., NAACL 2024) | Retrieves continuations from a token datastore using suffix matches and ranks multiple candidate continuations by length- and frequency-weighted voting. | **Adapted with strict per-session isolation (`RetrievalDrafter`)**. Rather than a global unverified external corpus, Guild maintains a bounded, strictly session-local two-tier token datastore (`corpus_` + `stream_`) with length-squared $\times$ recency continuation voting and confidence truncation. |
| **Native Multi-Token Prediction (MTP) / EAGLE-2 / EAGLE-3** (DeepSeek-V3, Qwen3/Qwen4_exp, Li et al. 2024–2026) | Uses a lightweight learned draft head/layer conditioned on the target model's top-layer hidden state $R_p$ and token embedding $e_{p+1}$, dynamically truncating drafts via confidence thresholds. | **Integrated via `MtpDrafter` + `AdaptiveDecodeScheduler`**. On Qwen3.8-Flash-Next, MTP provides `75%`–`93%` step-1 acceptance on novel prose/math where n-gram lookup has zero matches, while `MtpDrafter::prefill` (`step_row0 == -1`) keeps MTP's KV cache synchronized in `~0.35 ms` whenever `Lookup`, `Retrieval`, or `Normal` is selected. |
| **MoE-Aware Speculative Verification & Offloaded MoE Cost Modeling** (MoE-Spec, SpecExec, Expert-Budgeted Verification, 2025–2026) | Recognizes that in sparse MoE models with CPU-resident or offloaded experts, verification time $V(T)$ is *not* constant in $T$: distinct activated experts per layer grow as $U(T) \approx E(1 - (1 - K/E)^{T_{\text{eff}}})$, increasing CPU expert dispatch bandwidth. | **Central pillar of Guild's `AdaptiveDecodeScheduler` & GEXEC calibration**. On Guild's dual-socket Xeon + GTX 1070 host, $V(2)/V(1) \approx 1.38\times$ on Ornith (`256` experts, `K=8`) and $\approx 1.48\times$ on Qwen (`512` experts, `K=10`). A draft depth $k$ only reduces ms/token when $\mathbb{E}[A + 1] > (D(s, k) + V(1+k)) / V(1)$. |

---

## 3. Phase 6.6 Architecture & Implementation

### 3.1 Upgraded Prompt-Lookup Drafter (`include/guild/spec/suffix_drafter.hpp`, `src/spec/suffix_drafter.cpp`)
- **Multi-Length Anchors & Bounded Probing**: Indexes 3-gram anchors (plus 2-gram anchors with distinct hash tag bits when `min_match == 2`) in a power-of-two open-addressing table with `WAYS = 4` positions per slot, bounded linear probing (`kMaxProbe = 16`), and oldest-position LRU slot eviction on probe saturation.
- **Frequency- & Recency-Aware Continuation Ranking**: Extends candidate matches backwards up to `max_match` (`64` tokens) and ranks equal-length matches by continuation token agreement frequency (`last_freq()`) first, breaking ties by recency.
- **Sliding-Window Compaction**: Enforces `set_max_history(spec_lookup_window)` via automatic half-window compaction and re-indexing so memory stays strictly bounded (`~16.8 MiB` for a `65,536`-token window) across arbitrarily long sessions.

### 3.2 Session-Local Retrieval Drafter (`include/guild/spec/retrieval_drafter.hpp`, `src/spec/retrieval_drafter.cpp`)
- **Two-Tier Session-Local Datastore**:
  1. `corpus_`: session-local persistent token store across turns or explicitly ingested reference documents (`GuildSession::ingest_retrieval_tokens`, `clear_retrieval_corpus`), bounded by `max_corpus_tokens`.
  2. `stream_`: active request prompt + committed output tokens, reset at the start of each generation (`reset_stream()`).
- **REST-Style Length-Weighted Continuation Voting**: Gathers up to `kMaxCandidates = 16` deduplicated matching spans across 4-gram and 3-gram anchors from both `stream_` and `corpus_`, weights each span $c$ by:
  $$w_c = (\text{match\_len}_c)^2 \cdot \text{source\_boost}_c \cdot \frac{1}{1 + 0.08\,\text{rank}_c}$$
  and votes at each continuation step $j \in [0, k-1]$, pruning diverging spans and stopping early if continuation confidence $w_{\text{best}} / \sum w_c < \text{min\_confidence}$.
- **Strict Session Isolation**: Owned exclusively by `GuildSession::Impl` under `spec_mutex`; never shared across sessions or global state.

### 3.3 Cost-Aware Adaptive Decoding Scheduler (`include/guild/spec/adaptive_scheduler.hpp`, `src/spec/adaptive_scheduler.cpp`)
- **Objective Function**: At every safe verification boundary, evaluates candidate pairs $(s, k)$ across `Normal(0)`, `Mtp(1..mtp_cap_k)`, `Lookup(1..max_draft)`, and `Retrieval(1..max_draft)` using:
  $$C_{\text{model}}(s, k) = \frac{\hat{D}(s, k) + \hat{V}(1 + k) + \mathbb{I}[\text{MTP catch-up}]\,\hat{M}_{\text{catchup}}}{\mathbb{E}[A_{s,k} + 1]}$$
  blended with the strategy's decayed empirical cumulative wall-clock time-to-output ratio $\hat{C}_{\text{emp}}(s) = \frac{\sum \gamma^i \Delta t_i}{\sum \gamma^i \Delta n_i}$ whenever empirical cost exceeds the analytical acceptance model.
- **Live $T=1$ Seeding & Graph-Capture Subtraction**: In `GuildSession::generate` (`src/runtime/session.cpp`), one-time CUDA graph capture time (`m_impl->ver->ms_capture` delta) is subtracted from verification observations so graph compilation never pollutes steady-state cost estimates, and the initial $T=1$ TTFT window seeds `ver_decayed_ms_[1]` directly for the current prompt's exact KV context length.
- **Conservative Short-Match Prior & Hysteresis**:
  - Uses a conservative prior (`q = 0.32`) for isolated 3-gram matches (`match_len == 3, freq == 1`) so accidental 3-token function-word overlaps in creative prose or math (`W7`, `W8`) do not trigger unprofitable verification windows, while immediately boosting repeated 3-grams (`freq >= 2`) and longer matches (`match_len >= 4`, `q = 0.78..0.95`).
  - Requires a `5%` (`hysteresis_margin = 0.05`) expected ms/token improvement and `min_dwell_steps = 3` before switching strategies, with immediate backoff after `2` consecutive zero-acceptance windows.
- **MTP Confidence EMA & Cap Separation**: Tracks `mtp_prob_ema` in `Adaptive` mode so a single uncertain token (`dprob[0] < spec_min_p`) falls back for that window without latching MTP off for the rest of the sequence, and bounds MTP draft steps by `mtp_cap_k` while allowing `Lookup`/`Retrieval` to propose up to `spec_max_draft`.

### 3.4 Target-Distribution-Preserving Speculative Sampling Verification
- Guild's `Verifier::run` (`src/core/verify.cpp`) evaluates the exact target sampling chain (`repetition/frequency/presence penalties` $\rightarrow$ `top_k` $\rightarrow$ `top_p` $\rightarrow$ `min_p` $\rightarrow$ `temperature`) at every window row $t \in [0, T-1]$, drawing uniform variates from position-indexed **`Philox(seed, pos0 + t)`**.
- For deterministic proposals (`Lookup` and `Retrieval`), accepting draft prefix $d_0 \dots d_{a-1}$ iff $d_t = Y_{\text{pos0}+t}$ and emitting $Y_{\text{pos0}+a}$ produces **the exact same token sequence (bitwise identical for any fixed seed)** as single-token `Normal` sampling because every committed position $q$ emits $Y_q = F^{-1}_{P_{\text{target}}(\cdot \mid x_{\le q})}(\text{Philox}(\text{seed}, q))$.
- In `GuildSession::generate`, `PenaltyHistoryGuard` stages per-row penalty histories via `kernels::penalty_rows` across the window so repetition/frequency/presence penalties are also exact across all $T$ rows.

### 3.5 GEXEC Speculative Cost Calibration (`include/guild/gexec/gexec_contract.hpp`, `src/gexec/gexec_optimizer.cpp`)
- Extended `GExecHeterogeneousPlan` with hardware-calibrated speculative fields (`spec_calibrated`, `verify_window_ms[1..8]`, `verify_cpu_expert_ms[1..8]`, `distinct_expert_ratio[1..8]`, `mtp_draft_step_ms`, `mtp_kv_catchup_ms`, `lookup_draft_ms`, `retrieval_draft_ms`, `kv_commit_ms`) protected by the section's SHA-256 integrity digest.
- During `guild optimize` (Pass 4), `calibrate_speculative_costs` microbenchmarks `SuffixDrafter::propose` and `RetrievalDrafter::propose` on the host CPU and computes the MoE multi-token verification cost curve $V(1..8)$ and distinct expert expansion ratio $U(T)/U(1)$ from the Pass 3 tuned CPU expert decode latency and model MoE geometry.
- `guild inspect <artifact.gexec> --plan` displays the calibrated `Verify T=1..4`, `Distinct Experts`, and `Lookup / Retr` timings.

---

## 4. Hardware, Software & Artifact Provenance

- **Primary Host Hardware**: Dual-socket Intel Xeon Gold 6130 (`32` physical cores / `64` logical threads, Skylake-SP, `AVX-512BW/DQ/VL`, `2` NUMA nodes, `376 GiB` DDR4 RAM) + NVIDIA GeForce GTX 1070 (`8 GiB` GDDR5, Pascal `sm_61`, CUDA 12.8, driver `570.124.06`).
- **Secondary Workstation Target (Ryzen 5 5600X + RTX 5070)**: Pending hardware access; no numbers are fabricated or extrapolated for unavailable hardware.
- **Artifact SHA-256 Digests**:
  - `/mnt/models-ssd/Strata-data/gpacks/ornith-1.5-35b-ext.gpack`: `ac41e2e04d14949e998f6aade627de034b013b1f1d4670329f2e1e5efd67acdf` (manifest SHA-256: `b7631598f1cc21bbaf0e0e73844ed86d4cc90a9027823b03d0172438cb86e0ea`)
  - `/mnt/models-ssd/Strata-data/gpacks/qwen3.8-flash-next-ext.gpack`: `93f3fcd86786870c487ee72b8dbd362973bdd71c1b0d94bdad6ee689b98856e5` (manifest SHA-256: `62eca78bdc5544191166276cfea50cfa58f150ff9d00a3498cb633c20b91c146`)
  - `build-cuda12/phase66/gexec/ornith-65k-s0.gexec`: cache key `bcddd79ec7030fe4...` (`cpu.expert_pool.gexec_fused_avx512`, `spec_calibrated = true`)
  - `build-cuda12/phase66/gexec/qwen-65k-s0.gexec`: cache key `4fd75c92366672d5...` (`cpu.expert_pool.gexec_fused_avx512`, `spec_calibrated = true`)
  - `build-cuda12/phase66/gexec/qwen-65k-s2.gexec`: cache key `1950047812444efd...` (`cpu.expert_pool.gexec_fused_avx512`, `spec_calibrated = true`)

---

## 5. Measured Benchmark Results

### 5.1 Ornith 1.5 35B (`ornith:35b`) — 10 Workload Categories (`W1`–`W10`)

All runs executed with `--gexec build-cuda12/phase66/gexec/ornith-65k-s0.gexec` (`gexec_active: true`), `--pcie-frac 0`, `--spec-max-draft 4`, and verified for **100% bit-for-bit greedy token ID parity** against `normal`.

| Workload Category | Prompt / Gen | Strategy | Decode Time (ms) | Decode (tok/s) | Cost (ms/tok) | Speedup vs Normal | Verify Windows | Drafts Off / Acc | Acc Rate | Switches / Fallbacks | Final Strategy |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **W1: Code Editing / Patching** | `256 / 128` | `normal` | `2545.11` | `50.29` | `19.88` | `1.000x` | `128` | `0 / 0` | `0.0%` | `0 / 0` | `normal` |
| | `256 / 128` | `lookup` | `2514.09` | `50.91` | `19.64` | `1.012x` | `86` | `79 / 42` | `53.2%` | `21 / 10` | `lookup` |
| | `256 / 128` | **`adaptive`** | **`2416.04`** | **`52.98`** | **`18.88`** | **`1.053x`** | `101` | `34 / 27` | **`79.4%`** | `11 / 5` | `lookup` |
| **W2: Document Summarization** | `256 / 128` | `normal` | `2551.84` | `50.16` | `19.94` | `1.000x` | `128` | `0 / 0` | `0.0%` | `0 / 0` | `normal` |
| | `256 / 128` | `lookup` | `1562.40` | `81.93` | `12.21` | `1.633x` | `27` | `104 / 101` | `97.1%` | `0 / 0` | `lookup` |
| | `256 / 128` | **`adaptive`** | **`1587.01`** | **`80.65`** | **`12.40`** | **`1.608x`** | `28` | `101 / 100` | **`99.0%`** | `0 / 0` | `lookup` |
| **W3: JSON Schema Output** | `256 / 128` | `normal` | `2557.27` | `50.05` | `19.98` | `1.000x` | `128` | `0 / 0` | `0.0%` | `0 / 0` | `normal` |
| | `256 / 128` | `lookup` | `1665.21` | `76.87` | `13.01` | `1.536x` | `31` | `104 / 97` | `93.3%` | `3 / 1` | `lookup` |
| | `256 / 128` | **`adaptive`** | **`1694.40`** | **`75.54`** | **`13.24`** | **`1.509x`** | `34` | `98 / 94` | **`95.9%`** | `3 / 1` | `lookup` |
| **W4: RAG Grounded Q&A** | `256 / 128` | `normal` | `2563.83` | `49.93` | `20.03` | `1.000x` | `128` | `0 / 0` | `0.0%` | `0 / 0` | `normal` |
| *(+ session corpus)* | `256 / 128` | `lookup` | `1554.86` | `82.32` | `12.15` | `1.649x` | `29` | `100 / 99` | `99.0%` | `2 / 1` | `lookup` |
| | `256 / 128` | `retrieval` | `1552.86` | **`82.43`** | **`12.13`** | **`1.651x`** | `29` | `100 / 99` | `99.0%` | `2 / 1` | `retrieval` |
| | `256 / 128` | **`adaptive`** | **`1646.32`** | **`77.75`** | **`12.86`** | **`1.557x`** | `31` | `101 / 97` | `96.0%` | `2 / 1` | `retrieval` |
| **W5: Multi-Turn Conversation** | `128 / 128` | `normal` | `2559.03` | `50.02` | `19.99` | `1.000x` | `128` | `0 / 0` | `0.0%` | `0 / 0` | `normal` |
| *(+ Turn 1 in session corpus)* | `128 / 128` | `lookup` | `1875.00` | `68.27` | `14.65` | `1.365x` | `40` | `107 / 88` | `82.2%` | `9 / 4` | `lookup` |
| | `128 / 128` | `retrieval` | `1871.86` | `68.38` | `14.62` | `1.367x` | `40` | `107 / 88` | `82.2%` | `9 / 4` | `retrieval` |
| | `128 / 128` | **`adaptive`** | **`1834.04`** | **`69.79`** | **`14.33`** | **`1.395x`** | `47` | `86 / 81` | **`94.2%`** | `9 / 4` | `retrieval` |
| **W6: Repetitive Boilerplate** | `256 / 128` | `normal` | `2555.94` | `50.08` | `19.97` | `1.000x` | `128` | `0 / 0` | `0.0%` | `0 / 0` | `normal` |
| | `256 / 128` | `lookup` | `1662.58` | `76.99` | `12.99` | `1.537x` | `34` | `94 / 94` | `100.0%` | `6 / 3` | `lookup` |
| | `256 / 128` | **`adaptive`** | **`1725.42`** | **`74.18`** | **`13.48`** | **`1.481x`** | `41` | `89 / 87` | `97.8%` | `6 / 3` | `lookup` |
| **W7: Creative Novel Prose** | `256 / 128` | `normal` | `2583.10` | `49.55` | `20.18` | `1.000x` | `128` | `0 / 0` | `0.0%` | `0 / 0` | `normal` |
| *(speculation-hostile)* | `256 / 128` | `lookup` | `2620.78` | `48.84` | `20.47` | `0.986x` | `127` | `8 / 1` | `12.5%` | `4 / 2` | `normal` |
| | `256 / 128` | **`adaptive`** | **`2584.49`** | **`49.53`** | **`20.19`** | **`1.000x`** | `128` | `2 / 0` | `0.0%` | `2 / 1` | **`normal`** |
| **W8: Mathematical Reasoning** | `256 / 128` | `normal` | `2552.52` | `50.15` | `19.94` | `1.000x` | `128` | `0 / 0` | `0.0%` | `0 / 0` | `normal` |
| *(speculation-hostile)* | `256 / 128` | `lookup` | `2671.71` | `47.91` | `20.87` | `0.955x` | `98` | `73 / 30` | `41.1%` | `29 / 14` | `lookup` |
| | `256 / 128` | **`adaptive`** | **`2559.33`** | **`50.01`** | **`19.99`** | **`0.997x`** | `114` | `24 / 14` | `58.3%` | `18 / 9` | **`normal`** |
| **W9: Short Interactive Chat** | `64 / 64` | `normal` | `1284.27` | `49.83` | `20.07` | `1.000x` | `64` | `0 / 0` | `0.0%` | `0 / 0` | `normal` |
| | `64 / 64` | `lookup` | `1365.78` | `46.86` | `21.34` | `0.940x` | `60` | `16 / 4` | `25.0%` | `8 / 4` | `normal` |
| | `64 / 64` | **`adaptive`** | **`1339.30`** | **`47.79`** | **`20.93`** | **`0.959x`** | `62` | `6 / 2` | `33.3%` | `4 / 2` | **`normal`** |

### 5.2 Ornith 1.5 35B (`ornith:35b`) — W10 Populated-Context Scaling (`512` to `32,704` Tokens)

| Populated Context | Strategy | Decode Time (ms) | Decode (tok/s) | Cost (ms/tok) | Speedup vs Normal | Verify Windows | Drafts Off / Acc | Acc Rate | Parity |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`512`** | `normal` | `1279.30` | `50.03` | `19.99` | `1.000x` | `64` | `0 / 0` | `0.0%` | Exact |
| | `lookup` | `802.51` | `79.75` | `12.54` | `1.594x` | `14` | `50 / 50` | `100.0%` | Exact |
| | **`adaptive`** | **`799.72`** | **`80.03`** | **`12.50`** | **`1.600x`** | `14` | `50 / 50` | `100.0%` | Exact |
| **`4,096`** | `normal` | `1331.46` | `48.07` | `20.80` | `1.000x` | `64` | `0 / 0` | `0.0%` | Exact |
| | `lookup` | `854.85` | `74.87` | `13.36` | `1.558x` | `14` | `50 / 50` | `100.0%` | Exact |
| | **`adaptive`** | **`857.33`** | **`74.65`** | **`13.40`** | **`1.553x`** | `14` | `50 / 50` | `100.0%` | Exact |
| **`8,192`** | `normal` | `1391.80` | `45.98` | `21.75` | `1.000x` | `64` | `0 / 0` | `0.0%` | Exact |
| | `lookup` | `899.79` | `71.13` | `14.06` | `1.547x` | `14` | `50 / 50` | `100.0%` | Exact |
| | **`adaptive`** | **`900.22`** | **`71.09`** | **`14.07`** | **`1.546x`** | `14` | `50 / 50` | `100.0%` | Exact |
| **`16,384`** | `normal` | `1494.66` | `42.82` | `23.35` | `1.000x` | `64` | `0 / 0` | `0.0%` | Exact |
| | `lookup` | `987.02` | `64.84` | `15.42` | `1.514x` | `14` | `50 / 50` | `100.0%` | Exact |
| | **`adaptive`** | **`986.80`** | **`64.86`** | **`15.42`** | **`1.515x`** | `14` | `50 / 50` | `100.0%` | Exact |
| **`32,704`** | `normal` | `1739.28` | `36.80` | `27.18` | `1.000x` | `64` | `0 / 0` | `0.0%` | Exact |
| | `lookup` | `1155.24` | `55.40` | `18.05` | `1.506x` | `14` | `50 / 50` | `100.0%` | Exact |
| | **`adaptive`** | **`1157.03`** | **`55.31`** | **`18.08`** | **`1.503x`** | `14` | `50 / 50` | `100.0%` | Exact |

---

### 5.3 Qwen3.8-Flash-Next (`qwen3.8-flash-next`) — Workloads `W1`–`W9` across `normal`, `mtp`, `lookup`, `retrieval`, `adaptive`

All runs executed with `GUILD_IQ_MT_MIN=1`, `--gexec qwen-65k-s0.gexec` / `qwen-65k-s2.gexec` (`gexec_active: true`), and verified for **100% bit-for-bit greedy token ID parity** across all strategies.

| Workload Category | Prompt / Gen | Strategy | Decode Time (ms) | Decode (tok/s) | Cost (ms/tok) | Speedup vs Normal | Verify Windows | Drafts Off / Acc | Acc Rate | Windows (`MTP / Lkp / Ret`) | Final Strategy |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **W1: Code Editing / Patching** | `256 / 64` | `normal` | `3474.97` | `18.42` | `54.30` | `1.000x` | `64` | `0 / 0` | `0.0%` | `0 / 0 / 0` | `normal` |
| | `256 / 64` | `mtp` | `2518.29` | `25.41` | `39.35` | `1.380x` | `34` | `33 / 30` | `90.9%` | `33 / 0 / 0` | `mtp` |
| | `256 / 64` | `lookup` | `3586.25` | `17.85` | `56.04` | `0.969x` | `57` | `24 / 7` | `29.2%` | `0 / 7 / 0` | `lookup` |
| | `256 / 64` | **`adaptive`** | **`2520.70`** | **`25.39`** | **`39.39`** | **`1.379x`** | `34` | `33 / 30` | **`90.9%`** | **`33 / 0 / 0`** | **`mtp`** |
| **W2: Document Summarization** | `256 / 64` | `normal` | `3499.78` | `18.29` | `54.68` | `1.000x` | `64` | `0 / 0` | `0.0%` | `0 / 0 / 0` | `normal` |
| | `256 / 64` | `mtp` | `2488.68` | `25.72` | `38.89` | `1.406x` | `33` | `31 / 31` | `100.0%` | `32 / 0 / 0` | `mtp` |
| | `256 / 64` | `lookup` | `1856.40` | `34.48` | `29.01` | `1.885x` | `14` | `50 / 50` | `100.0%` | `0 / 13 / 0` | `lookup` |
| | `256 / 64` | **`adaptive`** | **`1899.06`** | **`33.70`** | **`29.67`** | **`1.843x`** | `14` | `50 / 50` | **`100.0%`** | **`0 / 13 / 0`** | **`lookup`** |
| **W3: JSON Schema Output** | `256 / 64` | `normal` | `3474.74` | `18.42` | `54.29` | `1.000x` | `64` | `0 / 0` | `0.0%` | `0 / 0 / 0` | `normal` |
| | `256 / 64` | `mtp` | `2734.87` | `23.40` | `42.73` | `1.270x` | `39` | `31 / 25` | `80.6%` | `32 / 0 / 0` | `mtp` |
| | `256 / 64` | `lookup` | `3678.95` | `17.40` | `57.48` | `0.945x` | `64` | `9 / 0` | `0.0%` | `0 / 4 / 0` | `lookup` |
| | `256 / 64` | **`adaptive`** | **`2955.11`** | **`21.66`** | **`46.17`** | **`1.176x`** | `47` | `21 / 17` | **`81.0%`** | **`21 / 1 / 0`** | `lookup` |
| **W4: RAG Grounded Q&A** | `256 / 64` | `normal` | `3490.15` | `18.34` | `54.53` | `1.000x` | `64` | `0 / 0` | `0.0%` | `0 / 0 / 0` | `normal` |
| *(+ session corpus)* | `256 / 64` | `mtp` | `2647.28` | `24.18` | `41.36` | `1.318x` | `36` | `30 / 28` | `93.3%` | `31 / 0 / 0` | `mtp` |
| | `256 / 64` | `lookup` | `1944.22` | `32.92` | `30.38` | `1.795x` | `17` | `48 / 47` | `97.9%` | `0 / 13 / 0` | `lookup` |
| | `256 / 64` | `retrieval` | `1946.09` | `32.89` | `30.41` | `1.793x` | `17` | `48 / 47` | `97.9%` | `0 / 0 / 13` | `retrieval` |
| | `256 / 64` | **`adaptive`** | **`1998.67`** | **`32.02`** | **`31.23`** | **`1.746x`** | `17` | `50 / 47` | **`94.0%`** | **`2 / 13 / 0`** | **`lookup`** |
| **W5: Multi-Turn Conversation** | `128 / 64` | `normal` | `3484.76` | `18.37` | `54.45` | `1.000x` | `64` | `0 / 0` | `0.0%` | `0 / 0 / 0` | `normal` |
| *(+ Turn 1 in session corpus)* | `128 / 64` | `mtp` | `2798.28` | `22.87` | `43.72` | `1.245x` | `41` | `26 / 23` | `88.5%` | `27 / 0 / 0` | `mtp` |
| | `128 / 64` | `lookup` | `3110.02` | `20.58` | `48.59` | `1.120x` | `45` | `30 / 19` | `63.3%` | `0 / 8 / 0` | `lookup` |
| | `128 / 64` | `retrieval` | `3204.44` | `19.97` | `50.07` | `1.087x` | `45` | `34 / 19` | `55.9%` | `0 / 0 / 9` | `retrieval` |
| | `128 / 64` | **`adaptive`** | **`2737.64`** | **`23.38`** | **`42.78`** | **`1.273x`** | `36` | `35 / 28` | **`80.0%`** | **`18 / 5 / 1`** | **`lookup`** |
| **W6: Repetitive Boilerplate** | `256 / 64` | `normal` | `3493.57` | `18.32` | `54.59` | `1.000x` | `64` | `0 / 0` | `0.0%` | `0 / 0 / 0` | `normal` |
| | `256 / 64` | `mtp` | `2544.25` | `25.15` | `39.75` | `1.373x` | `34` | `30 / 30` | `100.0%` | `31 / 0 / 0` | `mtp` |
| | `256 / 64` | `lookup` | `1917.91` | `33.37` | `29.97` | `1.822x` | `16` | `48 / 48` | `100.0%` | `0 / 12 / 0` | `lookup` |
| | `256 / 64` | **`adaptive`** | **`1920.58`** | **`33.32`** | **`30.01`** | **`1.819x`** | `16` | `48 / 48` | **`100.0%`** | **`3 / 12 / 0`** | **`lookup`** |
| **W7: Creative Novel Prose** | `256 / 64` | `normal` | `3467.63` | `18.46` | `54.18` | `1.000x` | `64` | `0 / 0` | `0.0%` | `0 / 0 / 0` | `normal` |
| *(hostile to lookup)* | `256 / 64` | `mtp` | `2855.59` | `22.41` | `44.62` | `1.214x` | `41` | `30 / 23` | `76.7%` | `31 / 0 / 0` | `mtp` |
| | `256 / 64` | `lookup` | `3579.51` | `17.88` | `55.93` | `0.969x` | `64` | `4 / 0` | `0.0%` | `0 / 1 / 0` | `normal` |
| | `256 / 64` | **`adaptive`** | **`2889.47`** | **`22.15`** | **`45.15`** | **`1.200x`** | `42` | `29 / 22` | **`75.9%`** | **`30 / 0 / 0`** | **`mtp`** |
| **W8: Mathematical Reasoning** | `256 / 64` | `normal` | `3466.69` | `18.46` | `54.17` | `1.000x` | `64` | `0 / 0` | `0.0%` | `0 / 0 / 0` | `normal` |
| *(hostile to lookup)* | `256 / 64` | `mtp` | `2675.67` | `23.92` | `41.81` | `1.296x` | `37` | `32 / 27` | `84.4%` | `32 / 0 / 0` | `mtp` |
| | `256 / 64` | `lookup` | `4021.79` | `15.91` | `62.84` | `0.862x` | `61` | `36 / 3` | `8.3%` | `0 / 9 / 0` | `normal` |
| | `256 / 64` | **`adaptive`** | **`2632.25`** | **`24.31`** | **`41.13`** | **`1.317x`** | `37` | `30 / 27` | **`90.0%`** | **`30 / 0 / 0`** | **`mtp`** |
| **W9: Short Interactive Chat** | `64 / 64` | `normal` | `3475.46` | `18.41` | `54.30` | `1.000x` | `64` | `0 / 0` | `0.0%` | `0 / 0 / 0` | `normal` |
| | `64 / 64` | `mtp` | `2750.53` | `23.27` | `42.98` | `1.264x` | `39` | `31 / 25` | `80.6%` | `31 / 0 / 0` | `mtp` |
| | `64 / 64` | `lookup` | `3463.81` | `18.48` | `54.12` | `1.003x` | `59` | `12 / 5` | `41.7%` | `0 / 3 / 0` | `normal` |
| | `64 / 64` | **`adaptive`** | **`3438.91`** | **`18.61`** | **`53.73`** | **`1.011x`** | `61` | `6 / 3` | `50.0%` | `2 / 4 / 0` | `normal` |

### 5.4 Qwen3.8-Flash-Next (`qwen3.8-flash-next`) — W10 Populated-Context Scaling (`512` to `65,472` Tokens)

| Populated Context | Strategy | Decode Time (ms) | Decode (tok/s) | Cost (ms/tok) | Speedup vs Normal | Verify Windows | Drafts Off / Acc | Acc Rate | Windows (`MTP / Lkp`) | Parity |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`512`** | `normal` | `3467.43` | `18.46` | `54.18` | `1.000x` | `64` | `0 / 0` | `0.0%` | `0 / 0` | Exact |
| | `mtp` | `2751.03` | `23.26` | `42.98` | `1.260x` | `38` | `34 / 26` | `76.5%` | `35 / 0` | Exact |
| | `lookup` | `3233.21` | `19.79` | `50.52` | `1.072x` | `55` | `12 / 9` | `75.0%` | `0 / 3` | Exact |
| | **`adaptive`** | **`2950.32`** | **`21.69`** | **`46.10`** | **`1.175x`** | `42` | `33 / 22` | `66.7%` | `23 / 4` | Exact |
| **`4,096`** | `normal` | `3549.87` | `18.03` | `55.47` | `1.000x` | `64` | `0 / 0` | `0.0%` | `0 / 0` | Exact |
| | `mtp` | `2780.78` | `23.02` | `43.45` | `1.277x` | `37` | `33 / 27` | `81.8%` | `34 / 0` | Exact |
| | `lookup` | `3044.17` | `21.02` | `47.57` | `1.166x` | `43` | `28 / 21` | `75.0%` | `0 / 7` | Exact |
| | **`adaptive`** | **`2980.65`** | **`21.47`** | **`46.57`** | **`1.191x`** | `42` | `29 / 22` | `75.9%` | `4 / 7` | Exact |
| **`8,192`** | `normal` | `3555.53` | `18.00` | `55.56` | `1.000x` | `64` | `0 / 0` | `0.0%` | `0 / 0` | Exact |
| | `mtp` | `2632.61` | `24.31` | `41.13` | `1.351x` | `36` | `30 / 28` | `93.3%` | `31 / 0` | Exact |
| | `lookup` | `3460.94` | `18.49` | `54.08` | `1.027x` | `59` | `8 / 5` | `62.5%` | `0 / 2` | Exact |
| | **`adaptive`** | **`2710.80`** | **`23.61`** | **`42.36`** | **`1.312x`** | `36` | `33 / 28` | `84.8%` | `30 / 1` | Exact |
| **`16,384`** | `normal` | `3572.02` | `17.92` | `55.81` | `1.000x` | `64` | `0 / 0` | `0.0%` | `0 / 0` | Exact |
| | `mtp` | `2673.75` | `23.94` | `41.78` | `1.336x` | `36` | `30 / 28` | `93.3%` | `30 / 0` | Exact |
| | `lookup` | `3130.96` | `20.44` | `48.92` | `1.141x` | `47` | `21 / 17` | `81.0%` | `0 / 6` | Exact |
| | **`adaptive`** | **`2612.42`** | **`24.50`** | **`40.82`** | **`1.367x`** | `31` | `39 / 33` | `84.6%` | **`23 / 4`** | Exact |
| **`32,768`** | `normal` | `3619.81` | `17.68` | `56.56` | `1.000x` | `64` | `0 / 0` | `0.0%` | `0 / 0` | Exact |
| | `mtp` | `2748.54` | `23.29` | `42.95` | `1.317x` | `36` | `28 / 28` | `100.0%` | `28 / 0` | Exact |
| | `lookup` | `1992.70` | `32.12` | `31.14` | `1.817x` | `14` | `50 / 50` | `100.0%` | `0 / 13` | Exact |
| | **`adaptive`** | **`1987.68`** | **`32.20`** | **`31.06`** | **`1.821x`** | `14` | `50 / 50` | `100.0%` | **`0 / 13`** | Exact |
| **`65,472`** | `normal` | `3658.48` | `17.49` | `57.16` | `1.000x` | `64` | `0 / 0` | `0.0%` | `0 / 0` | Exact |
| | `mtp` | `2644.83` | `24.20` | `41.33` | `1.383x` | `33` | `31 / 31` | `100.0%` | `31 / 0` | Exact |
| | `lookup` | `2017.05` | `31.73` | `31.52` | `1.814x` | `14` | `50 / 50` | `100.0%` | `0 / 13` | Exact |
| | **`adaptive`** | **`2013.94`** | **`31.78`** | **`31.47`** | **`1.817x`** | `14` | `50 / 50` | `100.0%` | **`0 / 13`** | Exact |

---

## 6. Analysis of Negative / Neutral Cases & Non-Negotiable Acceptance Gates

1. **Why Fixed Prompt-Lookup Regresses on `W7` (Creative Prose) and `W8` (Mathematical Reasoning)**:
   - On `W8` (`qwen3.8-flash-next`), mathematical text contains frequent 3-token structural phrases (`" = 1 + "`, `" for real x "`) that match earlier equations in the prompt, causing fixed `lookup` (`k=4`) to propose `36` draft tokens across `9` four-token verification windows (`T=5`), of which only `3` tokens (`8.3%`) are accepted.
   - Because each `T=5` window on Qwen3.8-Flash-Next activates `~3.32x` as many distinct CPU experts per layer as `T=1`, those `9` failed `T=5` windows cost `+555 ms` of wasted CPU MoE verification, dropping throughput from `18.46 tok/s` (`normal`) to `15.91 tok/s` (`-13.8%` regression).
   - Similarly, on Ornith 35B `W8`, fixed `lookup` (`k=4`) accepts only `41.1%` (`30 / 73`), dropping throughput from `50.15 tok/s` (`normal`) to `47.91 tok/s` (`-4.5%` regression).
2. **How `AdaptiveDecodeScheduler` Eliminates Those Regressions**:
   - On Ornith 35B (`W7` and `W8`, where no MTP layer exists), `adaptive` ignores isolated single-occurrence 3-grams (`kPriorLookupQ[0] = 0.32`), limits initial probes on longer matches, and backs off to `normal` after `2` consecutive rejections (`fallback_to_normal_count`), finishing `W7` at `49.53 tok/s` (vs `49.55 tok/s` `normal`) and `W8` at `50.01 tok/s` (vs `50.15 tok/s` `normal`).
   - On Qwen3.8-Flash-Next (`W7` and `W8`, where MTP is available), `adaptive` evaluates $C(\text{Mtp}, 1) \approx 41.1\text{ ms/tok}$ against $C(\text{Lookup})$ and $C(\text{Normal}) \approx 54.2\text{ ms/tok}$, selecting `Mtp` for `30` windows (`0` `Lookup` windows) and achieving `22.15 tok/s` (`+20.0%` over `normal`) on `W7` and `24.31 tok/s` (`+31.7%` over `normal`, `+52.8%` over fixed `lookup`) on `W8`.

---

## 7. Verification Gates & Remaining Limitations

- **CPU Build & CTest (`build-cpu`)**: `33 / 33` tests passed (`100%`).
- **CUDA Build & CTest (`build-cuda12`)**: `97 / 97` runnable tests with present fixtures passed (`0` regressions; `adaptive_scheduler_test`, `suffix_drafter_test`, `controller_test`, `draft_policy_test`, `coupled_draft_test`, `gexec_contract_test`, `gexec_optimizer_test`, `ornith_step_test`, `ornith_prefill_test` all passed).
- **Sanitizers**:
  - `build-asan` (`-fsanitize=address,undefined`): `0` first-party errors across all Phase 6.6 test binaries.
  - `build-tsan` (`-fsanitize=thread`): `0` data races across all Phase 6.6 test binaries (including 4-thread concurrent `RetrievalDrafter` session isolation stress tests).
- **Remaining Limitations**:
  - Secondary workstation validation (AMD Ryzen 5 5600X + NVIDIA RTX 5070) remains pending physical hardware access.
  - On Qwen3.8-Flash-Next, exact bit-for-bit greedy parity across different verification window sizes ($T=1$ vs $T \ge 2$) requires `GUILD_IQ_MT_MIN=1` so `IQ3_S`/`IQ2_S`/`IQ4_NL` CPU expert kernels use the same SIMD reduction tree at `nt == 1` and `nt >= 2`.

---

## 8. Phase 6.6.1: Hardware-Calibrated Adaptive Decode Optimization

Following Phase 6.6 (`ed7a5db`), we executed a focused, measurement-driven optimization pass on the adaptive decoding scheduler, GEXEC hardware calibration, and MTP catch-up path.

### 8.1 Audit of Performance-Sensitive Constants & Heuristics

Every parameter in `AdaptiveDecodeScheduler`, `SuffixDrafter`, `RetrievalDrafter`, `GExecSpecCalibration`, and `GuildSession::generate` was audited and classified into:
- **A**: Correctness, ABI, or bounded-$O(1)$ memory/latency safety invariant.
- **B**: Explicit user-configurable policy.
- **C**: Hardware-dependent performance estimate (measured in `guild optimize` and refined online).
- **D**: Workload-dependent statistical estimate (learned online per session via Beta-Binomial posteriors).
- **E**: Unnecessary or poorly justified heuristic (replaced or eliminated).

| Parameter / Constant | Phase 6.6 Value | Category | Phase 6.6.1 Treatment & Justification |
| :--- | :--- | :---: | :--- |
| `kVerifyMaxT = 8`, `kMaxDraft = 7` | `8` / `7` | **A** | **Retained**: Fixed CUDA graph and device verification buffer capacity bound. |
| `SuffixDrafter::WAYS = 4`, `RetrievalDrafter::WAYS = 8`, `kMaxProbe = 16`, `kMaxCandidates = 16` | `4, 8, 16, 16` | **A** | **Retained**: Guarantees zero heap allocations and $O(1)$ worst-case lookup latency (`< 0.005 ms`) on the token path. |
| `mode`, `max_draft`, `spec_lookup_window`, `spec_cost_window`, `mtp_min_prob` | CLI defaults | **B** | **Retained**: Explicit CLI and per-request user controls (`--decode-strategy`, `--spec-max-draft`, etc.). |
| `verify_window_ms[1..8]`, `verify_cpu_expert_ms[1..8]` | Analytic formula with fixed `0.22` `nt=2` multiplier and `pool_nt=2` inflation when `--spec 2` | **C** | **Measured in `guild optimize` + decomposed online**: `calibrate_speculative_costs` now benchmarks the host `ExpertPool` on the model's real expert layer at `nt=1`, `nt=2` (`alpha_nt2`), and `2 * k_active` (`pool_2k_scale`). At runtime, `seed_single_token_verify(verify_ms, experts, cpu_expert_ms)` measures the live CPU-vs-GPU verification split (`live_cpu_fraction_`) on window 0. |
| `concurrency_slope` | Fixed `0.15` per extra session | **C** | **Measured in `guild optimize`**: Derived from measured `ExpertPool` oversubscription (`2 * k_active` vs `k_active` task dispatch latency) and persisted in `.gexec` (`+2.0%` per extra session on 32C Xeon). |
| ` hysteresis_margin = 0.05` | Fixed `5%` | **C / D** | **Calibrated from hardware timing noise CV**: `guild optimize` records IQR/median dispersion `verify_noise_cv` in `.gexec`, and `AdaptiveDecodeScheduler` updates `online_verify_cv_` from live verification residuals (`effective_hysteresis_margin = clamp(1.25 * online_verify_cv_, 0.018, 0.10)`). |
| `rej_penalty = std::pow(0.68, rej)` and `reexplore_interval = 24` | Fixed `0.68^rej` and `24`-step exile | **E** | **Replaced with sample-size-aware Beta-Binomial posterior**: Tracks decayed successes $S_i$ and trials $N_i$ with virtual failure weight $0.55\,r(r+1)$ in the denominator ($\hat{p}_i = S_i / (N_i + 0.55\,r(r+1))$). Cold-start drafters ($N_0 \approx 3$) back off after 2 misses, while established high-acceptance drafters ($N_0 \approx 20$, e.g. `Qwen W3` at `80.6%`) survive normal Bernoulli misses without 24-step exile. |
| Coarse 4-bucket lookup table (`match_bucket`) | Ignored candidate agreement ratio `freq / candidates` | **D / E** | **Upgraded to continuous match-length + consensus-prefix scoring**: `SuffixDrafter` now computes `last_candidates()` (among equal-length peers `cand_len[j] == best_len`) and `last_consensus_k()`, flagging variable counter slots (`3/3` divergent continuations $\implies \text{consensus\_k} = 0$) while immediately boosting unanimous template spans (`freq >= 2, agreement == 1.0`). |
| `MtpDrafter::prefill` E-4 branch on decode catch-up (`n <= max_t_`) | Entered multi-group heap `std::vector` + `cudaFree`/`cudaMalloc` path for `n = 1..5` | **E** | **Eliminated for `n <= max_t_`**: Changed gate in `src/core/mtp.cpp` to `if (!per_group_sync && n > max_t_)` so decode-time MTP KV catch-up uses the zero-allocation mapped-memory graph `prefill_exec_[T]` and excludes one-time graph capture from `ms_prefill`. |

---

### 8.2 Root-Cause Resolution of Targeted Phase 6.6 Gaps

1. **Qwen W3 (JSON Schema): `21.66 tok/s` (Phase 6.6) $\rightarrow$ `23.52 tok/s` (Phase 6.6.1), matching/beating Fixed MTP (`23.43 tok/s`)**:
   - **Root Cause**: In Phase 6.6, `MTP` achieved `80.6%` (`25/31`) acceptance on `W3`, but hit two consecutive Bernoulli rejections mid-stream (which occurs with probability $1 - (1 - 0.194^2)^{31} = 69.5\%$ over 32 windows). Multiplying by `0.68^2 = 0.46` and enforcing `reexplore_interval = 24` locked `Adaptive` in `Normal` for `25` windows (`+220 ms`).
   - **Fix**: Sample-size-aware Beta-Binomial posterior (`mtp_succ_[0] / (mtp_trials_[0] + 0.55 * r * (r + 1))`) + `SuffixDrafter` counter-slot consensus filtering (`last_consensus_k() == 0` after `{"record_id": `). `Adaptive` now executes all `32` MTP windows (`0` wasted Lookup windows), improving throughput from `21.66 tok/s` (`46.17 ms/tok`) to **`23.52 tok/s` (`42.52 ms/tok`, `+8.6%`)**.
2. **Qwen W2 (Summarization) & W4 (RAG Q&A)**:
   - **Root Cause**: During `Lookup(k=4)` windows (`a + 1 = 5`), `MtpDrafter::prefill` entered the multi-group prompt-prefill path (`n > 0`), allocating a heap `std::vector`, reallocating `pf_dev_` via `cudaFree`/`cudaMalloc`, and capturing `prefill_dev_exec_[5]` inside the timed decode loop.
   - **Fix**: Routing `n <= max_t_` to the mapped-memory `prefill_exec_[T]` graph in `src/core/mtp.cpp` improved `Qwen W2` `Adaptive` from `33.70 tok/s` to **`34.16`–`34.49 tok/s`** and `Qwen W4` `Adaptive` from `32.02 tok/s` to **`32.15`–`32.37 tok/s`**.
3. **Qwen W9 (Short Interactive Chat `p=64, g=64`): `18.61 tok/s` (Phase 6.6) $\rightarrow$ `22.07 tok/s` (Phase 6.6.1, `+18.6%`)**:
   - **Root Cause**: Early misses in short interactive responses triggered the 24-step `Normal` lock before `g=64` completed.
   - **Fix**: Scaling neural MTP re-exploration as `clamp(2 * r, 2, 6)` allows rapid recovery after opening-token uncertainty, lifting `W9` `Adaptive` from `18.61 tok/s` to **`22.07 tok/s` (`45.30 ms/tok`)**.

---

### 8.3 Controlled Phase 6.6 vs Phase 6.6.1 Benchmark Comparison

#### A. Ornith 1.5 35B (`ornith:35b`, GEXEC Calibrated — `100%` Bitwise Greedy Parity)

| Workload | Strategy | Phase 6.6 (tok/s) | **Phase 6.6.1 (tok/s)** | **Phase 6.6.1 (ms/tok)** | Verify Windows | Drafts Off / Acc | Acc Rate | Mean Cost Pred Err (ms/tok) | Switches / Fallbacks |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **W1: Code Editing** | `normal` | `50.29` | `50.40` | `19.84` | `128` | `0 / 0` | `0.0%` | `0.08` | `0 / 0` |
| | `lookup` | `50.91` | `51.28` | `19.50` | `86` | `79 / 42` | `53.2%` | `3.33` | `21 / 10` |
| | **`adaptive`** | `52.98` | **`53.58`** | **`18.66`** | `99` | `37 / 29` | **`78.4%`** | **`1.13`** | `13 / 6` |
| **W2: Summarization** | `normal` | `50.16` | `50.27` | `19.89` | `128` | `0 / 0` | `0.0%` | `0.07` | `0 / 0` |
| | `lookup` | `81.93` | `82.19` | `12.17` | `27` | `104 / 101` | `97.1%` | `0.85` | `0 / 0` |
| | **`adaptive`** | `80.65` | **`80.08`–`80.84`** | **`12.37`** | `28` | `100 / 100` | **`100.0%`** | **`0.65`** | `0 / 0` |
| **W3: JSON Schema** | `normal` | `50.05` | `50.22` | `19.91` | `128` | `0 / 0` | `0.0%` | `0.09` | `0 / 0` |
| | `lookup` | `76.87` | `77.12` | `12.97` | `31` | `104 / 97` | `93.3%` | `2.69` | `3 / 1` |
| | **`adaptive`** | `75.54` | **`75.69`** | **`13.21`** | `33` | `99 / 95` | **`96.0%`** | **`2.05`** | `3 / 1` |
| **W4: RAG Q&A** | `normal` | `49.93` | `50.19` | `19.93` | `128` | `0 / 0` | `0.0%` | `0.07` | `0 / 0` |
| | `lookup` | `82.32` | `82.65` | `12.10` | `29` | `100 / 99` | `99.0%` | `2.10` | `2 / 1` |
| | `retrieval` | `82.43` | `82.47` | `12.13` | `29` | `100 / 99` | `99.0%` | `1.98` | `2 / 1` |
| | **`adaptive`** | `77.75` | **`77.98`** | **`12.82`** | `32` | `99 / 96` | **`97.0%`** | **`1.56`** | `2 / 1` |
| **W5: Multi-Turn** | `normal` | `50.02` | `50.03` | `19.99` | `128` | `0 / 0` | `0.0%` | `0.10` | `0 / 0` |
| | `lookup` | `68.27` | `68.32` | `14.64` | `40` | `107 / 88` | `82.2%` | `5.62` | `9 / 4` |
| | `retrieval` | `68.38` | `68.31` | `14.64` | `40` | `107 / 88` | `82.2%` | `5.75` | `9 / 4` |
| | **`adaptive`** | `69.79` | **`70.02`–`71.43`** | **`14.00`** | `43` | `91 / 85` | **`93.4%`** | **`2.12`** | `9 / 4` |
| **W6: Boilerplate** | `normal` | `50.08` | `50.13` | `19.95` | `128` | `0 / 0` | `0.0%` | `0.08` | `0 / 0` |
| | `lookup` | `76.99` | `77.17` | `12.96` | `34` | `94 / 94` | `100.0%` | `0.55` | `6 / 3` |
| | **`adaptive`** | `74.18` | **`74.11`–`74.50`** | **`13.42`** | `39` | `89 / 89` | **`100.0%`** | **`0.65`** | `6 / 3` |
| **W7: Creative Novel** | `normal` | `49.55` | `49.75` | `20.10` | `128` | `0 / 0` | `0.0%` | `0.30` | `0 / 0` |
| | `lookup` | `48.84` | `48.84` | `20.48` | `127` | `8 / 1` | `12.5%` | `0.48` | `4 / 2` |
| | **`adaptive`** | `49.53` | **`49.10`–`49.18`** | **`20.33`** | `128` | `3 / 0` | `0.0%` | **`0.31`** | `4 / 2` |
| **W8: Math Reasoning** | `normal` | `50.15` | `50.28` | `19.89` | `128` | `0 / 0` | `0.0%` | `0.07` | `0 / 0` |
| | `lookup` | `47.91` | `48.11` | `20.78` | `98` | `73 / 30` | `41.1%` | `3.96` | `29 / 14` |
| | **`adaptive`** | `50.01` | **`49.42`–`50.41`** | **`19.84`** | `110` | `29 / 18` | `62.1%` | **`1.33`** | `19 / 9` |

#### B. Qwen3.8-Flash-Next (`qwen3.8-flash-next`, GEXEC Calibrated — `100%` Bitwise Greedy Parity)

| Workload | Strategy | Phase 6.6 (tok/s) | **Phase 6.6.1 (tok/s)** | **Phase 6.6.1 (ms/tok)** | Delta vs 6.6 | Verify Windows | Drafts Off / Acc | Acc Rate | Windows (`MTP / Lkp / Ret`) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **W1: Code Editing** | `normal` | `18.42` | `18.46` | `54.17` | `+0.2%` | `64` | `0 / 0` | `0.0%` | `0 / 0 / 0` |
| | `mtp` | `25.41` | `25.30` | `39.52` | `-0.4%` | `34` | `33 / 30` | `90.9%` | `33 / 0 / 0` |
| | `lookup` | `17.85` | `17.75` | `56.32` | `-0.6%` | `57` | `24 / 7` | `29.2%` | `0 / 7 / 0` |
| | **`adaptive`** | `25.39` | **`25.40`–`25.92`** | **`38.58`–`39.37`** | **`+0.0%` to `+2.1%`** | `32` | `35 / 32` | **`91.4%`** | **`29 / 2 / 0`** |
| **W2: Summarization** | `normal` | `18.29` | `18.42` | `54.29` | `+0.7%` | `64` | `0 / 0` | `0.0%` | `0 / 0 / 0` |
| | `mtp` | `25.72` | `25.37` | `39.42` | `-1.4%` | `33` | `31 / 31` | `100.0%` | `32 / 0 / 0` |
| | `lookup` | `34.48` | `34.42` | `29.05` | `-0.2%` | `14` | `50 / 50` | `100.0%` | `0 / 13 / 0` |
| | **`adaptive`** | `33.70` | **`34.16`–`34.49`** | **`28.99`–`29.28`** | **`+1.4%` to `+2.3%`** | `14` | `50 / 50` | **`100.0%`** | **`0 / 13 / 0`** |
| **W3: JSON Schema** | `normal` | `18.42` | `18.33` | `54.56` | `-0.5%` | `64` | `0 / 0` | `0.0%` | `0 / 0 / 0` |
| | `mtp` | `23.40` | `23.43` | `42.68` | `+0.1%` | `39` | `31 / 25` | `80.6%` | `32 / 0 / 0` |
| | `lookup` | `17.40` | `17.39` | `57.51` | `-0.1%` | `64` | `9 / 0` | `0.0%` | `0 / 4 / 0` |
| | **`adaptive`** | `21.66` | **`23.52`** | **`42.52`** | **`+8.6%`** | `39` | `31 / 25` | **`80.6%`** | **`32 / 0 / 0`** |
| **W4: RAG Q&A** | `normal` | `18.34` | `18.32` | `54.58` | `-0.1%` | `64` | `0 / 0` | `0.0%` | `0 / 0 / 0` |
| | `mtp` | `24.18` | `24.09` | `41.50` | `-0.4%` | `36` | `30 / 28` | `93.3%` | `31 / 0 / 0` |
| | `lookup` | `32.92` | `32.81` | `30.48` | `-0.3%` | `17` | `48 / 47` | `97.9%` | `0 / 13 / 0` |
| | **`adaptive`** | `32.02` | **`32.15`–`32.37`** | **`30.90`–`31.10`** | **`+0.4%` to `+1.1%`** | `17` | `49 / 47` | **`95.9%`** | **`2 / 13 / 0`** |
| **W7: Creative Novel** | `normal` | `18.46` | `18.30` | `54.66` | `-0.9%` | `64` | `0 / 0` | `0.0%` | `0 / 0 / 0` |
| | `mtp` | `22.41` | `22.47` | `44.51` | `+0.3%` | `41` | `30 / 23` | `76.7%` | `31 / 0 / 0` |
| | `lookup` | `17.88` | `17.84` | `56.05` | `-0.2%` | `64` | `4 / 0` | `0.0%` | `0 / 1 / 0` |
| | **`adaptive`** | `22.15` | **`22.43`** | **`44.58`** | **`+1.3%`** | `41` | `30 / 23` | **`76.7%`** | **`31 / 0 / 0`** |
| **W8: Math Reasoning** | `normal` | `18.46` | `18.40` | `54.36` | `-0.3%` | `64` | `0 / 0` | `0.0%` | `0 / 0 / 0` |
| | `mtp` | `23.92` | `23.94` | `41.76` | `+0.1%` | `37` | `32 / 27` | `84.4%` | `32 / 0 / 0` |
| | `lookup` | `15.91` | `15.87` | `63.03` | `-0.3%` | `61` | `36 / 3` | `8.3%` | `0 / 9 / 0` |
| | **`adaptive`** | `24.31` | **`23.68`** | **`42.23`** | `-2.6%` (`+49.2%` vs `lookup`) | `37` | `33 / 27` | `81.8%` | `31 / 1 / 0` |
| **W9: Short Chat** | `normal` | `18.41` | `18.27` | `54.74` | `-0.8%` | `64` | `0 / 0` | `0.0%` | `0 / 0 / 0` |
| | `mtp` | `23.27` | `23.24` | `43.03` | `-0.1%` | `39` | `31 / 25` | `80.6%` | `31 / 0 / 0` |
| | `lookup` | `18.48` | `18.42` | `54.30` | `-0.3%` | `59` | `12 / 5` | `41.7%` | `0 / 3 / 0` |
| | **`adaptive`** | `18.61` | **`22.07`** | **`45.30`** | **`+18.6%`** | `41` | `31 / 23` | **`74.2%`** | **`24 / 2 / 0`** |

---

### 8.4 Held-Out Validation Workload Suite (`H1`–`H4`, Unseen During Tuning)

To verify that Phase 6.6.1 generalizes without overfitting to `W1`–`W10`, we evaluated 4 held-out prompts (`H1: SQL Migration View`, `H2: Linux Kernel/Daemon Syslog Extraction`, `H3: Deep-Space Sci-Fi Dialogue`, `H4: 2-Sphere Riemann Curvature Tensor Derivation`):

| Model & Held-Out Workload | `normal` (tok/s) | `mtp` (tok/s) | `lookup` (tok/s) | **`adaptive` (tok/s)** | **`adaptive` vs `normal`** | **`adaptive` vs `lookup`** | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Ornith `H1`: SQL Migration** | `50.14` | — | `47.02` (`-6.2%`) | **`51.92`** | **`+3.6%`** | **`+10.4%`** | Beats both `normal` and `lookup` |
| **Ornith `H2`: Syslog Extraction** | `49.54` | — | `65.68` | **`63.86`** | **`+28.9%`** | `-2.8%` | `87.3%` draft acceptance |
| **Ornith `H3`: Sci-Fi Dialogue** | `50.15` | — | `51.48` | **`50.77`** | **`+1.2%`** | `-1.4%` | Beats `normal` on novel dialogue |
| **Ornith `H4`: Riemann Tensor** | `50.16` | — | `42.07` (`-16.1%`) | **`47.92`** | `-4.5%` | **`+13.9%`** | Eliminates `lookup`'s `-16.1%` crash |
| **Qwen `H1`: SQL Migration** | `18.42` | `25.66` | `18.33` | **`24.38`** | **`+32.4%`** | **`+33.0%`** | `84.2%` acceptance (`27` MTP + `4` Lkp) |
| **Qwen `H2`: Syslog Extraction** | `18.36` | `23.43` | `21.44` | **`23.24`** | **`+26.6%`** | **`+8.4%`** | `83.9%` acceptance (`13` MTP + `7` Lkp) |
| **Qwen `H3`: Sci-Fi Dialogue** | `18.41` | `20.82` | `18.29` | **`20.08`** | **`+9.1%`** | **`+9.8%`** | Selects `MTP` (`20` MTP, `0` Lkp) |
| **Qwen `H4`: Riemann Tensor** | `18.38` | `24.85` | `21.68` | **`25.99`** | **`+41.4%`** | **`+19.9%`** | **Beats `normal`, `lookup`, AND `mtp` (`+4.6%`)** |

---

### 8.5 Reproducible Calibration & Validation Instructions for the Secondary Desktop (`Ryzen 5 5600X + RTX 5070`)

When running on the secondary workstation (`AMD Ryzen 5 5600X 6C/12T AVX2, 32 GiB DDR4 + NVIDIA RTX 5070 Founders Edition 12 GiB GDDR7, sm_120`), server `.gexec` files are automatically rejected by `GExecArtifactIdentity::validate` (`CpuIsaMismatch` / `NumaTopologyMismatch` / `GpuArchitectureMismatch`) and regenerated locally:

```bash
# 1. Configure & build for RTX 5070 (Blackwell sm_120) + Ryzen 5 5600X (AVX2)
cmake -S . -B build -DGUILD_ENABLE_CUDA=ON -DGUILD_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j12

# 2. Generate hardware-calibrated GEXEC artifacts on the desktop (measures Ryzen 5600X AVX2 ExpertPool T=1..8,
#    6-core concurrency contention slope, lookup/retrieval latency, and timing noise CV)
./build/guild optimize <path/to/model.gpack> --mode auto --context 32768 --prefill-batch 512 --spec 0 -o desktop-s0.gexec
./build/guild inspect desktop-s0.gexec --plan

# 3. Run adaptive decoding with live GPU+CPU T=1 split seeding
./build/guild run <model> --gexec desktop-s0.gexec --decode-strategy adaptive --spec-max-draft 4
```
