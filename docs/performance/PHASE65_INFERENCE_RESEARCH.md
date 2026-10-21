# Phase 6.5 inference research and experiments

Work log started 2026-10-20 at `98b2e5e`. Status: **completed** (see
[`PHASE65_GENERAL_OPTIMIZATION.md`](PHASE65_GENERAL_OPTIMIZATION.md) for the full
Phase 6.5 report and [`bench/results/phase65-general-optimization/`](../../bench/results/phase65-general-optimization/)
for all raw artifacts, manifests and graphs). Measurements
below are specific to dual Xeon Gold 6130 / GTX 1070; secondary Ryzen/RTX hardware
is unavailable. No third-party implementation is copied by this work.

## Evidence and research review

The following are author-reported results, not Guild measurements. Publication
pages, implementation details and evaluation boundaries matter more than headline
speedups. Sources were accessed 2026-10-20.

| Source / date | Technique and reported evidence | Guild applicability, cost and correctness |
| --- | --- | --- |
| [FlashInfer 0.7](https://flashinfer.ai/2026/09/22/flashinfer-v07.html), September 2026 | Open Prims-TS attention/MoE, communication/compute fusion, experimental opt-in dispatch. MegaMoE reports 1.06–1.32x on eight SM100 GPUs; some comparisons change checkpoint precision. | Scheduling principles transfer; distributed symmetric-memory kernels do not fit one Pascal GPU. High integration/resource cost, no required training for scheduling; precision changes require separate quality checks. FlashInfer Apache-2.0; retain attribution if adapting code. |
| [Autotuner v2](https://flashinfer.ai/2026/09/22/autotuner-v2.html), September 2026 | Include untuned default, time eager host+device work or graph replay separately, persist complete environment/shape identities atomically. 276 eager workloads: median selection regret 1.390% → 0.418%; not serving throughput. H100/B200/GB200/GB300. | Direct fit for GEXEC methodology. Existing leases, identity checks and atomic artifacts remain authoritative. Low numerical risk; representative tuning costs startup time, not per-token work. No training. |
| [CPU–GPU hybrid MoE](https://www.usenix.org/conference/osdi26/presentation/wang-wenxin), OSDI July 2026 | Stream-loading prefill, dual-batch attention/MoE overlap, AVX-512 FP8 GEMV. Abstract reports 1,200 prefill tok/s and 28 INT4 / 21.5 FP8 decode tok/s; distributed prefill uses two RTX 5090s. | Guild already stream-loads Qwen prefill. Investigate actual dependencies and idle intervals before adding buffers/streams. FP8 kernel is not a drop-in for I/K quants. High scheduling/concurrency complexity. No router modification needed. Source-license clearance needed before reuse; abstract alone is insufficient for matched performance claims. |
| [DoMoE](https://www.ijcai.org/proceedings/2026/657), IJCAI 2026 | Domain routing tables restrict semantic expert matching; reports 1.32x average throughput and 1.22x expert hit ratio. Full paper: L40, Xeon Platinum 8480+, 1TB RAM; Mixtral/Phi/Qwen, 70/30 profiling/test split. Table 2 throughput is only 1.21–2.32 tok/s, and FMoE is a reimplementation. | Only useful when expert transfers miss; first measure Guild RAM/GPU/file tiers. Prediction may prepare data but never override routing. Requires routing-history storage/domain calibration (about 24–26MB tables), not router fine-tuning; moderate/high complexity. Increasing tables from 2K to 3K entries loses 7–8% throughput despite better hits. Implementation license pending review. |
| [ReMoE](https://proceedings.mlr.press/v306/zhu26ab.html), ICML July 2026 | Router fine-tuning biases recent experts: 26% more reuse; 8.4% vLLM throughput improvement and 43.6–49.8% TPOT reduction on Jetson Orin NX/llama.cpp. | Research-only: changes model/router behavior and requires training/checkpoints. Not an exact inference optimization. Quality-task averages cannot establish logit parity. Code/checkpoint licenses require independent review. |
| [FlashAttention-4](https://arxiv.org/abs/2603.05451), March 2026 | Blackwell asynchronous MMA pipelines, larger tiles, conditional softmax rescaling and software exponential. Up to 1.3x cuDNN 9.13, 2.7x Triton; 1,613 BF16 TFLOP/s on B200. | SM100 instructions unavailable on Pascal; even RTX 5070 SM120 is not SM100. Useful lesson: optimize the newly limiting non-matmul stages. High implementation cost; altered reduction/exponential order needs tolerance/quality checks. No training. Paper CC-BY-NC-SA is not a source-code license. |
| [TurboQuant / PolarQuant / QJL](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/), March 2026 (papers 2024–2025) | Rotation, scalar/polar quantization and residual correction; 3-bit KV, attention-logit microbenchmark up to 8x versus FP32 keys on H100. Gemma/Mistral and retrieval evaluations. | Experimental lossy storage. Guild already exposes FP16/INT8/Q4/K8V4. New transforms, codebooks and dequantization consume compute and scratch; no training required. Quality evaluation mandatory, not bit-exact. License verification needed before code reuse. |
| [Independent TurboQuant study](https://vllm-project.github.io/2026/05/11/turboquant.html), May 2026 | vLLM 0.20.2 `6ec9bbec3`, 2–4 H100s, 30B–200B+ models, 10 warmups/30 latency samples; TQ adds 10–68% latency and aggressive variants lose reasoning/retrieval accuracy. Compression can still reduce queueing under memory pressure. | Do not infer latency improvement from smaller KV. FP8 tensor-core conclusions do not transfer to Pascal. Evaluate existing modes before adding another lossy format. |
| [EVICT](https://arxiv.org/abs/2605.00342), May / revised September 2026 | Draft-tree truncation using drafter signals and offline verification cost. Reports up to 2.35x autoregressive and average 1.21x EAGLE-3, with fewer activated experts. | Fits GEXEC cost profiles conceptually, but Guild currently verifies at most two tokens and already confidence-gates MTP. Benefit must exceed decision overhead. Training-free; unchanged acceptance/sampling required. Full hardware/implementation license review pending. |
| [SpecBranch](https://arxiv.org/html/2506.01979v4), June 2025 / April 2026 | Rollback-aware branch drafting overlaps target verification; A100 PCIe 40GB, HumanEval/GSM8K/CNN-DM/Spec-Bench, 1.8–4.5x autoregressive. Paper section 6 trains a small H-RAD predictor despite no draft-model retraining. | Research-only for current 8GB shared-GPU MTP: extra branch state, scheduling and predictor training; target features unavailable until dependency completes. Distribution-correct resampling is essential. Do not equate “draft-model-training-free” with no training. Code license not yet cleared. |
| [SGLang radix cache source](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/radix_cache.py), inspected October 2026 | Prefix matching, namespace/salt identity, lock references and eviction. Source inspection, no speedup inferred. | Exact prefix reuse requires both KV and recurrent/GDN checkpoints, model/position/KV identity and session lifetime isolation. Moderate/high memory and concurrency complexity. No training. Apache-2.0 source; no code copied. |
| [CUTLASS 4.8](https://docs.nvidia.com/cutlass/latest/overview.html), September 2026 | Hierarchical tiles, grouped scheduling, workspace reuse, asynchronous data movement, fused epilogues and host-latency guidance. SM100 and SM120 explicitly distinct. | Reuse principles, not an incompatible new backend. Most modern instructions absent on SM61. No general end-to-end speedup follows from GEMM utilization. No training; precision/reduction changes require parity. BSD-3-Clause code. |
| Guild's pinned GGML/native CPU implementation (`native_expert.cpp`, `kq_avx2.cpp`) | Multi-token weight unpack reuse, native activation formats, ISA-gated AVX2/AVX-512BW. Source inspection finds GEXEC sends contiguous groups larger than two back to runtime-token-count native K-quant kernels. | Test compile-time token-count specialization using existing GEXEC kernels before inventing another pack format. Same quant blocks and arithmetic order; bitwise gate/up, hidden-quant and down checks required. No extra weights, training, ABI or cache. MIT project code. |
| [FineMoE](https://arxiv.org/html/2502.05370), February / October 2025 | Semantic expert maps and asynchronous prefetch. Six RTX 3090s, Threadripper PRO 3955WX, 480GB RAM; 64 sampled LMSYS/ShareGPT prompts, average inputs only 37/43 tokens and outputs 127/122. Reports 53% TTFT / 22% TPOT reduction versus MoE-Infinity. Separate A100 experiment shows smaller gains. | No model training; historical map preparation and additional host memory (under 200MB at 32K maps). Does not establish long-prefill benefits for Guild. Async operations still contend for resources: only a trace can establish overlap. No cleared code implementation. |
| [TensorRT-LLM attention documentation](https://nvidia.github.io/TensorRT-LLM/advanced/gpt-attention.html), September 2025 snapshot `0c9430e` | Packed tokens, fused context attention, multi-block decode for low occupancy, shape-dependent XQA selection, on-the-fly KV dequantization. Implementation guidance, not a matched Guild benchmark. | General split-KV/fusion principles fit existing providers; FP8 compute requires newer hardware. Chunking/paging must preserve attention semantics. No training; changed FP order requires parity. Apache-2.0 project, no code copied. |

Source-level follow-up: FlashInfer `v0.7.0/flashinfer/autotuner/autotuner.py`
has `_effective_measure_policy`, `_apply_measure_policy`, graph replay counts,
runner `get_cache_key_extras` and eager atomic publication. EVICT appendix C.4
specifies A100 SXM4 80GB, up to TP8, 32-token trees (steps 4/top-k 8). Its
larger-batch table also includes regressions on high-acceptance HumanEval; adaptive
verification is not a universal win. The OSDI full PDF returned HTTP 403 here, so
its abstract-level numbers are not treated as independently audited methodology.

### Ranked initial experiments

Rank is provisional until fresh profiles are collected. Historical Phase 6
Ornith prefill has substantial CPU expert and GPU work; its 16K decode CPU pool
accounts for about one third of wall time. A 20% improvement to that third alone
predicts only about 7% whole-decode improvement by Amdahl's law.

| Rank | Hypothesis / subsystem | Expected scope and constraints | Initial class |
| --- | --- | --- | --- |
| 1 | Specialize K-quant expert token counts through existing GEXEC functions | CPU arithmetic/loop overhead; both models where formats match; AVX2/AVX-512 gated. Low memory/startup overhead, medium register-pressure risk. Existing per-pool binding preserves concurrency. | Experimental |
| 2 | Reduce prefill register pressure by bounded token tiles | Repeated unpack versus spills tradeoff; must measure 1/2/4/8-token groups and full requests. No approximation. | Experimental |
| 3 | Submit independent shared-expert work before host wait | GPU shared expert depends on post-attention activation, not routed CPU outputs. D2H completion must precede host reads; final combine waits for both. Small expected gain; event overhead may erase it. | Experimental |
| 4 | CPU worker/NUMA and fused-pool partition tuning | Synchronization versus bandwidth; should be selected per hardware/workload rather than assumed universal. Changes placement, not routing/arithmetic. GEXEC should retain any production policy. | Experimental |
| 5 | Batched KV append / attention tiling | Reduce launch/transfer work; preserve causal widths, host mirrors and shared-slot invalidation. External provider ownership limits direct kernel changes here. | Experimental |
| 6 | Cost-aware MTP gating | Already short windows; measure expert-union expansion and accepted tokens per unit cost. No deeper speculative state without proof. | Research-only initially |
| 7 | Exact prefix snapshots | Potential large repeat-request TTFT gain, but recurrent state and session lifetimes raise memory/concurrency cost. | Research-only |
| 8 | Expert prediction | Cannot improve a zero-miss resident CPU path; routing-history overhead may regress it. | Research-only pending misses |
| 9 | New aggressive KV formats / Blackwell attention | Hardware/quality gaps exceed available evidence. Existing precision modes remain the comparator. | Research-only |

## Measurement integrity

The preserved baseline binaries are under `build-cuda12/phase65/baseline/`.
The first old-GEXEC probe correctly fell back because of a package-digest mismatch;
it is a **portable** observation, not a GEXEC measurement. Fresh artifacts were
generated with the preserved binary, and subsequent runs explicitly check activation.
Full-shaped warmup and three repetitions are used for screening; final comparisons
must counterbalance A/B order and report dispersion. Profiling runs are separate.

Phase 6's `1,118x` prefill claim is not an established like-for-like result: no
completed 65,536-populated baseline accompanies it. Earlier measurements and raw
files remain historical evidence, not Phase 6.5 baselines. Similarly, a hash of
65,536 header bytes is not a full source-weights hash.
