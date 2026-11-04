# Decode, fourth round: where Qwen3.8-Flash-Next and GLM-5.3-Flash stand against the bandwidth bounds

Fresh profiles of both models' decode with the current build, four bit-exact changes to GLM-5.3-Flash, and what is
left: on both models the remaining distance to the bandwidth bound is spread over many small kernels. The two
large items are design choices about numerics, not kernel work.

PC: 2x Xeon Gold 6130 (16 cores each, AVX-512 BW without VNNI, 2 NUMA nodes), GTX 1070 8 GB (sm_61, 15 SMs), CUDA 12.
Branch `feat/nemotron-transcribe`, guild-kernels-cpu `9b7281a`, guild-kernels-cuda `f55b62d` (all local).

## Where decode stands

nsys captures of the current build (`cap.sh`; kernel times are valid under nsys, host timings are not):

**Qwen3.8-Flash-Next** (the server's settings, prose and code requests; `kw.py`, `mm.py`).
- **Bound:** the GPU sets the pace. A window reads ~10.5-11.1 GB: 50-53 ms at 210 GB/s.
- **Measured:** 63-66 ms a window without a profiler, the GPU busy nearly all of it.
- **The largest kernels, per window:**
  - The staged MMVQ, 39.6 ms (the dense Q8_0 products of two row groups, the shared expert, the heads).
    `attn_qkv` runs at ~203 GB/s, `attn_gate` and `ssm_out` at ~192-195 GB/s, the draft head at ~203 GB/s.
    The shared expert's 640-row matrices (~1.7 MB each) run at 75-87 GB/s, latency-bound, on a side stream.
  - The hyper-connection read (down, up, norm), 15.7 ms against an 11.6 ms bound.
  - GDN 5.4 ms; its recurrence (29 us a call, 6.3 MB of state read and written) is at its bound.
  - BF16 GEMVs 3.5 ms; waiting for the CPU experts 2.5 ms.

**GLM-5.3-Flash** (5-token prompt, `--spec`; `gk.py` and the decode-roofline lanes' `glmprof2.py`).
- **The GPU is busy 78.8 ms of a ~85 ms step.**
- **At their bound:** the Q6_K attention products (KDA q/k/v and output, DSA output) and the GPU shared expert,
  47.7 ms at 95-99% of their bound. The two 1-token groups of a verify window each read them.
- **The rest, ~13 ms above its bound:**
  - KDA core 5.0 ms (its recurrence at ~94% once the bytes it moves are counted);
  - mHC 3.8 ms (542 small kernels; `hc_pre_k` is one 1,024-thread block a token, 20 Sinkhorn iterations on one warp);
  - DSA q_b 3.2 ms (143 GB/s);
  - router 3.0 ms (the F32 GEMV at its bound, the top-8 one 32-thread block);
  - KDA small products 2.4 ms (~100 GB/s);
  - quantize and norms 1.2 ms (483 small kernels).
- **The CPU side:** experts ~68-75 ms a step. They drift up by ~5 ms over two to three minutes of decode in every
  process, with the machine otherwise idle (cores to 75 C), so in a long run GLM is about as CPU-bound as GPU-bound.

## The changes (GLM-5.3-Flash; the same ids in every A/B)

| Change | Commit | Kernel alone | In the engine (one process, arms alternated) |
| --- | --- | --- | --- |
| Short Q6_K rows (DSA q_b: 1,260 bytes) through aligned 16-byte loads; a warp's staging 32 bytes larger | `2fa9d76` | q_b 171-176 -> 129.5 us at one token; the 2-column head 4.02 -> 3.28 ms (the staging) | with the next row: verify 79.4 / 80.4 / 81.9 / 82.8 -> 79.2 / 80.0 / 81.1 / 82.0 ms a step |
| A verify window's group A writes its KDA state to the rollback copy only, group B starts from it (`GUILD_GLM_KDA_MID1=0`: both) | `2fa9d76` | 4.2 MB less written a KDA layer and step | (the row above) |
| A kept draft's NextN pass joins the next draft's (`GUILD_GLM_ACCEPT_FOLD=0`: at accept) | `541c733` | | accept 2.1-2.2 -> 0.5-0.6 ms, draft 4.4 -> 5.3 ms; 21.18 -> 21.26 tok/s |
| IQ4_XS down rows two to an AVX-512BW register, for one token as for many (`GUILD_IQ4XS_512=0`: ggml-cpu's dot) | `ac7d12a` | all 32 cores: 161 -> 193 GB/s at one token, 91 -> 162 at two | CPU experts 72.5 / 73.3 / 73.8 -> 70.4 / 70.8 / 71.4 ms a step; the step 86.0 / 86.8 / 87.3 -> 85.0 / 85.3 / 85.8 ms; 21.40 -> 21.53 tok/s; a 2,051-token prompt 32.0-32.2 -> 29.4-29.5 s |

Every kernel change is bitwise its predecessor: `glm5_kernels_test` (the Q6_K products, the split verify window's
KDA state and outputs) and `iq4xs_rows512_test` (15,336 results against ggml-cpu's dot, 1-8 tokens).

## Tried, not kept

- **Qwen's GDN a/b projections with their loads asked for ahead** (`gdn_ab_multi_kernel`). 22 us a call in the
  capture, but 7-14 us alone with its weights cold, 3-7% faster with the change, bitwise. Its time in the capture is
  mostly the graph's other streams running beside it. Reverted.
- **GLM's IQ4_XS decode rows through guild-kernels-cpu's single-row AVX-512 kernel:** no gain (one-token rows at
  165 against ggml-cpu's 157 GB/s on all cores). The two-row kernel above took its place.

## What is left

- **Bit-exact kernel work:** each item is worth 0.1-1 ms a window or step.
  - Qwen: the hyper-connection read (~4 ms over its bound, already worked on in round 1), the shared expert's
    latency-bound products, the small kernels of the verify graph.
  - GLM: `hc_pre_k`'s Sinkhorn on the critical path, the top-8, the absorb (172 GB/s), the narrow products, ~420
    quantize launches a step.
- **The two row groups of a window** read every dense weight twice, which lets the CPU experts of one group run
  behind the GPU work of the other. One group a window was measured slower on both models (the CPU experts then sit
  on the critical path).
- **The bytes themselves.** Qwen's dense products are Q8_0 (5.6 GB a window over the two groups). As Q6_K they
  would read ~23% fewer bytes, ~6 ms a window at the measured rate. GLM's attention products are already Q6_K at
  load; Q5_K would read ~16% fewer, ~7 ms of GPU a step. Both change the output (another rounding of the weights),
  so neither is done here.
