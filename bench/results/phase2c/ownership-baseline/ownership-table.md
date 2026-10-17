| Source | Category | Intended destination | Group |
|---|---|---|---|
| `src/core/device.cu` | C/D | Core device allocation and diagnostic poison kernel | device/memory |
| `src/core/pinned.cu` | C | Core pinned allocation, mapping and lifetime | host/memory |
| `src/core/remote_expert_opt.cu` | A+C | Core orchestration; extract reduce_experts and copy_cpu_rows only | multi-device |
| `src/kernels/cuda/bf16_gemv.cu` | A | guild-kernels-cuda | dense/bf16 |
| `src/kernels/cuda/cvec.cu` | B | guild-kernels-cuda: isolated qwen specialization | specialized/cvec |
| `src/kernels/cuda/dequant_bf16.cu` | A | guild-kernels-cuda | dense/quantization |
| `src/kernels/cuda/dequant_s2.cu` | A | guild-kernels-cuda | dense/quantization |
| `src/kernels/cuda/elementwise.cu` | A | guild-kernels-cuda | elementwise/doorbell |
| `src/kernels/cuda/fused_gdn.cu` | A | guild-kernels-cuda | recurrent/gdn |
| `src/kernels/cuda/fused_gr.cu` | B | guild-kernels-cuda: isolated qwen specialization | specialized/fused_gr |
| `src/kernels/cuda/gdn.cu` | A | guild-kernels-cuda | recurrent/gdn |
| `src/kernels/cuda/gr.cu` | B | guild-kernels-cuda: isolated qwen specialization | specialized/gr |
| `src/kernels/cuda/iq_kernels.cu` | A | guild-kernels-cuda | dense/iq |
| `src/kernels/cuda/kv_q4.cu` | A | guild-kernels-cuda | kv/q4 |
| `src/kernels/cuda/kv_q8.cu` | A | guild-kernels-cuda | kv/int8 |
| `src/kernels/cuda/kv_stream.cu` | A | guild-kernels-cuda | kv/streaming |
| `src/kernels/cuda/native_bf16.cu` | A | guild-kernels-cuda | dense/bf16 |
| `src/kernels/cuda/native_flash_attn.cu` | A | guild-kernels-cuda | attention/flash |
| `src/kernels/cuda/native_gdn.cu` | A | guild-kernels-cuda | recurrent/gdn |
| `src/kernels/cuda/native_gdn_preprocess.cu` | A | guild-kernels-cuda | recurrent/gdn |
| `src/kernels/cuda/native_gr_norm.cu` | B | guild-kernels-cuda: isolated qwen specialization | specialized/native_gr_norm |
| `src/kernels/cuda/native_gr_postops.cu` | B | guild-kernels-cuda: isolated qwen specialization | specialized/native_gr_postops |
| `src/kernels/cuda/native_mmvq.cu` | A | guild-kernels-cuda | dense/native-mmvq |
| `src/kernels/cuda/native_moe.cu` | A | guild-kernels-cuda | moe/native |
| `src/kernels/cuda/native_ple_postops.cu` | B | guild-kernels-cuda: isolated qwen specialization | specialized/native_ple_postops |
| `src/kernels/cuda/native_qsa.cu` | A | guild-kernels-cuda | attention/normalization |
| `src/kernels/cuda/native_qsa_indexer.cu` | A | guild-kernels-cuda | attention/indexer |
| `src/kernels/cuda/native_qsa_score.cu` | A | guild-kernels-cuda | attention/scoring |
| `src/kernels/cuda/native_rope.cu` | A | guild-kernels-cuda | attention/rope |
| `src/kernels/cuda/native_router.cu` | A | guild-kernels-cuda | moe/routing |
| `src/kernels/cuda/ple.cu` | B | guild-kernels-cuda: isolated qwen specialization | specialized/ple |
| `src/kernels/cuda/qsa.cu` | A | guild-kernels-cuda | attention/qsa |
| `src/kernels/cuda/qsa_decode_attn.cu` | A | guild-kernels-cuda | attention/decode |
| `src/kernels/cuda/qsa_prompt_attn.cu` | A | guild-kernels-cuda | attention/prefill |
| `src/kernels/cuda/qsa_select.cu` | A | guild-kernels-cuda | attention/selection |
| `src/kernels/cuda/quantize_act.cu` | A | guild-kernels-cuda | dense/quantization |
| `src/kernels/cuda/rope.cu` | A | guild-kernels-cuda | attention/rope |
| `src/kernels/cuda/router_top10.cu` | A | guild-kernels-cuda | moe/routing |
| `src/kernels/cuda/s2_expert_grouped.cu` | A | guild-kernels-cuda | moe/canonical |
| `src/kernels/cuda/s2_gemv.cu` | A | guild-kernels-cuda | dense/canonical |
| `src/kernels/cuda/s2_gemv_fast.cu` | A | guild-kernels-cuda | dense/canonical |
| `src/kernels/cuda/s2_gemv_q8.cu` | A | guild-kernels-cuda | dense/canonical |
| `src/kernels/cuda/s2_gemv_quads.cu` | A | guild-kernels-cuda | dense/canonical |
| `src/kernels/cuda/s_gemv.cu` | A | guild-kernels-cuda | dense/canonical |
| `src/kernels/cuda/sampler.cu` | A | guild-kernels-cuda | sampling/device-primitives |
| `src/kernels/cuda/shared_expert.cu` | A | guild-kernels-cuda | moe/shared |
| `src/kernels/cuda/verify_kernels.cu` | A | guild-kernels-cuda | verification/device-primitives |
| `src/prefill/gdn_rec_parity.cu` | D | Core test retained; standalone backend test consumer | tests |
| `src/prefill/gemm.cu` | A | guild-kernels-cuda | prefill/gemm |
| `src/prefill/gemm_bf16_parity.cu` | D | Core test retained; standalone backend test consumer | tests |
| `src/prefill/ggml_cuda_host.cu` | D | guild-kernels-cuda private pinned GGML context | ggml/context |
| `src/prefill/kernels.cu` | A/B | guild-kernels-cuda | prefill/kernels |
| `src/prefill/moe_fused.cu` | A | guild-kernels-cuda | prefill/moe_fused |
| `src/prefill/moe_fused_iq.cu` | A | guild-kernels-cuda | prefill/moe_fused_iq |
| `src/prefill/moe_mmq.cu` | A | guild-kernels-cuda | prefill/moe_mmq |
