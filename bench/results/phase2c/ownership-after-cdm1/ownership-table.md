| Source | Category | Intended destination | Group |
|---|---|---|---|
| `src/core/device.cu` | C/D | Core device allocation and diagnostic poison kernel | device/memory |
| `src/core/pinned.cu` | C | Core pinned allocation, mapping and lifetime | host/memory |
| `src/prefill/gdn_rec_parity.cu` | D | Core test retained; standalone backend test consumer | tests |
| `src/prefill/gemm.cu` | A | guild-kernels-cuda | prefill/gemm |
| `src/prefill/gemm_bf16_parity.cu` | D | Core test retained; standalone backend test consumer | tests |
| `src/prefill/ggml_cuda_host.cu` | D | guild-kernels-cuda private pinned GGML context | ggml/context |
| `src/prefill/kernels.cu` | A/B | guild-kernels-cuda | prefill/kernels |
| `src/prefill/moe_fused.cu` | A | guild-kernels-cuda | prefill/moe_fused |
| `src/prefill/moe_fused_iq.cu` | A | guild-kernels-cuda | prefill/moe_fused_iq |
| `src/prefill/moe_mmq.cu` | A | guild-kernels-cuda | prefill/moe_mmq |
