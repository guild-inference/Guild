# Decode, fifth round: fewer bytes for Qwen3.8-Flash-Next

Round 4 ended with the two models' kernels near their bandwidth bounds and the large items left being the bytes
themselves.  This round reads fewer of them on Qwen3.8-Flash-Next: its GGUF's own Q8_0 hyper-connection blocks in
place of the pack's bf16 rounding of them, and its dense matrices as Q6_K.  It also adds an opt-in expert
deferral, and tries a few levers that did not pay.  GLM-5.3-Flash gets three bit-exact changes (+0.1-0.7% each).

PC: 2x Xeon Gold 6130 (16 cores each, AVX-512 BW without VNNI, 2 NUMA nodes), GTX 1070 8 GB (sm_61, 15 SMs, 151 W
cap), CUDA 12.  Branch `feat/nemotron-transcribe`, guild-kernels-cpu `9b7281a`, guild-kernels-cuda `39007f6` (all
local).

## Qwen3.8-Flash-Next

`tools/bench_runtime.py` at the server's settings (MTP, `--spec 4`, 56 GiB of RAM experts, no GPU expert cache),
the prose and code requests, 256 tokens, arms alternated, two pairs each.  Tokens a window move with the text, so
ms a window is the fairer column.  Quality is `tests/core/qwen_kl.cpp` against today's output on a 2,085-token
text (repository prose and C++): KL mean of the next-token distributions, and how often the top token is the same.
A change of the CPU experts' summation order alone measures KL 0.0167 and top-1 95.25% there: the noise floor.

| Setting | ms a window (prose / code) | tok/s (prose / code) | KL mean | top-1 | perplexity |
| --- | ---: | ---: | ---: | ---: | ---: |
| today (round 4) | 63.2-63.4 / 66.2-66.5 | 34.3 / 44.9 | - | - | 5.082 |
| `--hc q8_0` | 59.7-59.8 / 62.6-62.9 | 35.4 / 48.0 | 0.020 | 95.0% | 5.077 |
| `--dense q6_k` | 60.5-60.8 / 64.2-64.4 | 34.0 / 47.4 | 0.029 | 93.1% | 5.033 |
| `--dense q6_k --hc q8_0` | 56.0-56.5 / 60.0-60.5 | 38.3 / 48.8 | 0.028 | 93.5% | 5.034 |
| the same, `--spec 5` | 59.7-59.8 / 64.6-64.7 | 37.9 / 52.1 | 0.028 | 93.5% | 5.034 |
| the same, `--defer-experts 4` | 51.9-52.3 / 57.1-57.4 | 41.0 / 53.2 | 0.099 | 87.2% | 5.057 |
| the same, `--defer-experts 3` | 49.8-50.3 / 54.0-54.5 | 41.3 / 53.0 | 0.115 | 87.1% | 5.149 |

(`--dense q6_k` alone: from `cded895`, measured before `--hc q8_0` existed; the others in this round's build.)

### The changes

- **`--hc q8_0`** (`5713c5f`, guild-kernels-cuda `39007f6`).  The GGUF holds every layer's hyper-connection down
  and up matrix as Q8_0.  The pack widened them to bf16 with a rounding: its `conversions.json` marks them
  `"exact": false`, up to 0.014 off.  The decode read those 2.5 GB a window at the server's settings.
  - With the option the engine repacks the GGUF's own blocks (int8 values, then f16 scales) and reads them:
    half the bytes, and the model file's own values.
  - The bf16 copies are not loaded: 638 MiB of Q8_0 in place of 1,200 MiB of bf16, 432 MiB more VRAM free.
  - New down and up kernels.  The down kernel stages each step's activations in shared memory for its block's
    four warps.  One token alone measured 21-23 us (down) and 20-23 us (up), against 34-41 us for the bf16
    kernels in the engine.  In the engine under nsys they take 33 and 26 us a call, sharing the card with the
    graph's other streams.
  - Exactness: `gr_parity` checks the read against a double reference on the dequantized weights (1.7e-7
    relative), and checks 1-8 tokens bit for bit against one-token reads and graph replay.
  - The prompt path's bf16 GEMMs get each matrix rounded back to bf16, the pack's own rounding (0 of 6.5 M values
    differ from its bytes on the matrices checked), so prompts compute what they did.
- **`--dense q6_k`** (`cded895`): the GPU's Q8_0 dense matrices of 2,048+ rows as Q6_K, re-quantized at load
  (132 matrices, 2,365 MiB).  It saves less than its bytes suggest: 2.5-3.5 ms a window, where 1.3 GB fewer
  bytes at 200 GB/s would be 6.6.  In the engine the products run at the card's 151 W cap (round 4), and the GPU
  already waits 2.5 ms a window for the CPU experts.
- **`--defer-experts k`** (`2177eb3`; the hyper-connection read's 3- and 4-token kernels `2bde884`): a layer's
  experts past its router's top k reach the residual before the next layer's FFN, not before its attention.  The
  CPU computes them beside the GPU's next layer, and a window runs as one row group.  This changes the model's
  computation.  With the bytes above cut, it gains 7-11% a window for KL 0.10-0.12.
- **The CPU pool's rows two to an AVX-512BW register** (`fc15389`): the repacked IQ3_S gate/up rows and the IQ4_NL
  down rows, each (row, token) bitwise ggml-cpu's one-token dot as before.  Texts identical in four runs.
  - Gate/up 4.97 -> 5.22 GB/s a thread, down 3.12 -> 3.32.  The deferral mode's windows 0.1-0.4 ms shorter.
- **Q6_K rows of more than 12 blocks summed unrolled up to 8 columns** (guild-kernels-cuda `19cee54`):
  ssm_out at two columns 84.9 -> 75.7 us, bitwise.

### Tried, not kept

| What | Result |
| --- | --- |
| One row group a window, exact (`GUILD_SPEC_SPLIT=0`), with `--dense q6_k --hc q8_0` | 56.0-56.2 -> 62.5-62.6 ms (prose), 60.1-60.4 -> 70.1-70.2 (code), the same text: the CPU experts (23-28 ms a window) sit on the critical path |
| Both hardware threads of every core for the CPU pool (63 workers + host) | 56.1 -> 61.3-61.6 ms (prose), 60.0-60.3 -> 64.9-67.4 (code), the same text |
| `--spec 6` with `--dense q6_k --hc q8_0` | 37.5 / 52.3 tok/s (prose / code) against 38.3 / 48.8 at `--spec 4`; `--spec 5` (37.9 / 52.1) costs code's windows less |
| Power cap 151 -> 170 W | no gain; the cap is back at 151 W |
| `--decode-strategy adaptive` | slower than MTP alone: prose 34.2 -> 23.7-30.8, code 44.8 -> 40.1, edit 55.3 -> 46.4 tok/s |
| Deferral at KTransformers' placement (the deferred sums added to the next layer's FFN output) | worse quality at the same speed (`2177eb3`'s table) |

### Where the time goes now

nsys, `--dense q6_k --hc q8_0`, the code request (69.9 ms a window under the profiler, 60 without):

- **Dense products: 35.6 ms** (two row groups; the shared expert; the verify head, 2.1 ms; the draft head, 2.7 ms
  over 2.5 drafts).  `attn_qkv` runs at 189 GB/s against a 222 GB/s read ceiling.
- **The hyper-connection read: 11.8 ms** (180 reads of 7.0 MB), against 6.0 ms for those bytes at 210 GB/s.
- **GDN: 7.4 ms.**  Its recurrence is at its bound; each group reads and writes the 3.1 MB state of each layer.
- **The router's and the other bf16 products: 3.7 ms.**  Waiting for the CPU experts: 2.5 ms.

The CPU expert pool runs at 116 GB/s on its wall time and is busy 84% of it.  With all 32 cores busy, the IQ4_NL
rows' kernel streams 4.5 GB/s a core (143-146 GB/s) where plain reads of the same items reach 6.6 GB/s a core
(212 GB/s): the kernels set the rate, not the memory.  One core alone streams 4.7-7.1 GB/s through the kernel; it
varied between runs.

## GLM-5.3-Flash (bit-exact)

`glm5_run --spec`, 256 tokens, arms alternated in one process, the same ids in every pair:

| Change | Commit | Result |
| --- | --- | --- |
| A token's Q6_K rows in a loop per warp; KDA q/k/v as one 24,576-row product | `8626cda` | DSA q_b 140.8 -> 102.3 us; 21.91 -> 21.94 tok/s |
| The CPU's gate/up and down items in one pool phase, a down item waiting per group | `fde4cc5` | CPU experts 67.6-67.8 -> 65.6-65.8 ms a step; 21.97 -> 22.12 tok/s |
| Launches that fit the GPU in one wave; a token's FFN gate and up as one product | `1cd5416` | verify 77.2-79.2 -> 76.2-78.4 ms a step; 21.93 -> 22.07 tok/s |

GLM stays balanced.  Its CPU experts take 65-70 ms of an ~85 ms step, 76-81% of their DRAM bound (12 GB a step at
226 GB/s: 53 ms), and the GPU is busy nearly all of the step.  A faster step needs both to drop.

## What is left

- **Two row groups read every dense weight twice** (~11 ms a window as Q6_K).  One exact group would need the CPU
  experts about twice as fast; deferral gets there by changing the computation.
- **The draft: 6.3-7.0 ms a window**, 2.7 ms of it 2.5 passes over the 213 MB Q6_K draft head.  A lighter draft
  head (fewer bits, or fewer tokens) changes only the drafts, not the output.  Its acceptance cost is not measured.
- **The hyper-connection read in the engine** runs at about half its kernels' speed alone; how much of that is
  the critical path, and not other streams' work beside it, is not measured.
