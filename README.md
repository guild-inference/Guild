# Guild

> **Origin:** Guild was originally derived from
> [Strata](https://github.com/Niko1221/Strata) by Niko1221 and contributors,
> then substantially reworked as an independent Mixture-of-Experts inference engine.

**Guild is a native C++ inference runtime focused on high-performance local execution of Mixture-of-Experts models.**

Guild is especially designed for asymmetric systems that conventional GPU-centric runtimes may underuse, such as servers with large amounts of system RAM, powerful CPUs, and limited GPU VRAM.

## Why Guild?

Large Mixture-of-Experts models do not need every parameter for every token.

Guild is designed around that fact.

It can distribute model execution across:

- GPU VRAM
- system RAM
- CPU compute
- storage-backed expert tiers

while dynamically planning KV cache placement, expert residency, speculative decoding, and other runtime resources around the hardware actually available.

Guild is particularly useful on systems such as:

- large-RAM servers with modest GPUs
- dual-socket and NUMA systems
- consumer GPUs paired with large system memory
- older GPUs with limited VRAM
- heterogeneous or otherwise asymmetric machines

## Current status

Guild is under active development.

Core remains v0.x. See [version and revision policy](docs/releases/VERSIONING.md)
and [build/release pipelines](docs/releases/PIPELINES.md) for build identity,
development artifacts and release validation gates.

Current work includes:

- native C++ model loading and inference
- architecture-independent model descriptors
- Mixture-of-Experts expert scheduling
- GPU / RAM / file expert tiering
- native execution memory planning
- host-only KV cache support
- speculative / MTP decoding
- native OpenAI-compatible HTTP serving
- native model registry and downloader
- persistent in-process model execution
- hardware telemetry and benchmarking

### Architecture support

| Architecture | Status |
| --- | --- |
| Qwen3.8 / qwen4exp | Supported / primary development target |
| Qwen3.5 MoE / Ornith | Experimental / active development |
| Fish Audio Drama 3 (text to speech) | Experimental: speech and voice cloning on the GPU + CPU pool |
| GLM-5.3-Flash (`glm5-next` GGUF) | Experimental: its own engine, attention on the GPU, experts on the CPU |
| DeepSeek-style MoE | Planned |
| Mixtral-style MoE | Planned |
| Other architectures | Experimental as added |

Support status may change quickly while Guild is under active development.

## Quick start (One-Command Install)

Install the precompiled Guild development preview into your user account (no root or administrator privileges required, and no compiler needed):

### Linux (x86-64)

```bash
curl -fsSL https://raw.githubusercontent.com/guild-inference/Guild/main/install.sh | bash
```

### Windows PowerShell (x86-64)

```powershell
irm https://raw.githubusercontent.com/guild-inference/Guild/main/install.ps1 | iex
```

### Version-pinned installation

```bash
# Linux (pinned to v0.2.0-preview.3)
curl -fsSL https://raw.githubusercontent.com/guild-inference/Guild/main/install.sh | GUILD_VERSION=v0.2.0-preview.3 bash
```

```powershell
# Windows PowerShell (pinned to v0.2.0-preview.3)
$env:GUILD_VERSION = "v0.2.0-preview.3"; irm https://raw.githubusercontent.com/guild-inference/Guild/main/install.ps1 | iex
```

You can also download and inspect [`install.sh`](install.sh) or [`install.ps1`](install.ps1) directly before running (`bash install.sh --dry-run` / `.\install.ps1 -DryRun`).

### Supported platforms, CUDA variants & installation paths

The installer automatically detects your operating system, CPU instruction support (`AVX2`, `FMA`, `F16C`, `BMI2`, `AVX-512`), NVIDIA GPU compute capability, and driver version:

| Variant | Target Hardware & Driver | Default Install Paths |
| --- | --- | --- |
| `linux-x86_64-cuda13` | Linux x86-64 (`AVX2+`), NVIDIA Turing / Ampere / Ada / Blackwell (`sm_75`–`sm_120`), driver `570+` | `~/.local/bin/guild` → `~/.local/lib/guild/current/guild` |
| `linux-x86_64-cuda12-sm61` | Linux x86-64 (`AVX2+`), NVIDIA Pascal / Volta / Turing / Ampere / Ada (`sm_61`–`sm_89`, including GTX 1070/1080), driver `525+` | `~/.local/bin/guild` → `~/.local/lib/guild/current/guild` |
| `linux-x86_64-cpu` | Linux x86-64 (`AVX2+`) without a compatible NVIDIA GPU (CLI management, GPACK conversion, and preview TUI) | `~/.local/bin/guild` → `~/.local/lib/guild/current/guild` |
| `windows-x86_64-cpu` | Windows 10/11 x86-64 (`AVX2+`) native CLI & TUI in PowerShell / Windows Terminal | `%LOCALAPPDATA%\Guild\bin\guild.exe` |

### Updating, rolling back & uninstalling

- **Update / Reinstall**: Re-run the install command. Existing models, `~/.local/share/guild` (or `%LOCALAPPDATA%\Guild\data`), GPACK containers, GEXEC caches, and kernel packages are never deleted.
- **Rollback**: Every update preserves the prior version pointer. Roll back atomically with:
  ```bash
  bash install.sh --rollback        # Linux
  .\install.ps1 -Rollback           # Windows PowerShell
  ```
- **Uninstall** (removes binaries while preserving your models and data store):
  ```bash
  bash install.sh --uninstall       # Linux
  .\install.ps1 -Uninstall          # Windows PowerShell
  ```

---

## Native Terminal User Interface (`guild tui`)

Launch the interactive native C++ terminal interface over local terminals, SSH sessions, or Windows Terminal:

```bash
guild tui
guild tui ornith:35b --decode-strategy adaptive --spec-max-draft 4
```

### TUI Views & Keyboard Shortcuts

| Shortcut / Command | Action |
| --- | --- |
| `F1` / `/chat` | Interactive streaming chat & multiline prompt view |
| `F2` / `/models` | Interactive model selector & local store manager (`Up`/`Down`, `Enter` to load) |
| `F3` / `/config` | Runtime configuration (`Decode Strategy`, `Spec Max Draft`, `Temperature`, `Max Tokens`, `Prefill Batch`) |
| `F4` / `/telemetry` | Deep-dive prefill/decode throughput, speculative window breakdown & GEXEC lockfile inspector |
| `F5` / `/help` | Keyboard shortcut & slash-command reference |
| `Enter` | Submit prompt (or load selected model in `F2 Models`) |
| `Alt+Enter` / `Ctrl+J` | Insert newline in multiline prompt (or end a line with `\`) |
| `Esc` / `Ctrl+C` | Cancel in-flight token generation immediately without exiting TUI |
| `PgUp` / `PgDn` (`Ctrl+U`/`Ctrl+D`) | Scroll conversation history up / down |
| `Ctrl+R` / `/reasoning` | Expand or fold `<think>...</think>` reasoning blocks |
| `Ctrl+L` / `/clear` | Clear conversation history and reset session KV / retrieval state |
| `Ctrl+Q` / `/quit` | Restore terminal state and exit |

---

## Building from source (Developers & Custom Hardware)

```bash
git clone https://github.com/guild-inference/Guild.git
cd Guild

# Standard NVIDIA CUDA build (SM75+)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGUILD_ENABLE_CUDA=ON
cmake --build build -j$(nproc)

# Pascal / Volta CUDA 12.x build (e.g. GTX 1070 sm_61)
cmake -S . -B build-cuda12 -DCMAKE_BUILD_TYPE=Release -DGUILD_ENABLE_CUDA=ON \
  -DGUILD_EXPERIMENTAL_SM60=ON -DCMAKE_CUDA_ARCHITECTURES=61
cmake --build build-cuda12 -j$(nproc)

# Older CPUs without AVX2 (Sandy/Ivy Bridge AVX or SSE4.2)
cmake -S . -B build-avx -DCMAKE_BUILD_TYPE=Release -DGUILD_ISA_FLOOR=avx
cmake --build build-avx -j$(nproc)
```

Then:

```bash
./build/guild --help
./build/guild tui
```

## Commands

```text
guild tui [model]        Launch the interactive native Terminal User Interface
guild pull <model>       Download a model
guild import <path>      Import an existing local model
guild list               List installed models
guild show <model>       Inspect a model
guild run <model>        Run a model interactively
guild serve <model>      Start the OpenAI-compatible server
guild inspect <path>     Inspect GGUF, GPACK, or GEXEC containers (--plan, --json)
guild convert <in>       Convert GGUF/safetensors into a portable .gpack container
guild optimize <gpack>   Compile a hardware-specialized .gexec execution artifact
guild verify <file>      Verify .gpack or .gexec integrity and SHA-256 digests
guild bench <model>      Benchmark a model
guild ps                 Show running model processes
```

Some commands and options are still evolving.

## Serving

Guild includes a native OpenAI-compatible HTTP server.

Example:

```bash
guild serve <model>
```

By default, Guild should bind locally.

Typical endpoint:

```text
http://127.0.0.1:11434/v1
```

Supported API functionality is still evolving.

### Prompt chunks on an 8 GB card

Qwen3.8-Flash-Next reads a prompt in chunks, and every chunk streams the routed experts the GPU does not hold over
PCIe (on a GTX 1070, all 55 GiB of the UD-IQ4_XS experts a chunk). `--prefill-rows 512` computes a chunk as 512-token
chunks (every product in 512-row slices, the 512-token chunk's expert path) while it streams the experts once, so a
larger chunk gives the 512-token chunks' results bit for bit where the expert products do not depend on how many rows
an expert has (llama.cpp's MMQ without stream-k, as on Pascal).

On an 8 GB card `--prefill auto` (setup's choice) starts from 512-token chunks and, on Pascal, grows them into the
VRAM the load leaves free, up to 4096 tokens, computed as chunks of 512 (`GUILD_PREFILL_GROW=1` does it on newer
cards too, whose results then move slightly; `--prefill N` keeps N). On a GTX 1070 + dual Xeon Gold 6130 with setup's
Qwen settings, the load left 2,954 MiB free and the chunks grew to 4096: a 4,008-token prompt 52.2 -> 19.6 s and a
31,918-token one 432.6 -> 169.7 s, the same tokens after each
([bench/results/2026-11-03-decode-round11](bench/results/2026-11-03-decode-round11/README.md)).

### Qwen3.8-Flash-Next with the DFlash 2 drafter

Setup fetches incoai's DFlash 2 drafter for Qwen3.8-Flash-Next (Apache-2.0, ~1.2 GB) and writes `--dflash <dir>`,
`--dflash-vocab <file>` and `--spec 8` into the run config. One pass of the drafter proposes up to seven tokens from
five of the model's layers, in place of the MTP layer, which is then not loaded. The verify window still decides
every token, so the output is the same. Setup leaves it out with `--no-dflash`, on AMD, across several GPUs, and for
an engine without `--dflash`.

```bash
guild-generate --serve --native <model>.gguf --pack <pack> --dflash <drafter dir> \
    --dflash-vocab data/dflash_vocab.bin --spec 8 ...
```

`--dflash-vocab` limits the drafter's head to a token subset (int32 ids; without it, all 248,320 rows). Setup passes
`data/dflash_vocab.bin`, 163,778 tokens of English, code, Chinese, Japanese, Korean, Russian and multilingual text,
whatever `--draft-vocab` says (that option chooses the MTP layer's subset). `data/dflash_vocab_en.bin` (65,536 tokens of
English and code) is there to pass by hand: English and code answers up to 1.4% faster, Chinese and Russian ones
28-31% slower. `tools/draft_vocab.py` builds both. The drafter's matrices are Q4_K (`GUILD_DFLASH_WEIGHTS=int8` keeps
them as int8 rows: 239 MiB more VRAM; prose 4% and code 3% slower, sql 3% faster).

On a dual Xeon Gold 6130 with a GTX 1070, both with `--dense q6_k --hc q8_0`, decode went from 39.7 / 49.9 to 42.9 /
56.2 tok/s (prose / code) against the MTP layer at `--spec 4`. The drafter takes 560 MiB of VRAM, the MTP layer about
1.1 GB ([bench/results/2026-11-03-decode-round9](bench/results/2026-11-03-decode-round9/README.md)).

### Qwen3.8-Flash-Next's dense matrices (UD-IQ4_XS)

Unsloth's UD-IQ4_XS stores every dense and hyper-connection matrix as Q8_0. For it setup writes
`--dense q4_k --dense-map ssm_out=q5_k,attn_output=q8_0 --dense-imatrix <imatrix> --hc q8_0`:
- The dense matrices are re-quantized at load: Q4_K, the SSM outputs Q5_K. The attention outputs keep their Q8_0:
  5% of the bytes, but they carry the most loss.
- The rounding is weighted by Unsloth's importance matrix (`imatrix_unsloth.gguf_file`, 553 MiB, downloaded beside
  the model).
- The hyper-connection matrices are read as the GGUF stores them.

On a dual Xeon Gold 6130 with a GTX 1070:
- **Loss:** the output moves a KL of 0.047 / 0.066 from the GGUF's own matrices (two texts). The `--dense q5_k` setup
  wrote before moved 0.053 / 0.077. The model's floor, a 2^-9 rounding of one kind of weight, is 0.020 / 0.029.
- **Speed:** decode is 1.4% faster a token than with `q5_k`. With the DFlash drafter, six prompts went from a
  geometric mean of 45.2 to 46.4 tok/s.

`--dense-map` and `--dense-imatrix` choose other points: `--dense q5_k --dense-map attn_output=q8_0` measured 0.031 /
0.053 at 1.5% slower than `q5_k`, and `--dense q6_k --dense-map attn_output=q8_0` 0.024 / 0.036 at 6.5% slower
([bench/results/2026-11-04-qwen-round13](bench/results/2026-11-04-qwen-round13/README.md)).

On Pascal setup also writes `--prefill-mmq dense`: the prompt's dense products go through int8 MMQ instead of fp32
cuBLAS. A 4,008-token prompt took 19.7 -> 16.0 s and a 31,918-token one 171.4 -> 142.0 s. The KL of the predictions
after the prompt went 0.012 -> 0.015 / 0.014 -> 0.016, under the floor. The GSQ-RCO quants' dense matrices are 3-6
bits already, so setup leaves them alone.

### Ornith-1.5-35B with its MTP draft layer

Ornith's GGUF ships a multi-token-prediction layer that drafts tokens for the verify window. Write it in the
engine's form once, then pass the folder to `guild-generate` (or `"mtp_dir"` in the model's manifest):

```bash
python tools/ornith_mtp_rt.py --gguf Ornith-1.5-35B-Q4_K_M.gguf --out <dir>    # ~630 MB
guild-generate --serve --native Ornith-1.5-35B-Q4_K_M.gguf --pack <pack> --mtp <dir> ...
```

Every emitted token is still the model's own: the drafts only decide how many tokens a step checks at once. On a
dual Xeon Gold 6130 with a GTX 1070, decode went from 54.9 / 54.7 to 64.7 / 74.7 tok/s (prose / code) and used
794 MiB more VRAM ([bench/results/2026-10-24-numa-iq3s](bench/results/2026-10-24-numa-iq3s/README.md)).
The tool's `--draft-vocab data/draft_vocab_en.bin` (an English-and-code draft vocabulary) took it to 68.2 / 79.1,
at the cost of never drafting tokens outside that subset; staged MMVQ kernels and batched routing, to 72.8 / 85.2
([bench/results/2026-10-27-staged-mmvq](bench/results/2026-10-27-staged-mmvq/README.md)).
Without the folder, `--decode-strategy auto` drafts from the prompt and the answer so far instead.

### Fish Audio Drama 3 (text to speech)

[Fish Audio Drama 3](https://huggingface.co/fishaudio/drama-3) speaks: a Qwen3.5-MoE backbone (Ornith's topology)
emits one audio frame per step, a codebook tower predicts the frame's codes, and a codec turns them into 44.1 kHz
audio. Guild runs the backbone like Ornith (experts on the CPU pool) and the tower and codec on the GPU. Convert the
download once (the tools need torch and transformers, and a llama.cpp checkout for its Qwen3.5 converter):

```bash
python tools/drama3_backbone_gguf.py --model drama-3 --llama-cpp <llama.cpp> --out Drama3-BF16.gguf   # ~69 GB
llama-quantize Drama3-BF16.gguf Drama3-Q4_K_M.gguf Q4_K_M                                               # 21 GB
python tools/iq_pack.py --gguf Drama3-Q4_K_M.gguf --out <pack> --compat-bf16
python tools/drama3_tower_rt.py --model drama-3 --out <dir>/tower     # 1.4 GB
python tools/drama3_codec_rt.py --codec drama-3/codec.pth --out <dir>/codec
guild-generate --native Drama3-Q4_K_M.gguf --pack <pack> --drama3 <dir> --max-context 10240 \
  --say "Hello from Guild." --wav hello.wav [--ref-wav voice.wav --ref-text "what the clip says"]
```

The server (`--config` with those `args`) answers `POST /v1/audio/speech` (OpenAI: `input`, `voice`, `instructions`,
`response_format` wav or pcm, `stream_format: audio`) and `POST /v1/tts` (Fish Audio: `text`, `references` with a
base64 WAV and its transcript); `voices_dir` in the config names voices (`<name>.wav` + `<name>.txt`). A request
without a `seed` gets a random one, sent back in the `X-Guild-Seed` header so the take can be replayed. On a dual Xeon
Gold 6130 with a GTX 1070 it speaks at 2.1x real time (the tower's last eight layers and the codec run beside the
backbone's next row; `GUILD_DRAMA3_OVERLAP=0` turns that off, 1.8x;
[bench/results/2026-11-03-decode-round11](bench/results/2026-11-03-decode-round11/README.md)) and uses 5.2 GB of VRAM;
the tower, the codec and the voice encoder match fish-speech's float32 modules
([bench/results/2026-10-24-drama3](bench/results/2026-10-24-drama3/README.md)). Sampled audio is the model's own
distribution, not the reference's exact draws. Drama 3's weights are under the Fish Audio Community License
(free for research and non-commercial use; commercial use only below USD 500,000 annual revenue, and any hosted
service needs a separate license from Fish Audio).  Guild does not ship them.  The HIP (AMD) build compiles the
speech parts but has not been run on an AMD card.

### GLM-5.3-Flash

[GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) (MIT license) runs from Unsloth's GGUF
([unsloth/GLM-5.3-Flash-GGUF](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF), UD-IQ4_XS, 5 shards, 147 GiB) on its
own engine: attention, mHC and the head on the GPU, the 288 routed experts on the CPU (copied into node-local memory,
so about 160 GiB of RAM), the NextN layer drafting a token per step. Export the tokenizer from the first shard once,
then pass both to `guild-generate` (or a server config with those `args`):

```bash
python tools/guild_tokenizer.py --gguf GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf --out <dir>
guild-generate --serve --native GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf --tokenizer <dir> --max-context 262144
```

The engine fits the context to the VRAM and says how much it holds. A token of context takes 1,088 bytes of VRAM in each
DSA layer (its FP16 latent and a quarter of a pooled indexer key; the indexer's per-token key and gate rows are rings of
512 rows), so an 8 GB card holds 32k tokens: with `--max-context 32768` the context takes the place of the last four of
the 43 shared experts on the GPU (the NextN block's and layers 42 to 44's), which their CPU copies then compute.
`GUILD_GLM_CTX_FIRST=16384` gives every shared expert its GPU place first (24,832 tokens on that card). A longer context
(`--max-context 262144`, as Qwen's server config) keeps the latent rows of its later cells in pinned host memory (1 KiB
a cell and DSA layer: 2.9 GiB at 262,144) and copies the 2,048 rows each query's indexer selects to the GPU (a decoded
token only those its layer's last token did not select); the VRAM keeps every cell's pooled key, and the first cells'
latents (11,776 on an 8 GB card, with the FFNs placed as at 32k). The output is the all-VRAM context's bit for bit;
`GUILD_GLM_KV_HOST=0` limits the context to the VRAM. When the GGUF's Q8_0 attention matrices would leave less than the
context asked for (up to 16k tokens), it re-quantizes them to Q6_K at load (about 16 s); `GUILD_GLM_DENSE=q8_0` keeps
Q8_0, `q6_k` always converts.

`--dense q5_k` / `--dense q4_k` re-quantize them and the shared experts' GPU copies to Q5_K / Q4_K instead.
`--dense-imatrix` weights that re-quantization with Unsloth's GLM-5.3-Flash `imatrix_unsloth.gguf_file`, and
`--dense-map` sets a type per matrix kind. On the PC below, at a 262,144-token context with NextN drafts, with KL from
the GGUF's Q8_0 attention on two texts (a change of rounding alone measures ~0.023 / 0.025):

| Format | Decode step | KL |
| --- | ---: | --- |
| Q6_K | 79.6 ms | 0.027 / 0.029 |
| `--dense q5_k` with the imatrix | 73.3 ms | 0.033 / 0.039 |
| `--dense q4_k --dense-map attn_output=q6_k` with it | 71.7 ms | 0.040 / 0.048 |
| `--dense q4_k` with it | 68.3 ms | 0.054 / 0.060 |

See [bench/results/2026-11-04-glm-dense-map](bench/results/2026-11-04-glm-dense-map/README.md). `GUILD_GLM_MTP=0`
decodes without the drafts. With the drafts and Q6_K attention, a verify window's second token takes its shared expert
from the CPU copy, which rounds differently, so the text can part from plain greedy decoding;
`GUILD_GLM_SHEXP_SPLIT=0` keeps it equal bit for bit (3-5% slower on the PC below). With `--dense q5_k` / `q4_k` both
groups take it on the GPU by default, and the text is greedy decoding's.
`data/draft_vocab_glm.bin`, copied into the tokenizer directory as `draft_vocab.bin`, makes the drafts read 117,354 of
the head's 154,880 rows (0.55 ms less a step there). `--dflash <dir>` drafts with incoai's GLM-5.3-Flash DFlash 2
checkpoint instead of the NextN block (CC BY-NC-ND 4.0, for research and evaluation: setup does not fetch it). A step
then verifies up to three of its drafts, while their cumulative probability stays at or above 0.7
(`GUILD_GLM_DFLASH_CUM`); the KDA layers replay the kept tokens on accept, and the text is plain greedy decoding's.
On the PC below that ran prose +1%, code +9% and sql +16% against NextN, and story -8%
([bench/results/2026-11-03-decode-round10](bench/results/2026-11-03-decode-round10/README.md)).

On a dual Xeon Gold 6130 with a GTX 1070 (8 GB): 262,144 tokens of context (32k in VRAM alone); at a short context 14.3
tok/s plain and 21.9 tok/s with the drafts (86% accepted) with every shared expert on the GPU, and 21.8 tok/s with the
32k context's four on the CPU; 19.0 tok/s after a 16,000-token prompt and 18.2 tok/s after a 32,000-token one (17.6 to
18.3 at a 262,144-token context with its row cache and FP16 attention rows, 18.0 to 18.2 after a 128,000-token prompt,
[bench/results/2026-11-01-glm-256k](bench/results/2026-11-01-glm-256k/README.md)), prompts of 640 to 2967 tokens at 50
to 54 tok/s (16k to 32k tokens at 44 to 57 tok/s), and a conversation's next turn reads only what is new. The GPU
computes one token's (or half a prompt window's) attention while the CPU computes the other's experts. Guild's decode
steps match llama.cpp's logits on the same GGUF to a mean KL of 0.015 to 0.028 nats, with the same top token on 98% of
them ([bench/results/2026-10-25-glm5](bench/results/2026-10-25-glm5/README.md); the decode against this PC's memory
roofline, and what moved it, in
[bench/results/2026-10-30-decode-roofline](bench/results/2026-10-30-decode-roofline/README.md)). It has run on CUDA only
so far; the server parses GLM's tool calls and its `<think>` blocks.

## Performance philosophy

Guild is not intended to be a generic wrapper around every model architecture.

Its primary goal is to extract high performance from sparse and Mixture-of-Experts models, especially on hardware where:

- system RAM is much larger than VRAM
- memory placement matters
- expert routing can be exploited
- CPU and GPU resources can work together
- conventional all-GPU execution is impractical

Performance claims should be based on reproducible benchmarks using equivalent model formats and settings.

## GPACK and GEXEC

Guild is developing two native model formats:

**GPACK**

A portable Guild model package containing model weights, architecture metadata, tokenizer assets, auxiliary models such as MTP components, and other hardware-independent information required to run the model.

**GEXEC**

A machine-optimized execution artifact produced from a GPACK.

A GEXEC may contain hardware-specific choices such as:

- tensor layouts
- kernel selections
- expert packing
- NUMA placement
- KV strategy
- speculative decoding configuration
- hardware-specific tuning results

GPACK is portable.

GEXEC is optimized for a specific execution environment.

Both formats are still under development and should not yet be considered stable.

## Project philosophy

Guild aims to remain:

- local-first
- open source
- native C++
- MoE-focused
- transparent about memory placement and execution
- usable without a required cloud service or account

Python bindings may be provided separately, but Python is not intended to be required by the Guild runtime.

## Contributing

Guild is young and changing quickly.

Bug reports, hardware results, architecture work, kernel optimizations, documentation improvements, and compatibility testing are welcome.

Please include relevant hardware details when reporting performance or runtime issues.

Start with [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) for the build, test and formatting commands, and
[docs/architecture/README.md](docs/architecture/README.md) for the component map and the per-subsystem documents.

## Security

Please report security vulnerabilities privately using GitHub's private vulnerability reporting feature.

See [SECURITY.md](SECURITY.md).

## Support

Guild is developed and tested on personally owned hardware.

Contributions can help fund:

- additional GPUs
- larger memory configurations
- storage for large model testing
- compatibility testing
- model validation

<Insert Buy me a Coffee link here>

## License

Guild is licensed under the [MIT License](LICENSE).

Guild was originally derived from
[Strata](https://github.com/Niko1221/Strata).
The original Strata copyright and license notices are preserved as required.
