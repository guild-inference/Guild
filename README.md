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
| GLM MoE | Planned |
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
