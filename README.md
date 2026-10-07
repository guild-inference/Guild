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

## Quick start

Guild currently builds from source.

```bash
git clone https://github.com/guild-inference/Guild.git
cd Guild

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGUILD_ENABLE_CUDA=ON

cmake --build build -j$(nproc)
```

Then:

```bash
./build/guild --help
```

The long-term installation path will be a native installer and prebuilt releases.

## Commands

```text
guild pull <model>       Download a model
guild import <path>      Import an existing local model
guild list               List installed models
guild show <model>       Inspect a model
guild run <model>        Run a model interactively
guild serve <model>      Start the OpenAI-compatible server
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