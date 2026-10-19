# AGENTS.md

Guild runs the Qwen3.8-Flash-Next mixture-of-experts model (and its Coder, Swift 1.5 and Unsloth variants) on a
normal PC: one NVIDIA or AMD graphics card plus system RAM, on Windows or Linux. It has a C++/CUDA/HIP engine
(`src/`, `include/`), a Python server with an OpenAI- and Anthropic-compatible API and a web app (`serve/`), and a
one-click installer (`setup.py`, started by `START-HERE.bat` / `setup.sh`).

## Installing Guild for a user

Follow [README.md](README.md) (Quick start, Serving): clone, `cmake -S . -B build -DGUILD_ENABLE_CUDA=ON`,
`cmake --build build -j`, then `./build/guild run <model>` or `./build/guild serve <model>`. The one-click
installer (`setup.py`, started by `START-HERE.bat` / `setup.sh`) writes the model, the engine and a run config
that `serve/server.py` reads. Never expose the server beyond `127.0.0.1` without the Python server's `--api-key`.
Model downloads and the local model store are `guild pull` / `guild list` / `guild show`. MCP servers for the web
app are listed in the run config and implemented in `serve/mcp.py` (stdio and Streamable HTTP transports).

## Working on the code

- Start at [docs/architecture/README.md](docs/architecture/README.md): the component map, the layering rule and
  an index of every architecture document. [docs/architecture/EXECUTION_LIFECYCLE.md](docs/architecture/EXECUTION_LIFECYCLE.md)
  follows one process from load to decode; [docs/architecture/MODULAR_RUNTIME.md](docs/architecture/MODULAR_RUNTIME.md)
  has the phase status; the ABI, package and container contracts are in
  [docs/architecture/KERNEL_ABI.md](docs/architecture/KERNEL_ABI.md),
  [docs/architecture/GKERNEL_FORMAT.md](docs/architecture/GKERNEL_FORMAT.md),
  [docs/architecture/GKMODULE.md](docs/architecture/GKMODULE.md),
  [docs/architecture/GPACK_FORMAT.md](docs/architecture/GPACK_FORMAT.md) and
  [docs/architecture/GEXEC_CONTRACT.md](docs/architecture/GEXEC_CONTRACT.md); runner selection is in
  [docs/architecture/MODEL_RUNNERS.md](docs/architecture/MODEL_RUNNERS.md).
- Measured numbers live in [docs/performance/](docs/performance/) (benchmarks, profiling, parity) and
  `bench/results/`; audits in [docs/quality/](docs/quality/); release pipelines in
  [docs/releases/PIPELINES.md](docs/releases/PIPELINES.md). Server or startup problems usually end up in
  [docs/CORRECTNESS_RECOVERY.md](docs/CORRECTNESS_RECOVERY.md).
- AMD (HIP) and multi-GPU: the HIP backend is `-DGUILD_ENABLE_HIP=ON` (`cmake/hip_backend.cmake`), the second-GPU
  expert tier is `include/guild/core/peer_experts.hpp`, and the ownership boundary between this repository and
  the external backends is [docs/architecture/EXTERNAL_BACKENDS.md](docs/architecture/EXTERNAL_BACKENDS.md).
- Build, test and change guidance: [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md).
- Setup's own tests run without a GPU or downloads: `python tools/test_setup_<name>.py` (for example
  `tools/test_setup_amd.py`, `tools/test_setup_choices.py`).
- Formatting: first-party C/C++/CUDA is formatted with clang-format 18.1.8; check it with
  `python tools/format_code.py --check`, and read [docs/quality/CODE_STYLE.md](docs/quality/CODE_STYLE.md) for the
  conventions, the pinned version, and what is excluded (vendored trees, generated tables, archived artifacts).
- Keep the docs' style: plain words, measured numbers with what they were measured on, no claims without a
  measurement.
