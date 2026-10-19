# Contributing

Guild is young and changing quickly; bug reports, hardware results, architecture work, kernel optimizations,
documentation improvements and compatibility testing are all welcome. Include your hardware and the exact model
and settings when reporting a performance or runtime issue, and a measured number rather than an adjective.

This file is the practical guide for changing code in this repository. The architecture itself is described in
[docs/architecture/README.md](architecture/README.md) and, in more detail, in the per-subsystem documents it
links to.

## Build

The engine is compiled into one binary plus its tests. CUDA is optional: a CPU-only build is a supported
configuration and is what the CI policy job exercises.

```bash
# CUDA build (NVIDIA)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGUILD_ENABLE_CUDA=ON -DGUILD_BUILD_TESTS=ON
cmake --build build -j"$(nproc)"

# CPU-only build
cmake -S . -B build-cpu -DCMAKE_BUILD_TYPE=Release -DGUILD_ENABLE_CUDA=OFF -DGUILD_BUILD_TESTS=ON
cmake --build build-cpu -j"$(nproc)"
```

AMD users build with `-DGUILD_ENABLE_HIP=ON` (see `cmake/hip_backend.cmake` for the supported architectures).
The build has no network dependency once the pinned backend sources are fetched; `cmake -L` lists every option,
and the comments next to `option()` in `CMakeLists.txt` say what each one does.

## Test

```bash
ctest --test-dir build --output-on-failure            # the C++ suite
python -m unittest discover -s tools -p 'test_*.py'   # the Python policy/tooling tests
```

The C++ suite covers units (containers, planner, registry, ABI), parity harnesses (kernels against reference
implementations, model output against a golden run) and lifecycle tests (packages, sessions, multi-session). Some
tests need fixtures that are not in git - a model file, a `pack/` directory, a benchmark log - and are skipped or
reported as unavailable when those are missing. Read the skip reason rather than assuming a silent pass.

When you add a kernel or a runner, add the parity test that pins its numbers. Bitwise equality where the
reference is deterministic, a tolerance with the measurement in a comment where it is not.

## Formatting

First-party C/C++/CUDA is formatted with clang-format 18.1.8:

```bash
python tools/format_code.py --check     # what CI runs
python tools/format_code.py             # apply
python tools/format_code.py --diff      # show what would change
```

[docs/quality/CODE_STYLE.md](quality/CODE_STYLE.md) documents the conventions, the two justified exceptions to the
120-column limit, and what is excluded (vendored trees, generated tables, the Intel DPCT output, archived
artifacts). Python, CMake, Markdown and JSON are not machine-formatted; match the file you are editing.

## Where to change what

| Change | Start at |
| --- | --- |
| A new model architecture | `include/guild/model/runner.hpp` and an existing runner in `src/model/`; see [MODEL_RUNNERS.md](architecture/MODEL_RUNNERS.md) |
| A new kernel or a faster implementation of one | `src/kernels/`, then register it in `src/kernels/registry.cpp` with its capability requirements |
| A container or format change | `src/artifact/` (`gpack.cpp`, `gguf_reader.cpp`), [GPACK_FORMAT.md](architecture/GPACK_FORMAT.md) |
| Tuning, layout or placement decisions | `src/gexec/`, [GEXEC_CONTRACT.md](architecture/GEXEC_CONTRACT.md) |
| Memory placement | `src/memory/planner.cpp`; the planner's decisions are validated against live memory at load |
| Expert tiers, caching, NUMA | `src/core/expert_source.cpp`, `expert_cache.cpp`, `src/kernels/cpu/pool.cpp` |
| The API, CLI or server | `src/server/`, `src/cli/guild_main.cpp`, and `serve/` for the Python layer |
| A package or ABI surface | `include/guild/abi/`, `src/abi/`; ABI changes need a version bump and both sides updated |

## House rules

* **Do not mix formatting with behaviour.** A formatting-only change stays formatting-only; a functional change
  does not reformat the file around it.
* **Keep the hot path free of dispatch.** Resolution happens at load; a token or layer loop gets function
  pointers. See the note at the end of [EXECUTION_LIFECYCLE.md](architecture/EXECUTION_LIFECYCLE.md).
* **Comments explain why.** The code says what it does. Constraints, hazards, ordering requirements and
  ownership rules are what a reader cannot recover from the code, and that is what belongs in a comment.
* **Docs carry measured numbers.** A performance claim names the hardware, the model and the settings it was
  measured on; state the result you measured, and say plainly when something is unverified or unavailable.
* **Preserve determinism.** Kernel selection is recorded, not re-derived; a tie is broken by an explicit rule, not
  by map iteration order.
* **Never commit** build output, model artifacts, caches, machine-local configuration or anything with a secret
  in it. Keep the tree free of generated files that the build can regenerate.
