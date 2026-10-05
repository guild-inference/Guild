# Python Functionality Inventory and Native Migration Plan

Guild's long-term architectural goal is a 100% native C++ runtime with zero Python dependency for installation, serving, inference, and CLI operations.

This document inventories every Python component in the repository into four categories:
- **REPLACE NOW**: High-priority runtime components actively being migrated to native C++.
- **REPLACE LATER**: Secondary runtime / server components scheduled for subsequent migration phases.
- **DELETE**: Obsolete scripts or legacy wrappers that can be retired once native paths land.
- **KEEP AS DEV TOOL**: Offline development, quantization research, synthetic test generators, and documentation tooling.

---

## 1. Inventory by Category

### REPLACE NOW (Phase 1: CLI, Hardware, Model Metadata & Archetypes)
- `chat.py` → Replaced by native `guild run <model>` interactive REPL.
- Hardware probing in `setup.py` (`probe_gpu`, `cpu_features`, `ram_total`) → Replaced by C++ `guild::cli::detect_hardware()` (`src/cli/hardware.cpp`).
- Model GGUF inspection in `tools/gguf_reader.py` → Replaced by C++ `guild inspect` using `GgufFile` and `ModelDescriptor`.
- Model archetype and geometry decoding in `setup.py` and `tools/` → Replaced by C++ `ArchetypeRegistry` (`src/model/archetype.cpp`).

### REPLACE LATER (Phase 2 & 3: Server, Memory Planning & Downloader)
- `serve/server.py`, `serve/responses.py`, `serve/structured.py` → Replaced by native C++ HTTP server (`include/guild/server/`, `src/server/`) supporting `/health`, `/v1/models`, `/v1/chat/completions`, and `/v1/completions` (JSON and SSE streaming). Python implementations retained temporarily as behavioral reference.
- `serve/runconfig.py` → Replace with C++ engine configuration loader.
- `serve/telemetry.py` → Replaced by native C++ `guild::server::Telemetry` (`src/server/telemetry.cpp`).
- `serve/winjob.py` → Replace with native OS process management.
- `serve/mcp.py` → Replace with native C++ MCP server handler.
- Memory budget calculation in `setup.py` (`compute_budget`, `choose_model`) → Replaced by native C++ `guild::memory::MemoryPlanner` (`include/guild/memory/`, `src/memory/`).
- Model downloading in `setup.py` → Replace with native curl/libcurl download engine in `guild pull` / `guild run`.

### DELETE (Obsolete / Redundant once Native CLI Lands)
- `START-HERE.bat` / `setup.sh` (Python wrappers) → Obsolete once native `guild` single-binary installer is shipped.
- `serve/mcp_fake_server.py` → Ephemeral test fixture to be removed.
- `sycl/setup_intel.py` → Unify under generic C++ hardware detection and setup.

### KEEP AS DEV TOOL (Development, Benchmarking & Weight Conversion)
- `tools/iq_pack.py`, `tools/mtp_pack.py`, `tools/ple_fp8_pack.py`, `tools/embd_bf16_pack.py` → Offline weight packing utilities for creating release shards.
- `tools/calibrate.py`, `tools/make_profile.py` → Offline kernel profile and calibration tools.
- `tools/needle_bench.py` → Long-context retrieval benchmark utility.
- `docs/media/make_figures.py` → Documentation asset generator.
- `tools/test_*.py` → Offline Python regression suites for pack formats and legacy test cases.

---

## 2. Seven-Stage Migration Roadmap

```
1. CLI/config parsing -> C++          [DONE: native guild binary with inspect, serve, ps, bench, run]
        │
2. Hardware detection -> C++          [DONE: guild::cli::detect_hardware]
        │
3. Memory planning -> C++             [DONE: guild::memory::MemoryPlanner]
        │
4. Model metadata/archetypes -> C++   [DONE: ModelDescriptor & ArchetypeRegistry]
        │
5. HTTP/OpenAI-compatible server -> C++ [DONE: native HTTP / SSE server in guild serve]
        │
6. Downloader/install logic -> C++
        │
7. Remove Python runtime dependencies
```
