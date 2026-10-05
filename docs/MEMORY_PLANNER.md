# Native Execution Memory Planner

Guild features a native C++ execution memory planner that derives an architecture-independent `ExecutionPlan` from model descriptors, hardware detection, requested context length, and KV precision.

The native planner replaces the legacy heuristic Python calculations in `setup.py` with exact byte accounting and explicit prompt scratch verification.

---

## Architecture

The planning layer is organized into modular components under `include/guild/memory/` and `src/memory/`:

- **`plan.hpp`**: Defines `ExecutionPlan`, `PlanValidation`, `KvMode`, and `KvPrecision`. Implements human-readable (`to_human_string()`) and machine-readable (`to_json_string()`) serializers.
- **`kv_budget.hpp` / `kv_budget.cpp`**: Computes exact KV cell bytes across precisions (`FP16`, `INT8`, `Q4_0`, `K8V4`), full host KV buffers, persistent GPU streaming windows, and host-only staging buffers (`16,448` cells / `~32.125 MiB`).
- **`expert_budget.hpp` / `expert_budget.cpp`**: Calculates total routed expert footprints (e.g. `55.43 GiB` for Qwen3.8 UD-IQ4_XS) and distributes experts across GPU cache, pinned RAM tier, and file tiers deterministically.
- **`planner.hpp` / `planner.cpp`**: Orchestrates planning decisions using prioritized allocation rules and provides `validate()` to guarantee that prompt prefill scratch has sufficient VRAM to allocate.

---

## Priority Allocation Rules

For automatic memory planning on VRAM-constrained systems, the planner enforces the following priority order:

1. **Choose normal KV residency** if it comfortably fits with prompt scratch and safety headroom.
2. **Otherwise use host-only KV** (authoritative KV in pinned host RAM, bounded ~32.1 MiB staging on GPU).
3. **Retain full RAM expert residency** whenever host RAM permits.
4. **Size GPU expert cache only from genuinely free VRAM** (retaining 0 GPU cache slots on 8 GB cards).
5. **Retain MTP/speculative decoding** if prompt scratch still fits.
6. **Reduce prefill chunk size** (e.g. from 2048 to 512) if necessary.
7. **Disable optional acceleration features** before reducing explicit context or KV precision.
8. **Never silently alter explicitly requested context length or KV format**.

---

## Comparison with `setup.py` & Intentional Differences

| Configuration | Legacy `setup.py` | Native C++ Planner (`guild inspect --plan`) | Intentional Rationale |
| :--- | :--- | :--- | :--- |
| **GTX 1070 8 GiB + 377 GiB RAM** (262K FP16, Qwen3.8) | `--kv-host-only`, `--prefill 512`, `--spec 4`, `--resident-budget-gib 55` | `host-only KV` (6.00 GiB host, 32.1 MiB staging), all 24,576 experts in RAM (55.43 GiB), 0 on file, 0 GPU cache, spec 4, prefill 512 | Native planner produces identical proven runtime behavior while verifying that prompt scratch (~398 MiB) and safety reserve (512 MiB) have guaranteed headroom in VRAM. |
| **8 GiB GPU + 64 GiB RAM** (262K FP16, Qwen3.8) | `resident_budget_gib = 34` (subtracted fixed 24 GB for OS/cache) | 21,281 experts in RAM (48.0 GiB), 3,295 experts on file (7.43 GiB) | `setup.py` subtracted a blanket 24 GB (`UNSLOTH_RAM_LEFT_GB = 24`), needlessly relegating 21 GiB of experts to disk. The native planner reserves 10 GiB for OS/engine plus exact 6.00 GiB for host KV, maximizing RAM expert residency. |
| **A100 80 GiB + 512 GiB RAM** (262K FP16, Qwen3.8) | `--kv-streaming auto`, prefill auto | `resident KV` (6.00 GiB GPU), all 24,576 experts in GPU cache, prefill chunk 2048 | Fully GPU-resident execution selected automatically without requiring manual overrides. |
| **8 GiB GPU + 32K context FP16** | Chose 32K resident KV (768 MiB) without accounting for prefill scratch; **OOM crashed during prefill chunk allocation** | Selects host-only KV or explicitly verifies that `dense + KV + scratch + reserve <= VRAM`. If prompt scratch cannot allocate, `validate()` marks the plan invalid. | Prevents silent startup OOMs at token 0 by guaranteeing prompt-path scratch reservations. |
| **Explicit Budget Overrides** | Hardcoded string arguments in batch/shell scripts | Strongly-typed `ExecutionPlan` structure consumed by CLI, TUI, and server | Decouples memory planning from string parsing and CLI flag formatting. |

---

## CLI Inspection Usage

```bash
# Human-readable execution memory plan
guild inspect Qwen3.8-Flash-Next --plan

# Machine-readable JSON execution memory plan
guild inspect Qwen3.8-Flash-Next --plan --json
```
