# Guild C++/CUDA code style

This is the formatting and comment standard for Guild's own C++/CUDA sources, the tooling that enforces it, and
what is deliberately left out.  It covers `include/`, `src/`, `tests/`, `tools/`, `plugins/`, `bench/` and `sycl/`
(the C++ parts of the Intel port).  It does not cover vendored or generated code - see
[What is excluded](#what-is-excluded).

Source of truth: [`.clang-format`](../../.clang-format) in the repository root, with
[`.clang-format-ignore`](../../.clang-format-ignore) naming what clang-format must not touch.
[`tools/format_code.py`](../../tools/format_code.py) applies and checks it.

---

## 1. The pinned formatter

Guild formats with **clang-format 18.1.8**, the same binary on every platform:

```bash
python -m pip install clang-format==18.1.8     # ships the LLVM 18.1.8 binary as a wheel
```

Formatting output changes between clang-format majors, so `tools/format_code.py` refuses to run on anything but
major version 18, and the CI job installs exactly that version.  On a machine with the distro package, point the
tool at it instead of the wheel:

```bash
CLANG_FORMAT=/usr/bin/clang-format-18 python tools/format_code.py --check
```

## 2. Running it

```bash
python tools/format_code.py                  # rewrite every eligible file in place
python tools/format_code.py --check          # exit 1 and list what is not formatted (what CI runs)
python tools/format_code.py --check --diff   # the same, with the diff clang-format would apply
python tools/format_code.py src/kernels      # restrict to one or more path prefixes
```

The file list comes from `git ls-files`, so untracked scratch files, build directories and vendored trees are
never rewritten.  The tool is idempotent: running it twice produces the same bytes, and a formatted tree passes
`--check` without changes.

Editors pick the configuration up automatically (`clang-format --style=file`, or the IDE's "format on save"),
which is why the rules live in `.clang-format` and not in a script.

The CI job `format` (`.github/workflows/ci.yml`) runs `--check` on every push and pull request, so a hand-written
reformat is caught in review rather than in a later merge.

## 3. The conventions, and why

Most of the configuration is deliberately empty because the `LLVM` preset already matches the tree (attached
braces, 4-space blocks, pointer with the type, `(type) expr` C-style casts, operators left at the end of a
wrapped line).  The options that are _not_ the preset default are the ones a preset gets wrong for this codebase:

| Option | Value | Why |
| --- | --- | --- |
| `ColumnLimit` | 120 | The tree already aligns SIMD macro continuations on column 120 and 10 455 lines sit between 100 and 120 characters; a 100-column rewrap rewrites a tenth of the tree for no gain in the kernel code. |
| `SortIncludes` | `Never` | Include order is load-bearing: `windows.h` must precede the Win32 headers it configures, some toolchains need the CUDA headers before the C library, and a translation unit's own header comes first.  Reordering is a separate, reviewed change. |
| `IndentPPDirectives` | `None` | `#if` / `#ifdef` stay in column 0 as in the existing sources (and as `cmake` and the CUDA headers expect). |
| `AllowShortIfStatementsOnASingleLine` | `Never` | Control flow is read at a glance: `if (x) return;` is written on two lines.  Single-line blocks remain for empty bodies only. |
| `AllowShortFunctionsOnASingleLine` | `None` | Function bodies start on their own line, including one-line accessors; the tree writes them that way. |
| `AllowShortEnumsOnASingleLine` | `false` | Enum tables stay vertical, one value per line, which keeps their comments and values aligned. |
| `SpaceAfterCStyleCast` | `true` | The tree writes casts as `(int) value` (6 499 occurrences against 67 of `(int)value`). |
| `PointerAlignment` | `Left` | `const float* row`; a declaration reads as one type followed by one name. |
| `SpacesBeforeTrailingComments` | 1 | `} // namespace`, `ms = 12; // measured with ...`. |
| `ReflowComments` | `true` | Prose comments are wrapped to the line width; clang-format leaves tables, code blocks and ASCII art alone. |
| `FixNamespaceComments` | `true` | `} // namespace guild::core` closes long namespace blocks visibly. |
| `InsertNewlineAtEOF` | `true` | Every file ends with exactly one newline. |

Choices that are intentionally _not_ made:

- **`AlignConsecutiveAssignments`/`AlignConsecutiveDeclarations` stay off.**  Aligned `=` runs are pleasant in a
  short table, but clang-format aligns any run of consecutive lines, which produces large, unstable whitespace
  changes in ordinary code.  Tables that read better aligned (ABI status enums, expert-pack layouts) keep their
  alignment only by hand, in a `// clang-format off` block.
- **No include sorting or grouping.**  See `SortIncludes` above.
- **No `InsertBraces`.**  Adding braces to every single-statement `if` is a behavioral review of its own, not a
  formatting pass.

### Blank lines, sections and comments

- One blank line between functions (`SeparateDefinitionBlocks: Leave` keeps the existing spacing).
- No leading blank line inside a block (`KeepEmptyLinesAtTheStartOfBlocks: false`), at most one consecutive
  blank line (`MaxEmptyLinesToKeep: 1`).
- Comments explain **why**, not what: the code says what it does, the comment says which invariant makes it
  safe, or which measurement made it fast.  See the AGENTS.md rule: plain words, measured numbers with what they
  were measured on, no claims without a measurement.

## 4. What is excluded

| Path | Why |
| --- | --- |
| `third_party/**` | Vendored: the pinned llama.cpp checkout and the external backend repositories have their own style and their own `.clang-format`. |
| `sycl/third_party/**` | Vendored ggml headers carried by the Intel port. |
| `sycl/include/dpct/**` | Intel's DPCT compatibility headers, shipped by the port, not written by us. |
| `bench/results/**` | Archived measurement artifacts (captured assembly, codegen dumps, CSV logs), not source. |
| `src/kernels/*_oracle_vectors.inc`, `sycl/src/kernels/*_host_tables.inc` | Generated tables (`// GENERATED by ... - DO NOT EDIT BY HAND`); reformatting them would be undone by the next regeneration. |
| `*.asm` under `bench/results/` | Captured compiler output, not source.  (There is no hand-written assembly in the tree.) |

`.clang-format-ignore` carries the same list, so editors and `git clang-format` agree with the tool.

## 5. Local exceptions

Two kinds of place exist where the formatting tool must keep its hands off:

```cpp
// clang-format off: a run of macro invocations is re-wrapped on every pass, so it is written by hand.
GUILD_Q6W(1)
GUILD_Q6W(2)
// clang-format on
```

1. **clang-format cannot stabilize the region.**  Runs of macro invocations expand to statements whose
   line-breaking oscillates between passes (clang-format 18.1.8, `sycl/src/kernels/cuda/native_mmvq.dp.cpp`).
   The block is written by hand and marked, so the next `--check` run still passes.
2. **An aligned table would lose its meaning.**  A short block of `NAME   = value,` lines whose alignment carries
   the grouping can be marked instead of reflowed.

Markers are never used to avoid cleaning up difficult code.  Both blocks above carry a one-line comment saying
which of the two cases they are; a marker without such a comment is a review finding.

## 6. Other languages

clang-format is only run on C/C++/CUDA (`SOURCES` in `tools/format_code.py`).  The other first-party languages
keep their own conventions:

- **Python** (`tools/`, `serve/`, `setup.py`): 4-space indentation, `snake_case`, a module docstring that says
  what the tool is for and how to run it, `from __future__ import annotations` in newer tools.  No auto-formatter
  is enforced; a repo-wide Python reformat is a separate change (see the Phase 5.5 audit report).
- **CMake**: 2-space indentation and lowercase commands, as in the existing `CMakeLists.txt` files.  Option
  groups are separated by a blank line and introduced by a comment naming the issue or phase they belong to.
- **Markdown** (`docs/`): plain words, measured numbers with the machine they were measured on, one sentence per
  line where the existing documents do so, no decorative banners.
- **JSON** (`schemas/`, `bench/results/`): 2-space indentation; schemas are checked by their validators, formatted
  artifacts are never hand-edited.
