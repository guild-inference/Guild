#!/usr/bin/env python3
"""Format (or check) Guild's first-party C++/CUDA sources with the pinned clang-format.

The file list comes from `git ls-files` and .clang-format-ignore, so vendored trees (third_party/, the DPCT
headers), generated tables and archived measurement artifacts are never rewritten.  See
docs/quality/CODE_STYLE.md for the conventions, the pinned tool version, and what is excluded and why.

    python tools/format_code.py                  # rewrite every eligible file in place
    python tools/format_code.py --check          # list files that are not formatted; exit 1 if any
    python tools/format_code.py --check --diff   # also print the diff clang-format would apply
    python tools/format_code.py src/kernels      # restrict to files under one or more path prefixes

CLANG_FORMAT names the binary to use; otherwise `clang-format` then `clang-format-18` are looked up on PATH.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import difflib
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hh", ".hpp")
# Kept in sync with .clang-format-ignore: the check has to hold for clang-format versions whose ignore-file
# support is missing or incomplete, and `git ls-files` is what makes the list reproducible.
EXCLUDED_PREFIXES = ("third_party/", "sycl/third_party/", "sycl/include/dpct/", "bench/results/")
EXPECTED_MAJOR = 18


def find_clang_format() -> str:
    for name in (os.environ.get("CLANG_FORMAT"), "clang-format", "clang-format-18"):
        if name and (path := shutil.which(name)):
            return path
    sys.exit("clang-format was not found: install the pinned version with "
             "`python -m pip install clang-format==18.1.8` or set CLANG_FORMAT")


def version_of(binary: str) -> tuple[int, int, int]:
    text = subprocess.run([binary, "--version"], capture_output=True, text=True, check=True).stdout
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", text)
    if not match:
        sys.exit(f"cannot read a version out of `{binary} --version`: {text.strip()!r}")
    return tuple(int(part) for part in match.groups())


def tracked_sources(prefixes: list[str]) -> list[Path]:
    """Every tracked first-party C/C++/CUDA file, optionally restricted to path prefixes."""
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    files = []
    for name in out.split("\0"):
        if not name or Path(name).suffix not in SOURCES:
            continue
        if name.startswith(EXCLUDED_PREFIXES):
            continue
        if prefixes and not name.startswith(tuple(prefixes)):
            continue
        files.append(ROOT / name)
    return sorted(files)


def missing_prefixes(prefixes: list[str]) -> list[str]:
    """Prefixes with no eligible file under them: a typo must not look like a clean tree."""
    return [p for p in prefixes if not tracked_sources([p])]


def format_one(binary: str, path: Path) -> tuple[Path, bytes, bytes]:
    """Return (path, current bytes, formatted bytes)."""
    current = path.read_bytes()
    formatted = subprocess.run([binary, "--style=file", str(path)], capture_output=True, check=True).stdout
    return path, current, formatted


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", help="path prefixes, relative to the repository root (default: all)")
    ap.add_argument("--check", action="store_true", help="do not write; fail if a file is not formatted")
    ap.add_argument("--diff", action="store_true", help="with --check, print the diff for each unformatted file")
    ap.add_argument("--jobs", type=int, default=0, help="parallel clang-format processes (default: CPU count)")
    args = ap.parse_args()

    prefixes = [p.strip("./") + "/" if p != "." else "" for p in args.paths]
    unknown = missing_prefixes(prefixes)
    if unknown:
        sys.exit(f"no tracked C/C++ sources under: {', '.join(unknown)}")

    binary = find_clang_format()
    version = version_of(binary)
    if version[0] != EXPECTED_MAJOR:
        sys.exit(f"{binary} is clang-format {'.'.join(map(str, version))}; Guild pins major version "
                 f"{EXPECTED_MAJOR} (formatting output differs between majors)")
    files = tracked_sources(prefixes)
    if not files:
        sys.exit("no tracked C/C++ sources to format")

    jobs = args.jobs or min(32, (os.cpu_count() or 4))
    unformatted: list[Path] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        for path, current, formatted in pool.map(lambda p: format_one(binary, p), files):
            if current == formatted:
                continue
            unformatted.append(path)
            if not args.check:
                path.write_bytes(formatted)
            elif args.diff:
                diff = difflib.unified_diff(
                    current.decode("utf-8", "replace").splitlines(keepends=True),
                    formatted.decode("utf-8", "replace").splitlines(keepends=True),
                    fromfile=str(path.relative_to(ROOT)), tofile=f"{path.relative_to(ROOT)} (clang-format)")
                sys.stdout.writelines(diff)

    what = "would be reformatted" if args.check else "reformatted"
    if unformatted:
        if args.check and not args.diff:
            for path in unformatted:
                print(path.relative_to(ROOT))
        print(f"{len(unformatted)} of {len(files)} files {what} by clang-format {'.'.join(map(str, version))}")
        if args.check:
            print("run `python tools/format_code.py` and commit the result (see docs/quality/CODE_STYLE.md)")
            return 1
    else:
        print(f"all {len(files)} files are formatted (clang-format {'.'.join(map(str, version))})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
