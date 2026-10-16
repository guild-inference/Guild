#!/usr/bin/env python3
"""Record public hardware/build settings and hash local model assets (no downloads)."""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import time

from run_workload_command import capture, public_environment, sha256


SOURCES = ("kq_avx2.cpp", "pool.cpp", "native_mmvq.cu", "verify_kernels.cu", "native_bf16.cu",
           "batched_prefill.cpp", "arch/x86/quants.c", "cpu_expert_workload.cpp", "ornith_prefill_bench.cpp")


def compiler_settings(build):
    """Read generated files only: root compile_commands may contain just ggml.

    Keep flags.make verbatim, including per-source ISA options and response
    files. These are generated settings, not reconstructed compiler invocations.
    """
    commands, flags, warnings = [], [], []
    database = build / "compile_commands.json"
    if database.is_file():
        commands = [row for row in json.loads(database.read_text())
                    if any(row["file"].replace("\\", "/").endswith(s) for s in SOURCES)]
    else:
        warnings.append(f"missing {database}")
    covered = {s for s in SOURCES if any(row["file"].replace("\\", "/").endswith(s) for row in commands)}
    # build.make tells us which target owns each source; no CMake reconfigure/build.
    for make in sorted(build.rglob("build.make")):
        if "_deps" in make.relative_to(build).parts:
            continue
        text = make.read_text()
        sources = [s for s in SOURCES if re.search(re.escape(s) + r"(?:\s|$)", text)]
        if not sources:
            continue
        path = make.with_name("flags.make")
        if not path.is_file():
            warnings.append(f"missing {path}")
            continue
        raw = path.read_text()
        variables = dict(re.findall(r"^(\w+)[ \t]*=[ \t]*(.*)$", raw, re.MULTILINE))
        responses = {}
        for value in variables.values():
            words = shlex.split(value)
            for i, word in enumerate(words):
                response = word[1:] if word.startswith("@") else (
                    word.split("=", 1)[1] if word.startswith("--options-file=") else (
                        words[i + 1] if word == "--options-file" and i + 1 < len(words) else None))
                if response:
                    candidate = build / response
                    if candidate.is_file():
                        responses[str(candidate)] = candidate.read_text()
                    else:
                        warnings.append(f"missing response file {candidate}")
        flags.append({"path": str(path), "sources": sources, "variables": variables,
                      "raw": raw, "response_files": responses})
        covered.update(sources)
    for source in SOURCES:
        if source not in covered:
            warnings.append(f"no compiler settings found for {source}")
    return {"compile_commands": commands, "target_compile_flags": flags,
            "compiler_settings_warnings": warnings}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build", type=Path, default=Path("build-cuda12"))
    p.add_argument("--model", type=Path, default=Path("/home/ubuntu/models/ornith-1.5-35b/Ornith-1.5-35B-Q4_K_M.gguf"))
    p.add_argument("--pack", type=Path, default=Path("/mnt/models-ssd/Strata-data/packs/ornith-1.5-35b"))
    p.add_argument("--output", type=Path, default=Path("bench/results/hybrid-workload-optimization/environment.json"))
    args = p.parse_args()
    if not args.model.is_file() or not args.pack.is_dir():
        p.error("--model must be a file and --pack must be a directory")
    data = {
        "utc_collected": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": public_environment(os.environ),
        "commands": [capture(c) for c in [
            ["uname", "-a"], ["lscpu", "-J"], ["numactl", "--hardware"],
            ["nvcc", "--version"], ["c++", "--version"], ["nvidia-smi", "-q"], ["git", "rev-parse", "HEAD"],
            ["git", "-C", "/home/ubuntu/llama.cpp", "rev-parse", "HEAD"],
            ["git", "-C", str(args.build / "_deps/guild_llamacpp-src"), "rev-parse", "HEAD"]]],
        "affinity": sorted(os.sched_getaffinity(0)), "assets": {},
        "cmake_cache": (args.build / "CMakeCache.txt").read_text(),
        **compiler_settings(args.build),
    }
    assets = [args.model] + sorted(f for f in args.pack.rglob("*") if f.is_file())
    for asset in assets:
        before = asset.stat()
        digest = sha256(asset)
        after = asset.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f"asset changed during hashing: {asset}")
        data["assets"][str(asset)] = {"bytes": before.st_size, "sha256": digest}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + "\n")
    print(f"Recorded {len(assets)} asset hashes, {len(data['compile_commands'])} compiler commands, "
          f"and {len(data['target_compile_flags'])} target flag files")


if __name__ == "__main__":
    main()
