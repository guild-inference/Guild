#!/usr/bin/env python3
"""Inventory CUDA translation units, launch definitions, includes and build inputs.

This source inventory deliberately keeps mixed orchestration files separate from
arithmetic candidates. It is not a C++ AST or proof of runtime execution.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SPECIALIZED = {"gr", "fused_gr", "native_gr_norm", "native_gr_postops", "ple", "native_ple_postops", "cvec"}
GROUPS = {
    "dequant_s2": "dense/quantization", "dequant_bf16": "dense/quantization",
    "s2_gemv": "dense/canonical", "s_gemv": "dense/canonical", "s2_gemv_quads": "dense/canonical",
    "s2_gemv_fast": "dense/canonical", "s2_gemv_q8": "dense/canonical",
    "bf16_gemv": "dense/bf16", "native_bf16": "dense/bf16", "native_mmvq": "dense/native-mmvq",
    "iq_kernels": "dense/iq", "quantize_act": "dense/quantization", "router_top10": "moe/routing",
    "native_router": "moe/routing", "s2_expert_grouped": "moe/canonical", "native_moe": "moe/native",
    "shared_expert": "moe/shared", "elementwise": "elementwise/doorbell",
    "rope": "attention/rope", "native_rope": "attention/rope", "qsa": "attention/qsa",
    "qsa_select": "attention/selection", "native_qsa_indexer": "attention/indexer",
    "native_qsa_score": "attention/scoring", "native_qsa": "attention/normalization",
    "native_flash_attn": "attention/flash", "qsa_decode_attn": "attention/decode",
    "qsa_prompt_attn": "attention/prefill", "kv_q8": "kv/int8", "kv_q4": "kv/q4",
    "kv_stream": "kv/streaming", "gdn": "recurrent/gdn", "native_gdn": "recurrent/gdn",
    "fused_gdn": "recurrent/gdn", "native_gdn_preprocess": "recurrent/gdn",
    "verify_kernels": "verification/device-primitives", "sampler": "sampling/device-primitives",
}


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def classify(path):
    if path.parts[:2] == ("src", "kernels"):
        if path.stem in SPECIALIZED:
            return "B", "guild-kernels-cuda: isolated qwen specialization", "specialized/" + path.stem
        if path.stem not in GROUPS:
            raise ValueError(f"unclassified kernel translation unit: {path}")
        return "A", "guild-kernels-cuda", GROUPS[path.stem]
    if path.parent.as_posix() == "src/prefill":
        if "parity" in path.stem:
            return "D", "Core test retained; standalone backend test consumer", "tests"
        if path.stem == "ggml_cuda_host":
            return "D", "guild-kernels-cuda private pinned GGML context", "ggml/context"
        return "A/B" if path.stem == "kernels" else "A", "guild-kernels-cuda", "prefill/" + path.stem
    if path.name == "remote_expert_opt.cu":
        return "A+C", "Core orchestration; extract reduce_experts and copy_cpu_rows only", "multi-device"
    if path.name == "device.cu":
        return "C/D", "Core device allocation and diagnostic poison kernel", "device/memory"
    if path.name == "pinned.cu":
        return "C", "Core pinned allocation, mapping and lifetime", "host/memory"
    raise ValueError(f"unclassified CUDA translation unit: {path}")


def inventory():
    units = []
    for path in sorted((ROOT / "src").rglob("*.cu")):
        relative = path.relative_to(ROOT)
        category, owner, group = classify(relative)
        text = path.read_text()
        units.append({"source": relative.as_posix(), "sha256": sha(path), "category": category,
                      "destination": owner, "group": group, "lines": len(text.splitlines()),
                      "includes": re.findall(r'^\s*#\s*include\s*[<"]([^>"\n]+)', text, re.MULTILINE),
                      "device_kernels": [{"name": match[1], "line": text.count("\n", 0, match.start()) + 1}
                                         for match in re.finditer(r'__global__\s+(?:__launch_bounds__\([^\n]+?\)\s+)?void\s+(\w+)\s*\(', text)],
                      "host_definitions": sorted(set(re.findall(
                          r'^\s*(?:void|bool|int|int64_t|uint64_t|size_t|std::size_t)\s+([\w:]+)\s*\([^;{}]*\)\s*(?:noexcept\s*)?\{',
                          re.sub(r'/\*.*?\*/|//[^\n]*', '', text, flags=re.DOTALL), re.MULTILINE)))})
    return units


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshot-build", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    units = inventory()
    report = {"core_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "cuda_units": len(units), "device_kernel_definitions": sum(len(u["device_kernels"]) for u in units),
              "units": units}
    if args.snapshot_build:
        build = args.snapshot_build.resolve()
        snapshots = {}
        for name in ("guild", "ornith_prefill_bench", "external_runner_parity_test"):
            source, dest = build / name, output / "reference" / name
            dest.parent.mkdir(exist_ok=True)
            if dest.exists() and sha(dest) != sha(source):
                raise RuntimeError("refusing to replace a preserved executable: " + str(dest))
            if not dest.exists():
                shutil.copy2(source, dest)
            snapshots[name] = sha(dest)
        report["reference_binaries"] = snapshots
        for name in ("CMakeCache.txt", "compile_commands.json", "build-info.json"):
            if (build / name).is_file():
                shutil.copy2(build / name, output / "reference" / name)
    (output / "ownership.json").write_text(json.dumps(report, indent=2) + "\n")
    rows = ["| Source | Category | Intended destination | Group |", "|---|---|---|---|"]
    rows.extend(f"| `{u['source']}` | {u['category']} | {u['destination']} | {u['group']} |" for u in units)
    (output / "ownership-table.md").write_text("\n".join(rows) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "units"}, indent=2))


if __name__ == "__main__":
    main()
