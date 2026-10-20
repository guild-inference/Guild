#!/usr/bin/env python3
"""Serial populated-context Guild measurements, using the existing workload recorder.

Each prompt really passes through prefill. Capacity includes generation headroom;
it is never reported as populated history. Failed/time-limited cases retain their
logs and have no throughput value. No model downloads or privileged commands.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from summarize_workload_benchmarks import summarize_directory


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exe", type=Path, default=Path("build-cuda12/ornith_prefill_bench"))
    p.add_argument("--model", required=True)
    p.add_argument("--gpack", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--prompts", type=int, nargs="+", default=[512, 4096, 8192, 16384, 32768, 65536, 131072, 262144])
    p.add_argument("--capacity", type=int, default=0, help="fixed allocation, or 0 for prompt + generation rounded to 4096")
    p.add_argument("--max-context", type=int, default=262144, help="runner/model's supported maximum, including output")
    p.add_argument("--generate", type=int, default=64)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--spec", type=int, default=0)
    p.add_argument("--timeout", type=float, default=3600)
    p.add_argument("--full-warmup", action="store_true")
    p.add_argument("--env", action="append", default=[])
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    a = p.parse_args()
    if (min([a.generate, a.runs, a.batch, a.max_context, *a.prompts]) < 1 or a.capacity < 0
            or not 0 < a.timeout < float("inf") or a.label in (".", "..") or Path(a.label).name != a.label):
        p.error("invalid positive counts, capacity, timeout or label")
    if len(set(a.prompts)) != len(a.prompts):
        p.error("duplicate prompt lengths")
    controlled = {"--model", "--gpack", "--prompt", "--context", "--generate", "--runs", "--batch", "--spec",
                  "--csv", "--startup-json", "--write-ids", "--full-warmup", "--sessions", "--json-summary"}
    if any(word.split("=", 1)[0] in controlled for word in a.extra):
        p.error("--extra cannot override curve-controlled settings")
    a.output.mkdir(parents=True, exist_ok=True)
    manifest = a.output / (a.label + "-curve.json")
    if manifest.exists():
        p.error("curve exists; choose a new output/label to preserve earlier results")
    report = {"schema_version": 1, "label": a.label, "model": a.model,
              "population": "real_prompt_prefill", "full_warmup": a.full_warmup, "cases": []}
    manifest.write_text(json.dumps(report, indent=2) + "\n")
    for prompt in a.prompts:
        capacity = a.capacity or min(a.max_context, ((prompt + a.generate + 4095) // 4096) * 4096)
        name = f"{a.label}-p{prompt}"
        row = {"name": name, "prompt_tokens": prompt, "context_capacity": capacity}
        report["cases"].append(row)
        if prompt + a.generate > capacity or capacity > a.max_context:
            row.update(status="unsupported", reason="prompt plus requested output exceeds supported capacity")
        else:
            prefix = a.output / name
            cmd = [sys.executable, str(Path(__file__).with_name("run_workload_command.py")),
                   "--name", name, "--output", str(a.output), "--timeout", str(a.timeout)]
            for env in a.env:
                cmd += ["--env", env]
            cmd += ["--", str(a.exe), "--model", a.model, "--gpack", str(a.gpack),
                    "--prompt", str(prompt), "--context", str(capacity), "--generate", str(a.generate),
                    "--runs", str(a.runs), "--batch", str(a.batch), "--spec", str(a.spec), "--ignore-eos",
                    "--csv", str(prefix) + ".csv", "--startup-json", str(prefix) + ".startup.json",
                    "--write-ids", str(prefix) + ".prompt.ids"]
            if a.full_warmup:
                cmd += ["--full-warmup"]
            cmd += a.extra
            result = subprocess.run(cmd)
            row.update(status="complete" if result.returncode == 0 else "failed", exit_code=result.returncode,
                       command_metadata=str(prefix) + ".command.json")
        manifest.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    summary = summarize_directory(a.output)
    (a.output / (a.label + "-summary.json")).write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return int(any(row["status"] == "failed" for row in report["cases"]))


if __name__ == "__main__":
    raise SystemExit(main())
