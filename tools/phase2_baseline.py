#!/usr/bin/env python3
"""Preserve the Phase 1 benchmark binary and measure it serially before extraction."""
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "bench/results/external-kernels/phase1"
    output.mkdir(parents=True, exist_ok=True)
    binary = output / "ornith_prefill_bench"
    if not binary.exists():
        shutil.copy2(root / "build-cuda12/ornith_prefill_bench", binary)
    identity = {
        "commit": "42018db2e63e7febbdb7bcf58a6392166518cdc1",
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "context": 32768, "batch": 256, "generate": 128, "runs": 10,
    }
    (output / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    env = os.environ.copy()
    env["GUILD_DATA_DIR"] = str(output / "data")
    results = []
    for prompt in (512, 4096, 8192, 16384):
        dest = output / f"p{prompt}.csv"
        command = [str(binary), "--context", "32768", "--prompt", str(prompt),
                   "--generate", "128", "--batch", "256", "--runs", "10",
                   "--full-warmup", "--csv", str(dest)]
        with (output / f"p{prompt}.log").open("w") as log:
            subprocess.run(command, cwd=root, env=env, stdout=log, stderr=log, check=True)
        with dest.open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 10 and all(int(r["generated_tokens"]) == 128 for r in rows)
        results.append({"prompt": prompt, "prefill": statistics.median(float(r["prefill_tps"]) for r in rows),
                        "decode": statistics.median(float(r["decode_tps"]) for r in rows)})
        (output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    (output / "complete.json").write_text(json.dumps(identity, indent=2) + "\n")


if __name__ == "__main__":
    main()
