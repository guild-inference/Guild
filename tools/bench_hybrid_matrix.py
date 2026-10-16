#!/usr/bin/env python3
"""Resumable, serial real-model matrix. Run only one instance per machine.

Baseline disables the new projection batching and host binding in the same
binary. Final uses the retained defaults. All cases generate 128 tokens at
context 32768, after one full-shaped warmup, then ten measured repetitions.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["baseline", "final"], required=True)
    p.add_argument("--prompts", type=int, nargs="+", default=[512, 2048, 4096, 8192, 16384])
    p.add_argument("--batches", type=int, nargs="+", default=[64, 128, 256, 512])
    p.add_argument("--runs", type=int, default=10)
    p.add_argument("--build", type=Path, default=Path("build-cuda12"))
    p.add_argument("--output", type=Path, default=Path("bench/results/hybrid-workload-optimization"))
    args = p.parse_args()
    for prompt in args.prompts:
        for batch in args.batches:
            name = f"{args.mode}-p{prompt}-b{batch}"
            metadata = args.output / f"{name}.command.json"
            if metadata.exists():
                result = json.loads(metadata.read_text())
                if result.get("exit_code") == 0:
                    print(f"{name}: already complete", flush=True)
                    continue
                raise RuntimeError(f"incomplete measurement: {metadata}; inspect before resuming")
            command = [sys.executable, "tools/run_workload_command.py", "--name", name, "--output", str(args.output)]
            if args.mode == "baseline":
                command += ["--env", "GUILD_PREFILL_BATCH_PROJECTIONS=0", "--env", "GUILD_HOST_AFFINITY=0"]
            command += ["--", str(args.build / "ornith_prefill_bench"), "--context", "32768", "--prompt", str(prompt),
                        "--generate", "128", "--batch", str(batch), "--runs", str(args.runs), "--full-warmup",
                        "--write-ids", str(args.output / f"prompt-{prompt}.ids"), "--csv", str(args.output / f"{name}.csv")]
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
