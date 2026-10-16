#!/usr/bin/env python3
"""Durable serial launcher for this hardware-owner's requested matrix only.

Existing tools own raw command metadata and refuse incomplete measurements.
This launcher never repairs, replaces, or deletes those measurements.
"""
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
STATE = OUT / "hardware-execution-state.json"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_state(state):
    state["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    STATE.write_text(json.dumps(state, indent=2) + "\n")


def main():
    with (OUT / "hardware-execution.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        benchmark = ROOT / "build-cuda12/ornith_prefill_bench"
        identity = digest(benchmark)
        state = {"pid": os.getpid(), "status": "running", "completed_groups": [],
                 "benchmark_sha256": identity, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        write_state(state)
        groups = [("baseline", prompt, [256]) for prompt in [512, 2048, 4096, 8192, 16384]]
        groups += [("final", prompt, [64, 128, 256, 512]) for prompt in [512, 2048, 4096, 8192, 16384]]
        groups += [("baseline", 512, [64, 128, 512])]
        try:
            for mode, prompt, batches in groups:
                label = f"{mode}-p{prompt}-b" + ",".join(map(str, batches))
                state["active_group"] = label
                write_state(state)
                if digest(benchmark) != identity:
                    raise RuntimeError("Benchmark binary changed during matrix; stop for inspection")
                gpu = subprocess.check_output([
                    "nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"
                ], text=True)
                if gpu.strip():
                    raise RuntimeError("GPU already has an active compute process; stop for inspection")
                command = [sys.executable, "tools/bench_hybrid_matrix.py", "--mode", mode,
                           "--prompts", str(prompt), "--batches", *map(str, batches),
                           "--runs", "10", "--output", str(OUT)]
                print(f"START {label}", flush=True)
                subprocess.run(command, cwd=ROOT, check=True)
                for batch in batches:
                    name = f"{mode}-p{prompt}-b{batch}"
                    metadata = json.loads((OUT / f"{name}.command.json").read_text())
                    with (OUT / f"{name}.csv").open(newline="") as stream:
                        rows = list(csv.DictReader(stream))
                    if (metadata.get("exit_code") != 0 or metadata.get("wrapper_exit_code", 0) != 0
                            or "--full-warmup" not in metadata["command"]
                            or sorted(int(row["run"]) for row in rows) != list(range(10))
                            or any(int(row["generated_tokens"]) != 128 for row in rows)
                            or any(int(row["prompt_tokens"]) != prompt for row in rows)):
                        raise RuntimeError(f"Incomplete or nonmatching matrix result: {name}")
                    ids = OUT / f"prompt-{prompt}.ids"
                    key = f"prompt_{prompt}_ids_sha256"
                    ids_hash = digest(ids)
                    if key in state and state[key] != ids_hash:
                        raise RuntimeError(f"Prompt token IDs changed: {prompt}")
                    state[key] = ids_hash
                state["completed_groups"].append(label)
                write_state(state)
                print(f"DONE {label}", flush=True)
            state["status"] = "completed"
            state.pop("active_group", None)
            write_state(state)
        except BaseException as error:
            state["status"] = "blocked"
            state["error"] = f"{type(error).__name__}: {error}"
            write_state(state)
            raise


if __name__ == "__main__":
    main()
