#!/usr/bin/env python3
"""Development-only real-model gates. No asset downloads or runtime dependency."""
import argparse
import json
import pathlib
import subprocess
import time

p = argparse.ArgumentParser()
p.add_argument("--build", default="build-cuda12")
p.add_argument("--long", action="store_true")
p.add_argument("--cases", type=int, nargs="*")
p.add_argument("--model", default="/home/ubuntu/models/ornith-1.5-35b/Ornith-1.5-35B-Q4_K_M.gguf")
p.add_argument("--pack", default="/mnt/models-ssd/Strata-data/packs/ornith-1.5-35b")
p.add_argument("--output", default="bench/results/ornith-prefill")
args = p.parse_args()
out = pathlib.Path(args.output)
out.mkdir(parents=True, exist_ok=True)
json_path = out / ("long-parity.json" if args.long else "short-parity.json")
existing = {}
if json_path.exists():
    for item in json.loads(json_path.read_text()):
        existing[item["prompt_tokens"]] = item
cases = [1, 16, 127, 128, 129, 512]
if args.long:
    cases = [2050, 2051, 2052, 4096, 8192, 16384]
if args.cases:
    cases = args.cases
for n in cases:
    cmd = [str(pathlib.Path(args.build).resolve() / "ornith_prefill_test"), args.model, args.pack, str(n)]
    start = time.monotonic()
    log = out / f"parity-{n}.log"
    with log.open("w") as f:
        status = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=1800).returncode
    existing[n] = dict(command=cmd, prompt_tokens=n, exit_code=status, elapsed_seconds=time.monotonic()-start, log=str(log))
    records = [existing[k] for k in sorted(existing)]
    json_path.write_text(json.dumps(records, indent=2)+"\n")
    print(f"prompt={n}: {'PASS' if status == 0 else 'FAIL'}", flush=True)
    if status:
        raise SystemExit(status)
