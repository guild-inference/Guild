#!/usr/bin/env python3
"""Measure real CPU expert kernels across ISA, group sizes and core counts."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",default="/home/ubuntu/models/ornith-1.5-35b/Ornith-1.5-35B-Q4_K_M.gguf")
    p.add_argument("--output",type=Path,default=Path("bench/results/hybrid-workload-optimization"))
    p.add_argument("--counters",action="store_true",help="use sudo perf and turbostat")
    args=p.parse_args()
    for isa in ["avx512","avx2"]:
        for groups,cores in [(1,1),(256,8),(256,16),(256,32)]:
            for tokens in [1,8]:
                name=f"cpu-{isa}-g{groups}-c{cores}-t{tokens}"
                metadata = args.output / f"{name}.command.json"
                csv_path = args.output / f"{name}.csv"
                if metadata.exists():
                    result = json.loads(metadata.read_text())
                    if (result.get("exit_code") == 0 and result.get("wrapper_exit_code", 0) == 0
                            and result.get("state", "completed") == "completed" and not result.get("error")
                            and not result.get("cleanup_errors") and csv_path.is_file()):
                        with csv_path.open(newline="") as f:
                            rows = list(csv.DictReader(f))
                        try:
                            complete = (len(rows) == 10 and sorted(int(row["run"]) for row in rows) == list(range(10))
                                        and all(None not in row and all(value is not None for value in row.values()) for row in rows))
                        except (KeyError, ValueError, TypeError):
                            complete = False
                        if not complete:
                            raise RuntimeError(f"incomplete CSV: {csv_path}; inspect before resuming")
                        print(f"{name}: already complete", flush=True)
                        continue
                    raise RuntimeError(f"incomplete measurement: {metadata}; inspect before resuming")
                if csv_path.exists():
                    raise RuntimeError(f"CSV without command metadata: {csv_path}; choose a new output directory")
                command=[sys.executable,"tools/run_workload_command.py","--name",name,"--output",str(args.output)]
                if args.counters:command+=["--turbostat"]
                command+=["--"]
                if args.counters:
                    command += ["sudo","-n","perf","stat","-x",",","-e","cycles,instructions,cache-misses,context-switches,cpu-migrations","-o",str(args.output/f"{name}.perf.txt"),"--"]
                command += ["numactl","--interleave=all","env",f"GUILD_FORCE_AVX2={int(isa=='avx2')}","./build-cuda12/cpu_expert_workload",args.model,"0",str(groups),str(tokens),"50" if groups>1 else "500",str(cores),str(args.output/f"{name}.csv"),"10"]
                subprocess.run(command,check=True)


if __name__=="__main__":
    main()
