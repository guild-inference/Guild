#!/usr/bin/env python3
"""Counterbalanced frozen-Core/current/module-resident inference comparisons.

Each sample is a fresh process with full-shaped warmup. Also compares first-token
logits and generated token IDs bit-for-bit. No compilation or tuning is performed.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from bench_paired_plugins import capture, cpu_state, digest, gpu_state, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--preserved", type=Path, required=True)
    parser.add_argument("--baseline-sha256", required=True)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompts", type=int, nargs="+", default=[512, 4096])
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    if args.rounds < 1 or any(p < 1 or p + 128 > 32768 for p in args.prompts):
        parser.error("positive rounds and valid prompt lengths required")
    build, output = args.build.resolve(), args.output.resolve()
    time_binary = shutil.which("time")
    if not time_binary:
        raise RuntimeError("GNU time is required for per-process peak RSS measurements")
    if digest(args.preserved) != args.baseline_sha256:
        raise RuntimeError("frozen baseline digest mismatch")
    output.mkdir(parents=True, exist_ok=False)
    cases = ["preserved", "builtin", "module"]
    identity = {"preserved_sha256": args.baseline_sha256, "current_sha256": digest(build / "ornith_prefill_bench"),
                "guild_sha256": digest(build / "guild"), "cases": cases, "prompts": args.prompts,
                "rounds": args.rounds, "context": 32768, "batch": 256, "generate": 128,
                "environment": {k: v for k, v in os.environ.items() if k.startswith(("GUILD_", "CUDA_", "OMP_"))}}
    (output / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    (output / "hardware.json").write_text(json.dumps({"cpu": capture(["lscpu"]), "gpu": gpu_state(),
                                                       "affinity": sorted(os.sched_getaffinity(0))}, indent=2) + "\n")
    with tempfile.TemporaryDirectory(prefix="phase3-bench-", dir=build) as temporary:
        work = Path(temporary)
        cli = str(build / "guild")
        envs = {case: dict(os.environ, GUILD_DATA_DIR=str(work / case)) for case in cases}
        unexpected = [k for k in os.environ if k.startswith("GUILD_")]
        if unexpected:
            raise RuntimeError(f"clear inherited Guild overrides before measuring: {unexpected}")
        stage = work / "module-source"
        shutil.copytree(args.module, stage)
        metadata = json.loads((stage / ".gkmetadata").read_text())
        key = json.loads(subprocess.check_output([cli, "kernel", "keygen", "--json"], env=envs["module"], text=True))
        seed = work / "seed"
        with os.fdopen(os.open(seed, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600), "w") as stream:
            stream.write(key["private_seed_hex"] + "\n")
        subprocess.run([cli, "kernel", "trust-key", "--publisher", metadata["publisher_id"], "--key-id", "benchmark",
                        "--pubkey", key["public_key_hex"]], env=envs["module"], check=True)
        bundle = work / "scan.gkmodule"
        subprocess.run([cli, "kernel", "sign", str(stage), "--key-id", "benchmark", "--seed-file", str(seed),
                        "--bundle", str(bundle)], env=envs["module"], check=True)
        subprocess.run([cli, "kernel", "install", str(bundle)], env=envs["module"], check=True)
        report = json.loads(subprocess.check_output([cli, "--version", "--json"], env=envs["module"], text=True))
        assert any(p["package_id"] == metadata["package_id"] and p["loaded_in_process"] for p in report["external_packages"])
        (output / "module.json").write_text(json.dumps({"metadata": json.loads((stage / ".gkmetadata").read_text()),
                                                        "report": report}, indent=2) + "\n")
        for prompt in args.prompts:
            reference = None
            for round_number in range(args.rounds):
                order = cases if round_number % 2 == 0 else list(reversed(cases))
                for case in order:
                    stem = output / f"p{prompt}-r{round_number:02}-{case}"
                    binary = args.preserved.resolve() if case == "preserved" else build / "ornith_prefill_bench"
                    dump = work / "dump"
                    command = [str(binary), "--prompt", str(prompt), "--generate", "128", "--batch", "256",
                               "--context", "32768", "--runs", "1", "--full-warmup", "--csv", str(stem.with_suffix(".csv")),
                               "--dump", str(dump)]
                    before = {"cpu": cpu_state(), "gpu": gpu_state()}
                    rss = work / "rss-kib.txt"
                    with stem.with_suffix(".log").open("w") as log:
                        subprocess.run([time_binary, "-f", "%M", "-o", str(rss), *command],
                                       env=envs[case], stdout=log, stderr=log, check=True)
                    with stem.with_suffix(".csv").open() as stream:
                        rows = list(csv.DictReader(stream))
                    assert len(rows) == 1 and int(rows[0]["generated_tokens"]) == 128 and int(rows[0]["file_reads"]) == 0
                    parity = {suffix: digest(dump.with_suffix(suffix)) for suffix in (".f32", ".ids")}
                    if reference is None:
                        reference = parity
                    assert reference == parity, f"inference parity failed for {stem}"
                    receipt = {"case": case, "round": round_number, "command": command, "exit_code": 0,
                               "peak_rss_kib": int(rss.read_text().strip()),
                               "measurement": rows[0], "parity_sha256": parity, "before": before,
                               "after": {"cpu": cpu_state(), "gpu": gpu_state()}}
                    stem.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n")
                    summarize(output, cases, args.prompts, args.rounds)
                    print(f"p{prompt} {case} round {round_number}: pp={rows[0]['prefill_tps']} tg={rows[0]['decode_tps']}", flush=True)
    assert digest(build / "ornith_prefill_bench") == identity["current_sha256"]


if __name__ == "__main__":
    main()
