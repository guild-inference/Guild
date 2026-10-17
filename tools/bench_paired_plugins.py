#!/usr/bin/env python3
"""Serial, counterbalanced Phase 1/Core/plugin comparisons with paired uncertainty.

No compilation or clock/affinity changes are performed by this driver. Each
sample is a fresh process, one full-shaped warmup and one measured repetition.
Ten rounds produce ten measurements per case. Reverse order on odd rounds and
rotate each forward/reverse pair to reduce fixed ordering and thermal bias.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import tempfile
from datetime import datetime, timezone

PHASE1_SHA256 = "71dca3597ba64f135517e0d32f5dd585171d4913b18a1919a537f926c2decc7c"
ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def capture(command):
    result = subprocess.run(command, text=True, capture_output=True)
    return {"exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def gpu_state():
    return capture(["nvidia-smi", "--query-gpu=name,driver_version,temperature.gpu,clocks.gr,clocks.mem,pstate,power.draw",
                    "--format=csv,noheader"])


def cpu_state():
    result = {"load_average": os.getloadavg()}
    for field in ("scaling_governor", "scaling_cur_freq", "energy_performance_preference"):
        result[field] = {p.parent.parent.name: p.read_text().strip()
                         for p in Path("/sys/devices/system/cpu").glob(f"cpu[0-9]*/cpufreq/{field}")}
    return result


def intervals(values):
    """Paired bootstrap CI of the median ratio; never pool unmatched rounds."""
    if not values:
        raise ValueError("empty comparison")
    rng = random.Random(20261017)
    samples = sorted(statistics.median(rng.choices(values, k=len(values))) for _ in range(10000))
    return {"median_percent": 100 * (statistics.median(values) - 1),
            "bootstrap_95_percent": [100 * (samples[250] - 1), 100 * (samples[9749] - 1)],
            "pairs": len(values)}


def summarize(output, cases, prompts, rounds):
    report = {}
    for prompt in prompts:
        observations = {}
        for case in cases:
            observations[case] = {}
            for r in range(rounds):
                path = output / f"p{prompt}-r{r:02}-{case}.json"
                if path.exists():
                    item = json.loads(path.read_text())
                    if item["exit_code"] == 0:
                        observations[case][r] = item["measurement"]
        section = {}
        for case in cases:
            rows = observations[case]
            section[case] = {"samples": len(rows)}
            for field in ("prefill_tps", "decode_tps", "cpu_gu_ms", "cpu_quant_ms", "cpu_down_ms", "host_wait_ms"):
                if rows:
                    section[case][field] = statistics.median(float(v[field]) for v in rows.values())
            for baseline in ("preserved", "builtin"):
                if baseline == case or baseline not in observations:
                    continue
                paired = sorted(rows.keys() & observations[baseline].keys())
                for field in ("prefill_tps", "decode_tps"):
                    if paired:
                        section[case][f"{field}_vs_{baseline}"] = intervals([
                            float(rows[r][field]) / float(observations[baseline][r][field]) for r in paired])
        report[str(prompt)] = section
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preserved", type=Path, default=ROOT / "bench/results/external-kernels/phase1/ornith_prefill_bench")
    p.add_argument("--build", type=Path, default=ROOT / "build-cuda12")
    p.add_argument("--cpu", type=Path, default=Path("/home/ubuntu/guild-kernels-cpu/build/package"))
    p.add_argument("--cuda", type=Path, default=Path("/home/ubuntu/guild-kernels-cuda/build/package"))
    p.add_argument("--runner", type=Path, default=Path("/home/ubuntu/guild-runners-qwen/build/package"))
    p.add_argument("--cases", nargs="+", choices=["preserved", "builtin", "cpu", "cuda", "runner", "all"],
                   default=["preserved", "builtin", "cpu", "cuda", "runner", "all"])
    p.add_argument("--prompts", type=int, nargs="+", default=[512, 4096, 8192, 16384])
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.rounds < 1 or len(set(args.cases)) != len(args.cases):
        p.error("positive rounds and unique cases required")
    build, output = args.build.resolve(), args.output.resolve()
    if digest(args.preserved) != PHASE1_SHA256:
        raise RuntimeError("preserved Phase 1 binary digest mismatch")
    output.mkdir(parents=True, exist_ok=True)
    identity = {"preserved_sha256": PHASE1_SHA256, "current_sha256": digest(build / "ornith_prefill_bench"),
                "guild_sha256": digest(build / "guild"), "cases": args.cases, "prompts": args.prompts,
                "rounds": args.rounds, "context": 32768, "batch": 256, "generate": 128,
                "packages": {name: {"manifest_sha256": digest(path / ".gkmetadata"),
                                     "library_sha256": digest(path / json.loads((path / ".gkmetadata").read_text())["library_path"])}
                             for name, path in (("cpu", args.cpu), ("cuda", args.cuda), ("runner", args.runner))}}
    relevant_environment = {name: os.environ.get(name) for name in
                            ("CUDA_MODULE_LOADING", "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "OMP_PROC_BIND", "OMP_PLACES")}
    identity["environment"] = relevant_environment
    identity_path = output / "identity.json"
    if identity_path.exists():
        previous = json.loads(identity_path.read_text())
        if "environment" not in previous and all(v is None for v in relevant_environment.values()):
            previous["environment"] = relevant_environment
        if previous != identity:
            raise RuntimeError("refusing to mix different benchmark inputs in an existing output directory")
    identity_path.write_text(json.dumps(identity, indent=2) + "\n")
    (output / "hardware.json").write_text(json.dumps({"cpu": capture(["lscpu"]), "gpu": gpu_state(),
                                                      "affinity": sorted(os.sched_getaffinity(0))}, indent=2) + "\n")
    with tempfile.TemporaryDirectory(prefix="paired-packages-", dir=build) as temporary:
        work = Path(temporary)
        cli = str(build / "guild")
        key = json.loads(subprocess.check_output([cli, "kernel", "keygen", "--json"], text=True))
        seed = work / "test.seed"
        with os.fdopen(os.open(seed, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600), "w") as f:
            f.write(key["private_seed_hex"] + "\n")
        bundles = {}
        for name, source in (("cpu", args.cpu), ("cuda", args.cuda), ("runner", args.runner)):
            stage = work / f"stage-{name}"
            shutil.copytree(source, stage)
            bundles[name] = work / f"{name}.gkernel"
            subprocess.run([cli, "kernel", "sign", str(stage), "--key-id", "paired-test",
                            "--seed-file", str(seed), "--bundle", str(bundles[name])], check=True, stdout=subprocess.DEVNULL)
        seed.unlink()
        environments = {}
        for case in args.cases:
            env = os.environ.copy()
            env["GUILD_DATA_DIR"] = str(work / f"data-{case}")
            # Refuse to inherit arithmetic/performance overrides accidentally.
            unexpected = sorted(k for k in env if k.startswith("GUILD_") and k != "GUILD_DATA_DIR")
            if unexpected:
                raise RuntimeError(f"explicitly clear inherited Guild overrides before measuring: {unexpected}")
            names = ["cpu", "cuda", "runner"] if case == "all" else [case] if case in bundles else []
            if names:
                subprocess.run([cli, "kernel", "trust-key", "--publisher", "org.guild-inference",
                                "--key-id", "paired-test", "--pubkey", key["public_key_hex"]], env=env,
                               check=True, stdout=subprocess.DEVNULL)
                for name in names:
                    subprocess.run([cli, "kernel", "install", str(bundles[name])], env=env,
                                   check=True, stdout=subprocess.DEVNULL)
            environments[case] = env
        for prompt in args.prompts:
            for r in range(args.rounds):
                shift = (r // 2) % len(args.cases)
                order = args.cases[shift:] + args.cases[:shift]
                if r % 2:
                    order = list(reversed(order))
                for case in order:
                    stem = output / f"p{prompt}-r{r:02}-{case}"
                    receipt = stem.with_suffix(".json")
                    if receipt.exists() and json.loads(receipt.read_text())["exit_code"] == 0:
                        continue
                    binary = args.preserved.resolve() if case == "preserved" else build / "ornith_prefill_bench"
                    command = [str(binary), "--context", "32768", "--prompt", str(prompt), "--generate", "128",
                               "--batch", "256", "--runs", "1", "--full-warmup", "--csv", str(stem.with_suffix(".csv"))]
                    before = {"cpu": cpu_state(), "gpu": gpu_state()}
                    start = datetime.now(timezone.utc).isoformat()
                    with stem.with_suffix(".log").open("w") as log:
                        result = subprocess.run(command, cwd=ROOT, env=environments[case], stdout=log, stderr=log)
                    item = {"command": command, "case": case, "round": r, "start_utc": start,
                            "exit_code": result.returncode, "before": before, "after": {"cpu": cpu_state(), "gpu": gpu_state()}}
                    if result.returncode == 0:
                        with stem.with_suffix(".csv").open() as f:
                            rows = list(csv.DictReader(f))
                        if len(rows) != 1 or int(rows[0]["generated_tokens"]) != 128 or int(rows[0]["file_reads"]) != 0:
                            raise RuntimeError(f"invalid measurement: {stem}")
                        item["measurement"] = rows[0]
                    receipt.write_text(json.dumps(item, indent=2) + "\n")
                    if result.returncode:
                        raise RuntimeError(f"benchmark failed; inspect {stem.with_suffix('.log')}")
                    summarize(output, args.cases, args.prompts, args.rounds)
                    print(f"p{prompt} round {r+1}/{args.rounds} {case}: "
                          f"pp={rows[0]['prefill_tps']} tg={rows[0]['decode_tps']}", flush=True)
    if digest(build / "ornith_prefill_bench") != identity["current_sha256"]:
        raise RuntimeError("benchmark binary changed during measurement")


if __name__ == "__main__":
    main()
