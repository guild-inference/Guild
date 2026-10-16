#!/usr/bin/env python3
"""Durable, serial hardware-owner matrix; all new raw artifacts are exclusive.

Operational entry points: --audit, --launch, --status SESSION_DIRECTORY.
The existing benchmark tools and binary are invoked without modification.
"""
import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
import uuid

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
BINARY = ROOT / "build-cuda12/ornith_prefill_bench"
EXPECTED_HEAD = "87fc64fc596bb859f6b5f4eca4db992475d115ca"
OLD_BINARY = "d64290965bfe580156d090a24cdc2e95c1a643136261a8f24ef9dc7160c0c1c4"
PROMPTS = [512, 2048, 4096, 8192, 16384]
BATCHES = [64, 128, 256, 512]
METRICS = ["prefill_tps", "decode_tps", "prefill_ms", "decode_ms", "ttft_ms",
           "wall_ms", "vram_sample_bytes", "host_peak_bytes", "activation_transfer_bytes"]


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value, exclusive=False):
    if exclusive:
        with path.open("x") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    else:
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("w") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)


def head():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def gpu_check():
    command = ["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
               "--format=csv,noheader"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=10, check=True)
    # Fail closed, preserving the query; never signal an unrelated process.
    processes = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return {"utc": utc(), "command": command, "exit_code": result.returncode,
            "processes": processes, "clear": not processes}


def validate(directory, mode, prompt, batch, expected_sha=None):
    name = f"{mode}-p{prompt}-b{batch}"
    metadata_path = directory / f"{name}.command.json"
    metadata = json.loads(metadata_path.read_text())
    command = metadata["command"]
    for option, expected in [("--context", "32768"), ("--prompt", str(prompt)),
                             ("--generate", "128"), ("--batch", str(batch)), ("--runs", "10")]:
        if command.count(option) != 1 or command[command.index(option) + 1] != expected:
            raise RuntimeError(f"{name}: command shape mismatch for {option}")
    if "--full-warmup" not in command:
        raise RuntimeError(f"{name}: missing full-shaped warmup")
    if (metadata.get("exit_code") != 0 or metadata.get("wrapper_exit_code") != 0
            or metadata.get("state") != "completed" or metadata.get("cleanup_errors")):
        raise RuntimeError(f"{name}: command metadata has not recorded clean completion")
    binary_sha = metadata["build-cuda12/ornith_prefill_bench_sha256"]
    if expected_sha and binary_sha != expected_sha:
        raise RuntimeError(f"{name}: unexpected benchmark binary identity")
    env = metadata.get("environment", {})
    for key in ["GUILD_PREFILL_BATCH_PROJECTIONS", "GUILD_HOST_AFFINITY"]:
        if mode == "baseline" and env.get(key) != "0":
            raise RuntimeError(f"{name}: baseline environment mismatch")
        if mode == "final" and key in env:
            raise RuntimeError(f"{name}: final case did not use retained defaults")
    csv_path = directory / f"{name}.csv"
    with csv_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 10 or sorted(int(row["run"]) for row in rows) != list(range(10)):
        raise RuntimeError(f"{name}: expected exactly ten distinct measured runs")
    for row in rows:
        if (int(row["batch"]) != batch or int(row["prompt_tokens"]) != prompt
                or int(row["prefill_tokens"]) != prompt - 1 or int(row["generated_tokens"]) != 128
                or row["finish"] != "length"):
            raise RuntimeError(f"{name}: incomplete or mismatched measured row")
        for metric in METRICS:
            if not math.isfinite(float(row[metric])) or float(row[metric]) <= 0:
                raise RuntimeError(f"{name}: invalid numeric measurement for {metric}")
    ids = directory / f"prompt-{prompt}.ids"
    if len(ids.read_text().split()) != prompt:
        raise RuntimeError(f"{name}: unexpected prompt-ID count")
    original_ids = OUT / f"prompt-{prompt}.ids"
    if sha(ids) != sha(original_ids):
        raise RuntimeError(f"{name}: prompt-ID identity differs from saved baseline")
    git = metadata["git_head"]
    if git.get("status") != 0:
        raise RuntimeError(f"{name}: missing recorded git identity")
    return {"id": name, "mode": mode, "prompt": prompt, "batch": batch,
            "command_metadata": str(metadata_path), "command_metadata_sha256": sha(metadata_path),
            "csv": str(csv_path), "csv_sha256": sha(csv_path), "rows": len(rows),
            "exit_code": metadata["exit_code"], "wrapper_exit_code": metadata["wrapper_exit_code"],
            "full_shaped_warmup": True, "context": 32768, "generated_tokens_per_run": 128,
            "prompt_ids_sha256": sha(ids), "benchmark_sha256": binary_sha,
            "git_head": git["stdout"].strip(), "diff_sha256": metadata.get("diff_sha256"),
            "utc_start": metadata.get("utc_start"), "utc_end": metadata.get("utc_end"),
            "elapsed_seconds": metadata.get("elapsed_seconds"),
            "identity_label": "saved baseline: 2909102 plus recorded diff; old binary"
                if binary_sha == OLD_BINARY else "reviewed binary at recorded HEAD",
            "statistics": {metric: describe([float(row[metric]) for row in rows]) for metric in METRICS}}


def describe(values):
    values = sorted(values)
    def percentile(q):
        position = (len(values) - 1) * q
        lower = int(position)
        upper = min(lower + 1, len(values) - 1)
        return values[lower] + (values[upper] - values[lower]) * (position - lower)
    median = statistics.median(values)
    return {"n": len(values), "median": median, "p05": percentile(0.05), "p95": percentile(0.95),
            "mad": statistics.median(abs(value - median) for value in values),
            "sample_stddev": statistics.stdev(values), "min": values[0], "max": values[-1]}


def audit():
    records = [validate(OUT, "baseline", prompt, 256, OLD_BINARY) for prompt in PROMPTS]
    # Inventory failures without emitting archived model stdout or token strings.
    failures = []
    for path in sorted(OUT.glob("*.command.json")):
        data = json.loads(path.read_text())
        if data.get("exit_code") != 0 or data.get("wrapper_exit_code", 0) != 0:
            failures.append({"metadata": str(path), "state": data.get("state"),
                             "exit_code": data.get("exit_code"),
                             "wrapper_exit_code": data.get("wrapper_exit_code"),
                             "has_error": bool(data.get("error"))})
    return {"utc": utc(), "git_head": head(), "benchmark_sha256": sha(BINARY),
            "validated_existing_baselines": records, "prior_nonzero_or_unfinished_metadata": failures,
            "gpu_check": gpu_check()}


def planned_cases():
    return [("final", prompt, batch) for prompt in PROMPTS for batch in BATCHES] + [
        ("baseline", 512, batch) for batch in [64, 128, 512]]


def save_state(session, state):
    state["updated_utc"] = utc()
    state["completed_new_cases"] = len(state["completed_cases"])
    state["validated_total_cases"] = 5 + len(state["completed_cases"])
    state["validated_total_rows"] = 10 * state["validated_total_cases"]
    write_json(session / "state.json", state)


def write_report(session, state):
    cases = [validate(OUT, "baseline", prompt, 256, OLD_BINARY) for prompt in PROMPTS]
    for mode, prompt, batch in planned_cases():
        cases.append(validate(session / f"{mode}-p{prompt}-b{batch}", mode, prompt, batch,
                              state["benchmark_sha256"]))
    report = {"status": "completed", "utc": utc(), "expected_case_count": 28,
              "validated_case_count": len(cases), "measured_row_count": sum(case["rows"] for case in cases),
              "new_case_count": 23, "saved_baseline_case_count": 5,
              "method": "All ten runs per case; median, MAD, sample standard deviation; p05/p95 linear interpolation at (n-1)*q. No sample exclusions.",
              "identity_note": "Saved batch-256 baselines used the old binary at 2909102 plus its recorded diff. All final cases and additional prompt-512 baselines used the reviewed new binary. Compare identities per case.",
              "controller_identity": {key: state[key] for key in ["benchmark_sha256", "git_head", "controller_sha256"]},
              "cases": cases}
    write_json(session / "report.json", report, exclusive=True)
    with (session / "report.csv").open("x", newline="") as stream:
        fields = ["id", "mode", "prompt", "batch", "rows", "benchmark_sha256", "git_head", "diff_sha256"]
        fields += [f"{metric}_{stat}" for metric in METRICS for stat in ["median", "p05", "p95", "mad", "sample_stddev"]]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for case in cases:
            row = {key: case[key] for key in fields if key in case}
            row.update({f"{metric}_{stat}": case["statistics"][metric][stat]
                        for metric in METRICS for stat in ["median", "p05", "p95", "mad", "sample_stddev"]})
            writer.writerow(row)
    with (session / "report.md").open("x") as stream:
        stream.write("# Hardware matrix\n\n28/28 cases; 280 measured rows, ten per case, no exclusions.\n\n")
        stream.write(report["identity_note"] + "\n\n" + report["method"] + "\n\n")
        stream.write(f"New binary SHA-256: `{state['benchmark_sha256']}`; HEAD `{state['git_head']}`.\n\n")
        stream.write(f"Old binary SHA-256: `{OLD_BINARY}`; per-case metadata records HEAD and diff.\n\n")
        stream.write("| Case | n | Binary | Prefill tok/s median [p05, p95] | Decode tok/s median [p05, p95] | Wall ms median |\n")
        stream.write("|---|---:|---|---:|---:|---:|\n")
        for case in cases:
            pre = case["statistics"]["prefill_tps"]
            dec = case["statistics"]["decode_tps"]
            label = "old" if case["benchmark_sha256"] == OLD_BINARY else "new"
            stream.write(f"| {case['id']} | {case['rows']} | {label} | {pre['median']:.2f} [{pre['p05']:.2f}, {pre['p95']:.2f}] | {dec['median']:.2f} [{dec['p05']:.2f}, {dec['p95']:.2f}] | {case['statistics']['wall_ms']['median']:.2f} |\n")


def run(session):
    os.chdir(ROOT)
    with (OUT / "hardware-execution.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (session / "controller.lock").open("x") as session_lock:
            fcntl.flock(session_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            state = {"status": "initializing", "pid": os.getpid(), "pgid": os.getpgrp(),
                     "sid": os.getsid(0), "started_utc": utc(), "session": str(session),
                     "benchmark_sha256": sha(BINARY), "git_head": head(),
                     "controller_sha256": sha(__file__), "planned_new_cases": 23,
                     "completed_cases": [], "case_commands": [], "gpu_checks": []}
            child = None
            def interrupted(signum, _frame):
                raise InterruptedError(f"Controller received signal {signum}")
            for sig in [signal.SIGTERM, signal.SIGINT, signal.SIGHUP]:
                signal.signal(sig, interrupted)
            save_state(session, state)
            try:
                preflight = audit()
                write_json(session / "preflight.json", preflight, exclusive=True)
                if preflight["git_head"] != EXPECTED_HEAD:
                    raise RuntimeError("Unexpected initial git HEAD; inspect before measuring")
                if not preflight["gpu_check"]["clear"]:
                    raise RuntimeError("Foreign GPU compute job found in preflight")
                inherited = {key: value for key, value in os.environ.items()
                             if key.startswith(("GUILD_", "CUDA_", "OMP_", "GGML_"))}
                if inherited:
                    raise RuntimeError("Nonempty inherited workload environment; inspect before measuring")
                old_state = OUT / "hardware-execution-state.json"
                with (session / "prior-controller-state.archive.json").open("xb") as stream:
                    stream.write(old_state.read_bytes())
                write_json(session / "handoff-note.json", {
                    "utc": utc(), "prior_state": str(old_state), "prior_state_sha256": sha(old_state),
                    "prior_pid": 404419,
                    "observation": "Prior PID absent at handoff inspection. Prior state still says running. User reports coordinator terminated it after baseline-p16384-b256 completed. All five saved baselines independently validate exit_code=0, wrapper_exit_code=0 and ten complete rows.",
                    "prior_controller_exit_code": None,
                    "active_state": str(session / "state.json"),
                    "raw_artifact_policy": "Every new case has an exclusive directory; existing raw measurements and prompt-ID files are never overwritten."}, exclusive=True)
                state["status"] = "running"
                save_state(session, state)
                for mode, prompt, batch in planned_cases():
                    name = f"{mode}-p{prompt}-b{batch}"
                    directory = session / name
                    # mkdir without exist_ok intentionally prevents accidental reuse.
                    directory.mkdir()
                    command = [sys.executable, "-u", "tools/bench_hybrid_matrix.py", "--mode", mode,
                               "--prompts", str(prompt), "--batches", str(batch), "--runs", "10",
                               "--build", "build-cuda12", "--output", str(directory)]
                    command_record = {"id": name, "command": command, "cwd": str(ROOT),
                                      "command_metadata": str(directory / f"{name}.command.json"),
                                      "benchmark_sha256": sha(BINARY), "git_head": head(),
                                      "started_utc": utc(), "exit_code": None}
                    if command_record["benchmark_sha256"] != state["benchmark_sha256"]:
                        raise RuntimeError("Benchmark binary changed between measurements")
                    check = gpu_check()
                    check["before_case"] = name
                    state["gpu_checks"].append(check)
                    if not check["clear"]:
                        raise RuntimeError(f"{name}: foreign GPU compute job detected between measurements")
                    state["active_case"] = name
                    state["active_case_started_utc"] = utc()
                    state["case_commands"].append(command_record)
                    save_state(session, state)
                    print(f"{utc()} START {name}", flush=True)
                    child = subprocess.Popen(command, cwd=ROOT, env={**os.environ, "PYTHONUNBUFFERED": "1"})
                    state["matrix_tool_pid"] = child.pid
                    save_state(session, state)
                    while True:
                        try:
                            exit_code = child.wait(timeout=30)
                            break
                        except subprocess.TimeoutExpired:
                            save_state(session, state)
                            print(f"{utc()} HEARTBEAT {name}", flush=True)
                    command_record["exit_code"] = exit_code
                    command_record["ended_utc"] = utc()
                    state.pop("matrix_tool_pid", None)
                    save_state(session, state)
                    if exit_code != 0:
                        raise RuntimeError(f"{name}: matrix tool exited {exit_code}; inspect saved metadata and logs before any retry")
                    record = validate(directory, mode, prompt, batch, state["benchmark_sha256"])
                    state["completed_cases"].append(record)
                    state.pop("active_case", None)
                    state.pop("active_case_started_utc", None)
                    save_state(session, state)
                    print(f"{utc()} DONE {name}: rows=10 generated=128 full_warmup=true exit=0 ({len(state['completed_cases'])}/23 new cases)", flush=True)
                    child = None
                check = gpu_check()
                check["after_all_cases"] = True
                state["gpu_checks"].append(check)
                if not check["clear"]:
                    raise RuntimeError("Foreign GPU job found after final case; inspect before reporting")
                write_report(session, state)
                state["status"] = "completed"
                state["finished_utc"] = utc()
                state["report"] = str(session / "report.md")
                save_state(session, state)
                print(f"{utc()} COMPLETED 28/28 cases, 280 measured rows: {session / 'report.md'}", flush=True)
            except BaseException as error:
                for sig in [signal.SIGTERM, signal.SIGINT, signal.SIGHUP]:
                    signal.signal(sig, signal.SIG_IGN)
                # The workload wrapper starts its own private benchmark group.
                # Signal that wrapper so its existing cleanup reaps the owned child.
                if child is not None and child.poll() is None:
                    metadata_path = Path(state["case_commands"][-1]["command_metadata"])
                    if metadata_path.exists():
                        data = json.loads(metadata_path.read_text())
                        benchmark_pid = data.get("child_pid")
                        if benchmark_pid:
                            try:
                                wrapper_pid = int(subprocess.check_output(
                                    ["ps", "-o", "ppid=", "-p", str(benchmark_pid)], text=True).strip())
                                # Require wrapper to belong to this controller's private process group.
                                if os.getpgid(wrapper_pid) == os.getpgrp():
                                    os.kill(wrapper_pid, signal.SIGTERM)
                            except (ProcessLookupError, ValueError):
                                pass
                    try:
                        state["case_commands"][-1]["exit_code"] = child.wait(timeout=45)
                    except subprocess.TimeoutExpired:
                        state["cleanup_pending"] = True
                state["status"] = "blocked"
                state["error"] = f"{type(error).__name__}: {error}"
                save_state(session, state)
                print(f"{utc()} BLOCKED {state['error']}", flush=True)
                raise


def launch():
    session = OUT / ("matrix-owner-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:8])
    session.mkdir()
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--run", str(session)]
    with (session / "progress.log").open("x") as log:
        child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
    record = {"pid": child.pid, "command": command, "session": str(session),
              "state": str(session / "state.json"), "log": str(session / "progress.log"), "utc": utc()}
    write_json(session / "launch.json", record, exclusive=True)
    print(json.dumps(record, indent=2))


def status(session):
    state = json.loads((session / "state.json").read_text())
    keys = ["status", "pid", "pgid", "sid", "started_utc", "updated_utc", "active_case",
            "active_case_started_utc", "matrix_tool_pid", "planned_new_cases", "completed_new_cases",
            "validated_total_cases", "validated_total_rows", "benchmark_sha256", "git_head", "error", "report"]
    result = {key: state[key] for key in keys if key in state}
    result["completed_case_ids"] = [case["id"] for case in state["completed_cases"]]
    try:
        os.kill(state["pid"], 0)
        result["controller_process_exists"] = True
    except ProcessLookupError:
        result["controller_process_exists"] = False
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--audit", action="store_true")
    actions.add_argument("--launch", action="store_true")
    actions.add_argument("--run", type=Path)
    actions.add_argument("--status", type=Path)
    args = parser.parse_args()
    if args.audit:
        result = audit()
        print(json.dumps({**{key: result[key] for key in ["utc", "git_head", "benchmark_sha256", "gpu_check", "prior_nonzero_or_unfinished_metadata"]},
                          "validated_existing_baselines": [{key: record[key] for key in ["id", "rows", "exit_code", "wrapper_exit_code", "benchmark_sha256", "git_head", "diff_sha256"]}
                                                           for record in result["validated_existing_baselines"]]}, indent=2))
    elif args.launch:
        launch()
    elif args.run:
        run(args.run.resolve())
    else:
        status(args.status.resolve())


if __name__ == "__main__":
    main()
