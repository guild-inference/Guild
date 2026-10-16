#!/usr/bin/env python3
"""Run one isolated measurement, preserving command, build identity and telemetry.

Use --name NAME -- COMMAND ... . Stdout/stderr are archived, not interpreted.
Never run two inference measurements concurrently. No downloads or dependencies.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
import signal


def sha256(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def credential_name(name):
    return bool(re.search(r"(?:^|_)(?:KEY|APIKEY|TOKEN|SECRET|PASSWORD|PASSWD|AUTH|AUTHORIZATION|CREDENTIALS?|COOKIE)(?:_|$)",
                          name.upper().replace("-", "_")))


def public_environment(env):
    """Only workload settings, never credentials from the launcher's environment."""
    return {k: v for k, v in sorted(env.items())
            if k.startswith(("GUILD_", "CUDA_", "OMP_", "GGML_"))
            and not credential_name(k)}


def public_command(command):
    """Preserve argv structure but redact explicit credential assignments/options."""
    result, redact_next = [], False
    for word in command:
        name, equal, _value = word.partition("=")
        if redact_next:
            result.append("<redacted>")
            redact_next = False
        elif equal and credential_name(name):
            result.append(name + "=<redacted>")
        else:
            result.append(word)
            redact_next = word.startswith("--") and credential_name(word)
    return result


def capture(cmd, timeout=30):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return {"command": cmd, "status": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"command": cmd, "error": str(e)}


def signal_process_group(pid, sig):
    if pid <= 1 or pid == os.getpgrp():
        raise ValueError("refusing to signal the launcher's process group")
    try:
        os.killpg(pid, sig)
    except PermissionError as e:
        # sudo can exec a root command directly, leaving no user-owned signal
        # forwarder. Elevate only for this exact group we created, never pkill.
        result = capture(["sudo", "-n", "kill", "-s", str(int(sig)), "--", f"-{pid}"], timeout=5)
        if result.get("status") != 0:
            # Distinguish a group that exited during cleanup from denied sudo.
            os.killpg(pid, 0)
            raise PermissionError(f"cannot signal owned process group {pid}: {result.get('stderr', result.get('error', 'sudo kill failed'))}") from e


def stop_process_group(process, first_signal=signal.SIGTERM, timeout=10):
    """Signal only a session created by this wrapper, including its descendants.

    Also signal after the leader exits: perf/launchers can leave children behind.
    sudo forwards signals it receives to its privileged command.
    """
    for sig in (first_signal, signal.SIGTERM, signal.SIGKILL):
        try:
            signal_process_group(process.pid, sig)
        except ProcessLookupError:
            break
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            continue
        # The leader is reaped; do not leave an uncooperative grandchild alive.
        if sig != signal.SIGKILL:
            try:
                signal_process_group(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        break
    process.wait(timeout=timeout)


def execute(cmd, env, prefix, metadata, turbostat=False):
    """Run and finalize metadata even if a launch, telemetry read or signal fails."""
    def artifact(suffix):
        return Path(str(prefix) + suffix)

    start = time.monotonic()
    metadata.update(monotonic_start_ns=time.monotonic_ns(), exit_code=None, state="running")
    path = artifact(".command.json")
    path.write_text(json.dumps(metadata, indent=2) + "\n")
    monitor = child = monitor_log = None
    exit_code = 1
    previous_handlers = {}

    def interrupted(signum, _frame):
        metadata["interrupted_signal"] = signum
        raise KeyboardInterrupt

    try:
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            previous_handlers[sig] = signal.signal(sig, interrupted)
        if turbostat:
            monitor_log = artifact(".turbostat-log.txt").open("x")
            monitor = subprocess.Popen(
                ["sudo", "-n", "turbostat", "--quiet", "--Summary", "--show", "Busy%,Bzy_MHz,TSC_MHz",
                 "--interval", "1", "--out", str(artifact(".frequencies.txt"))],
                stdout=monitor_log, stderr=monitor_log, start_new_session=True)
            metadata["turbostat_pid"] = monitor.pid
        with artifact(".stdout.txt").open("x") as stdout, artifact(".stderr.txt").open("x") as stderr, \
                artifact(".telemetry.jsonl").open("x") as telemetry:
            child = subprocess.Popen(cmd, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
            metadata["child_pid"] = child.pid
            next_sample = time.monotonic()
            while child.poll() is None:
                record = {"seconds": time.monotonic() - start, "monotonic_ns": time.monotonic_ns(), "pid": child.pid}
                for name in ("status", "numa_maps"):
                    try:
                        record[name] = Path(f"/proc/{child.pid}/{name}").read_text()
                    except OSError as e:
                        # A root perf/sudo process may hide procfs; its benchmark is still valid.
                        record[name + "_error"] = type(e).__name__
                record["gpu"] = capture(["nvidia-smi", "--query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw,memory.used,pstate",
                                         "--format=csv,noheader,nounits"], timeout=2)
                telemetry.write(json.dumps(record) + "\n")
                next_sample = max(next_sample + 1, time.monotonic())
                try:
                    child.wait(timeout=max(0, next_sample - time.monotonic()))
                except subprocess.TimeoutExpired:
                    pass
            metadata["exit_code"] = child.returncode
            exit_code = child.returncode if child.returncode >= 0 else 128 - child.returncode
    except KeyboardInterrupt:
        exit_code = 128 + metadata.get("interrupted_signal", signal.SIGINT)
        metadata["error"] = "measurement interrupted"
    except Exception as e:
        metadata["error"] = f"{type(e).__name__}: {e}"
        if child is None and isinstance(e, FileNotFoundError):
            exit_code = 127
        elif child is None and isinstance(e, PermissionError):
            exit_code = 126
    finally:
        # A second interrupt must not bypass monitor/child cleanup.
        for sig in previous_handlers:
            signal.signal(sig, signal.SIG_IGN)
        metadata["elapsed_seconds"] = time.monotonic() - start
        for label, process, sig in (("child", child, signal.SIGTERM), ("turbostat", monitor, signal.SIGINT)):
            if process is not None:
                try:
                    stop_process_group(process, sig)
                except (OSError, subprocess.TimeoutExpired) as e:
                    metadata.setdefault("cleanup_errors", []).append(f"{label}: {type(e).__name__}: {e}")
                if label == "child":
                    metadata["exit_code"] = process.returncode
                else:
                    metadata["turbostat_exit_code"] = process.returncode
        if monitor_log is not None:
            try:
                monitor_log.close()
            except OSError as e:
                metadata.setdefault("cleanup_errors", []).append(f"turbostat log: {type(e).__name__}: {e}")
        if metadata.get("cleanup_errors"):
            exit_code = exit_code or 1
        metadata["wrapper_exit_code"] = exit_code
        metadata["state"] = "completed" if exit_code == 0 and not metadata.get("cleanup_errors") else "failed"
        metadata["utc_end"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        try:
            metadata["gpu_after"] = capture(["nvidia-smi", "-q"])
            path.write_text(json.dumps(metadata, indent=2) + "\n")
        finally:
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)
    return exit_code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output", type=Path, default=Path("bench/results/hybrid-workload-optimization"))
    parser.add_argument("--env", action="append", default=[])
    parser.add_argument("--turbostat", action="store_true", help="optional sudo hardware frequency sampler")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cmd = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not cmd or args.name in (".", "..") or Path(args.name).name != args.name:
        parser.error("a command and a simple measurement name are required")
    args.output.mkdir(parents=True, exist_ok=True)
    prefix = args.output / args.name
    env = os.environ.copy()
    for pair in args.env:
        if "=" not in pair or not pair.split("=", 1)[0]:
            parser.error("--env requires NAME=VALUE")
        k, v = pair.split("=", 1)
        env[k] = v
    metadata = {
        "command": public_command(cmd), "cwd": str(Path.cwd()), "utc_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": public_environment(env),
        "launcher_affinity": sorted(os.sched_getaffinity(0)),
        "git_head": capture(["git", "rev-parse", "HEAD"]),
        "git_status": capture(["git", "status", "--short"]),
        "diff_sha256": hashlib.sha256(subprocess.check_output(["git", "diff", "HEAD"])).hexdigest(),
    }
    executables = {"build-cuda12/ornith_prefill_bench", "build-cuda12/libguild_activity.so"}
    executables.update(word for word in cmd if Path(word).name in
                       {"ornith_prefill_bench", "cpu_expert_workload", "llama_prefill_reference"})
    for executable in sorted(executables):
        if Path(executable).is_file():
            metadata[executable + "_sha256"] = sha256(executable)
    metadata["gpu_before"] = capture(["nvidia-smi", "-q"])
    path = Path(str(prefix) + ".command.json")
    try:
        # Reserve the name atomically; never replace another wrapper's measurement.
        with path.open("x") as f:
            f.write(json.dumps(metadata, indent=2) + "\n")
    except FileExistsError:
        parser.error("measurement already exists; choose a new name")
    exit_code = execute(cmd, env, prefix, metadata, args.turbostat)
    print(f"{args.name}: exit={metadata['exit_code']}, {metadata['elapsed_seconds']:.1f}s", flush=True)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
