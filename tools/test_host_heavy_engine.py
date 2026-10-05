"""Opt-in real-model integration: zero GPU experts, RAM budget, resident/host-only greedy parity.

python tools/test_host_heavy_engine.py --config strata-MODEL.json --engine build/strata --output /tmp/strata-test
Requires an installed model, a GPU and enough RAM for all its experts. Does not edit the config or download files.
"""
import argparse
import json
import os
import queue
import re
import subprocess
import threading
from pathlib import Path

from batch_test import tokenizer


def run(engine, original, output, name, context, host_only, optional_mtp=False, pcie_fraction=0,
        expert_cache="off", resident_budget_gib=64, optional_reserve_mib=1500):
    args = list(original)
    values = ("--expert-cache", "--max-context", "--kv", "--kv-resident", "--resident-budget-gib", "--prefill",
              "--spec", "--short-read", "--pcie-frac", "--vram-reserve-mib")
    if not optional_mtp:
        values += ("--mtp",)
    for flag in values:
        while flag in args:
            at = args.index(flag)
            del args[at:at + 2]
    for flag in ("--kv-host-only", "--mtp-optional", "--resident-experts", "--serve", "--stats"):
        while flag in args:
            args.remove(flag)
    args += ["--serve", "--expert-cache", expert_cache, "--max-context", str(context), "--kv", "fp16",
             "--resident-budget-gib", str(resident_budget_gib), "--prefill", "512", "--spec", "4" if optional_mtp else "2",
             "--short-read", "0", "--pcie-frac", str(pcie_fraction), "--stats"]
    if host_only:
        args += ["--kv-host-only"]
    if optional_mtp:
        args += ["--mtp-optional"]
        if optional_reserve_mib is not None:
            # Force the optional-MTP branch on the 8 GB test card while leaving its prompt fit.
            args += ["--vram-reserve-mib", str(optional_reserve_mib)]
    log_path = output / (name + ".stderr.log")
    protocol = output / (name + ".stdout.log")
    tok = tokenizer(Path(args[args.index("--pack") + 1]) / "tokenizer")
    text = ("This test checks that the same prompt produces the same tokens when KV lives in host RAM. " * 6 +
            "Count from one to ten, spelling each number in English.")
    prompt = tok.encode(f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                        parse_special=True)
    env = dict(os.environ, STRATA_STATE_HASH="1")
    with log_path.open("w") as log, protocol.open("w") as stdout:
        process = subprocess.Popen([str(engine), *args], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=log, text=True, bufsize=1, env=env)
        lines = queue.Queue()
        def collect():
            for line in process.stdout:
                stdout.write(line); stdout.flush(); lines.put(line.strip())
            lines.put(None)
        thread = threading.Thread(target=collect, daemon=True)
        thread.start()
        def until(prefix, timeout=600):
            observed = []
            while True:
                line = lines.get(timeout=timeout)
                if line is None:
                    raise AssertionError(f"{name}: engine exited {process.poll()}; see {log_path}")
                if line.startswith("ERR "):
                    raise AssertionError(f"{name}: {line}; see {log_path}")
                observed.append(line)
                if line.startswith(prefix):
                    return observed
        try:
            info = until("READY ")
            print(f"{name}: READY", flush=True)
            outputs = []
            for request in range(2):
                # Different final token forces rewind/re-read on the second request, covering snapshots/maps too.
                ids = prompt if request == 0 else prompt + tok.encode("One")
                process.stdin.write("GEN 4 temperature=0 " + ",".join(map(str, ids)) + "\n")
                process.stdin.flush()
                result = until("DONE ")
                done = result[-1].split()
                if resident_budget_gib >= 64:
                    assert int(done[12]) == 0, f"{name}: steady decode read experts from disk: {done}"
                assert int(done[11]) > 0 or int(done[15]) > 0, f"{name}: expected RAM/PCIe expert execution"
                if pcie_fraction > 0:
                    assert int(done[15]) > 0, f"{name}: requested PCIe execution did not run"
                outputs.append([int(line.split()[1]) for line in result if line.startswith("T ")])
                assert len(outputs[-1]) == 4, f"{name}: expected four real greedy decode steps"
                print(f"{name}: request {request}, tokens {outputs[-1]}, RAM blobs {done[11]}, file blobs {done[12]}", flush=True)
            process.stdin.write("QUIT\n"); process.stdin.flush()
            process.wait(timeout=60)
            assert process.returncode == 0
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait(timeout=10)
            thread.join(timeout=5)
    log = log_path.read_text()
    assert "expert tiers: GPU " in log
    if expert_cache == "off":
        assert "expert tiers: GPU 0 slots" in log
    assert "RAM budget (--resident-budget-gib) cannot be kept" not in log
    if host_only:
        assert "GPU staging shared" in log
    if optional_mtp and optional_reserve_mib is not None:
        assert "automatic MTP disabled:" in log, "optional acceleration was not disabled in the constrained case"
    fingerprints = []
    for match in re.finditer(r"STATE_HASH L=([^\n]+)", log):
        fields = dict(item.split("=", 1) for item in match.group(1).split() if "=" in item)
        fingerprints.append({key: fields[key] for key in ("gdn", "ple", "tail", "pooled", "kv", "dead", "pooled_full")})
    return outputs, fingerprints


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--engine", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    options = parser.parse_args()
    options.output.mkdir(parents=True, exist_ok=True)
    original = json.loads(options.config.read_text())["args"]
    baseline = run(options.engine.resolve(), original, options.output, "resident-8k", 8192, False)
    host = run(options.engine.resolve(), original, options.output, "host-only-8k", 8192, True)
    assert baseline == host, (baseline, host)
    long_host = run(options.engine.resolve(), original, options.output, "host-only-262k", 262144, True)
    assert baseline == long_host, (baseline, long_host)
    optional = run(options.engine.resolve(), original, options.output, "optional-mtp-262k", 262144, True, optional_mtp=True)
    assert baseline == optional, (baseline, optional)
    run(options.engine.resolve(), original, options.output, "pcie-no-cache-262k", 262144, True, pcie_fraction=1)
    # Setup's automatic cache with the exact requested budget: reserve own prefill buffers even if a small
    # nonzero cache cannot lend them. MTP may stay on or turn off; in either case prompts must be runnable.
    run(options.engine.resolve(), original, options.output, "auto-55gib-default-mtp", 262144, True,
        optional_mtp=True, expert_cache="auto", resident_budget_gib=55, optional_reserve_mib=None)
    print("PASS: resident/host-only greedy parity, 262K startup/prefill/rewind, zero GPU slots, no steady disk reads")


if __name__ == "__main__":
    main()
