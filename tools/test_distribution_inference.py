#!/usr/bin/env python3
"""Run the shipped executable against an operator-maintained real-model golden fixture.

Fixture: model, model_manifest (local manifest JSON), request (OpenAI chat
request), expected_text (exact greedy completion), optionally prefill_batch.
This never starts a mock engine or downloads model data.
"""
import argparse
import json
import os
from pathlib import Path
import socket
import shutil
import subprocess
import tempfile
import time
from urllib.error import URLError
from urllib.request import Request, urlopen
from package_linux_cuda import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--guild", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fixture = json.loads(args.fixture.read_text())
    manifest_path = Path(fixture["model_manifest"])
    manifest = json.loads(manifest_path.read_text())
    if fixture["model"] not in [manifest["name"], *manifest.get("aliases", [])]:
        raise ValueError("fixture model does not name its supplied manifest")
    if not manifest.get("metadata", {}).get("pack_dir"):
        raise ValueError("release model manifest must specify its prepared pack directory")
    request = dict(fixture["request"], temperature=0, stream=False)
    if not fixture["expected_text"] or request.get("max_tokens", 0) < 8:
        raise ValueError("a nonempty golden completion and at least eight requested tokens are required")
    executable = args.guild.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    with tempfile.TemporaryDirectory(prefix="inference-data-", dir=args.output) as data:
        env = dict(os.environ, GUILD_DATA_DIR=data)
        manifests = Path(data) / "models/manifests"
        manifests.mkdir(parents=True)
        shutil.copyfile(manifest_path, manifests / "release-fixture.json")
        metadata = json.loads(subprocess.check_output([str(executable), "--version", "--json"], env=env, text=True))
        if metadata["backend"] != "cuda" or metadata["external_packages"]:
            raise ValueError("release smoke requires the CUDA distribution with bundled baselines")
        command = [str(executable), "serve", "--model", fixture["model"], "--host", "127.0.0.1",
                   "--port", str(port), "--data-dir", data, "--quiet",
                   "--prefill-batch", str(fixture.get("prefill_batch", 256))]
        with (args.output / "server.log").open("w") as log:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + 600
                while True:
                    if process.poll() is not None:
                        raise RuntimeError("real inference server exited; see server.log")
                    try:
                        with urlopen(base + "/v1/models", timeout=5) as response:
                            json.load(response)
                        break
                    except URLError:
                        if time.monotonic() >= deadline:
                            raise TimeoutError("real inference server did not become ready")
                        time.sleep(1)
                with urlopen(Request(base + "/v1/chat/completions", data=json.dumps(request).encode(),
                                     headers={"Content-Type": "application/json"}), timeout=600) as response:
                    result = json.load(response)
                (args.output / "response.json").write_text(json.dumps(result, indent=2) + "\n")
                if result["choices"][0]["message"]["content"] != fixture["expected_text"]:
                    raise RuntimeError("packaged executable differs from the maintained greedy completion golden")
                if result.get("usage", {}).get("completion_tokens", 0) < 8:
                    raise RuntimeError("completion did not exercise eight decode steps")
            finally:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        evidence = {"binary_sha256": sha(executable.parent / "guild.bin") if (executable.parent / "guild.bin").exists() else sha(executable),
                    "fixture_sha256": sha(args.fixture), "model_manifest_sha256": sha(manifest_path),
                    "metadata": metadata, "command": command,
                    "actual_gpu_inference": True, "greedy_completion_matches": True}
        (args.output / "verification.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print("Shipped CUDA executable matched the real-model greedy completion golden")


if __name__ == "__main__":
    main()
