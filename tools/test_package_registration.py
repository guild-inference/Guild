#!/usr/bin/env python3
"""Installed independent-package signing/loading check; does not test inference."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
from package_linux_cuda import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--package", type=Path, action="append", required=True)
    args = parser.parse_args()
    build = args.build.resolve()
    cli = str(build / "guild")
    digest = sha(build / "guild")
    with tempfile.TemporaryDirectory(prefix="registration-test-", dir=build) as directory:
        work = Path(directory)
        env = dict(os.environ, GUILD_DATA_DIR=str(work / "data"))
        key = json.loads(subprocess.check_output([cli, "kernel", "keygen", "--json"], env=env, text=True))
        seed = work / "seed"
        with os.fdopen(os.open(seed, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as stream:
            stream.write(key["private_seed_hex"] + "\n")
        subprocess.run([cli, "kernel", "trust-key", "--publisher", "org.guild-inference",
                        "--key-id", "ci-ephemeral", "--pubkey", key["public_key_hex"]], env=env, check=True)
        expected = {}
        for index, package in enumerate(args.package):
            package = package.resolve()
            metadata = json.loads((package / ".gkmetadata").read_text())
            expected[metadata["package_id"]] = metadata["version"]
            bundle = work / f"package-{index}.gkernel"
            subprocess.run([cli, "kernel", "sign", str(package), "--key-id", "ci-ephemeral",
                            "--seed-file", str(seed), "--bundle", str(bundle)], env=env, check=True)
            subprocess.run([cli, "kernel", "install", str(bundle)], env=env, check=True)
        report = json.loads(subprocess.check_output([cli, "--version", "--json"], env=env, text=True))
        loaded = {p["package_id"]: p["version"] for p in report["external_packages"] if p["loaded_in_process"]}
        if loaded != expected:
            raise RuntimeError(f"installed packages were not loaded: {loaded} != {expected}")
        providers = {p["package_id"] for p in report["registered_kernels"] + report["registered_runners"] if p["enabled"]}
        if not set(expected).issubset(providers):
            raise RuntimeError("loaded packages did not register enabled implementations")
        if sha(build / "guild") != digest:
            raise RuntimeError("Core executable changed during package installation")
    print("Independent package registration passed (ephemeral test identity; no inference claim)")


if __name__ == "__main__":
    main()
