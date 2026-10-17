#!/usr/bin/env python3
"""Exercise real CLI provenance and fail-closed package-store diagnostics."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
from release_version import source_metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    args = parser.parse_args()
    build = args.build.resolve()
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="version-test-", dir=build) as directory:
        env = dict(os.environ, GUILD_DATA_DIR=directory)
        command = [str(build / "guild"), "--version"]
        report = json.loads(subprocess.check_output(command + ["--json"], env=env, text=True))
        expected = json.loads((build / "build-info.json").read_text())
        for key, value in expected.items():
            if report.get(key) != value:
                raise RuntimeError(f"CLI/build provenance differs at {key}")
        if report["software_version"] != source_metadata(root / "VERSION.json", report["release_tag"])["software_version"]:
            raise RuntimeError("CLI reports a stale software version")
        if not report["package_load_ok"] or report["external_packages"]:
            raise RuntimeError("empty test package store was not used")
        text = subprocess.check_output(command, env=env, text=True)
        if report["display_version"] not in text or report["kernel_sdk_version"] not in text:
            raise RuntimeError("human-readable report omits version information")
        # Initialize the exact store location through the CLI, then corrupt it.
        stores = list(Path(directory).rglob("trust_store.json"))
        if len(stores) != 1:
            raise RuntimeError("expected a single isolated trust store")
        stores[0].write_text("{broken-json")
        broken = subprocess.run(command + ["--json"], env=env, text=True, capture_output=True)
        failure = json.loads(broken.stdout)
        if broken.returncode == 0 or failure["package_load_ok"] or not failure.get("package_error"):
            raise RuntimeError("version reporting hid a corrupt enabled-package store")
    print("CLI text/JSON provenance and corrupt-store failure passed")


if __name__ == "__main__":
    main()
