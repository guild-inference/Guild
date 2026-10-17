#!/usr/bin/env python3
"""Validate immutable tags, revisions and the explicit Guild v1.0 milestone gate."""
import argparse
import json
from pathlib import Path
import subprocess
from release_version import ReleaseVersion, source_metadata

ROOT=Path(__file__).resolve().parents[1]
REVISION_ONLY_PREFIXES=(".github/workflows/", "packaging/", "release/", "docs/releases/")


def validate_revision_changes(paths):
    invalid=[p for p in paths if not p.startswith(REVISION_ONLY_PREFIXES)]
    if invalid:
        raise ValueError("release revisions cannot contain software changes; use a patch/minor version: "+", ".join(invalid))


def validate_v1_gate(version,gates):
    if version.software_key[0][0]<1:
        return
    families=gates.get("validated_architecture_families",[])
    required=("stable_interfaces","validated_package_ecosystem","mature_gpack","mature_gexec",
              "supported_hardware_inference","reproducible_releases","compatibility_guarantees","regression_coverage")
    if len(set(families))<5 or not all(gates.get(name) is True for name in required):
        raise ValueError("v1.0 requires five distinct validated families and all documented maturity gates")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag",required=True)
    p.add_argument("--source",type=Path,default=ROOT/"VERSION.json")
    p.add_argument("--gates",type=Path,default=ROOT/"release/compatibility-gates.json")
    p.add_argument("--require-tag",action="store_true")
    args=p.parse_args()
    version=ReleaseVersion.from_tag(args.tag)
    metadata=source_metadata(args.source,args.tag)
    validate_v1_gate(version,json.loads(args.gates.read_text()))
    if args.require_tag:
        tag_sha=subprocess.check_output(["git","rev-parse",f"refs/tags/{args.tag}^{{commit}}"],cwd=ROOT,text=True).strip()
        head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
        if tag_sha!=head:
            raise ValueError("release tag does not identify the checked-out commit")
        if subprocess.check_output(["git","status","--porcelain","--untracked-files=normal"],cwd=ROOT,text=True).strip():
            raise ValueError("release source tree contains local changes")
    if version.revision:
        previous=ReleaseVersion(version.version,version.revision-1).tag
        previous_sha=subprocess.check_output(["git","rev-parse",f"refs/tags/{previous}^{{commit}}"],cwd=ROOT,text=True).strip()
        changes=subprocess.check_output(["git","diff","--name-only",previous_sha,"HEAD"],cwd=ROOT,text=True).splitlines()
        validate_revision_changes(changes)
    print(json.dumps(metadata,indent=2))


if __name__=="__main__":main()
