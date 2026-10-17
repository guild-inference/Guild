#!/usr/bin/env python3
"""Verify configured CPU source ownership and record external-only archive evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build",type=Path,default=Path("build-cuda12"))
    p.add_argument("--external-only",action="store_true")
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    build=args.build.resolve()
    database=json.loads((build/"compile_commands.json").read_text())
    files=[entry["file"] for entry in database if "CMakeFiles/guild_kernels_cpu.dir/" in entry.get("command","")]
    arithmetic={"expert.cpp","q2_avx2.cpp","native_expert.cpp","iq_avx512.cpp","iq_avx2.cpp",
                "kq_avx2.cpp","kq_avx1.cpp","canonical_avx2.cpp","backend.cpp","vector_ops.cpp"}
    root=Path(__file__).resolve().parents[1]
    for file in files:
        if Path(file).name in arithmetic and Path(file).is_relative_to(root/"src"):
            raise RuntimeError(f"Core still compiles an in-tree arithmetic implementation: {file}")
    members=subprocess.check_output(["ar","t",str(build/"libguild_kernels_cpu.a")],text=True).splitlines()
    if args.external_only:
        if any(Path(file).name in arithmetic for file in files):
            raise RuntimeError("external-only configuration still compiles bundled arithmetic")
        if set(members)!={"pool.cpp.o","pack_layout.cpp.o","features.cpp.o","isa.cpp.o","external_only.cpp.o"}:
            raise RuntimeError(f"unexpected external-only archive members: {members}")
    result={"external_only":args.external_only,"cpu_translation_units":files,"archive_members":members,
            "guild_sha256":hashlib.sha256((build/"guild").read_bytes()).hexdigest()}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n")


if __name__=="__main__":main()
