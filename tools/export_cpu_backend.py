#!/usr/bin/env python3
"""One-time audited CPU arithmetic export; never used by an external build.

The generated repository owns its copies after migration. Core retains the
validated migration reference until external-only execution has passed its gates.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("destination",type=Path)
    args=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    revision="42018db2e63e7febbdb7bcf58a6392166518cdc1"
    sources=[f"src/kernels/cpu/{s}.cpp" for s in
             ("expert","features","q2_avx2","native_expert","iq_avx512","iq_avx2","kq_avx2","kq_avx1")]
    headers=[f"include/guild/kernels/cpu/{s}.hpp" for s in
             ("expert","native_expert","expert_layout","iq_avx512","iq_avx2","kq_avx2","kq_avx1")]
    files=sources+headers+["third_party/ggml/ggml-common.h","third_party/ggml/LICENSE","LICENSE"]
    records=[]
    for name in files:
        data=subprocess.check_output(["git","show",f"{revision}:{name}"],cwd=root)
        # The one representation change is explicitly recorded: use the new C
        # format contract instead of exporting a private C++ NativeFmt object.
        if name.endswith("/native_expert.hpp"):
            data=(root/name).read_bytes()
        target=args.destination/name
        if target.exists():
            raise RuntimeError(f"refusing to overwrite {target}")
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(data)
        records.append({"source":name,"revision":revision,"sha256":hashlib.sha256(data).hexdigest()})
    probe=subprocess.check_output(["git","show",f"{revision}:src/kernels/cpu/expert_layout.cpp"],cwd=root,text=True)
    probe=probe.split("#if !defined(GUILD_NATIVE_EXPERTS)")[0]+"\n} // namespace guild::kernels::cpu\n"
    (args.destination/"src/kernels/cpu/isa.cpp").write_text(probe)
    (args.destination/"SOURCE_PROVENANCE.json").write_text(json.dumps(records,indent=2)+"\n")


if __name__=="__main__":
    main()
