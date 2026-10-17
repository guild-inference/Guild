#!/usr/bin/env python3
"""Record arithmetic machine-code fingerprints without changing build flags."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess

TARGETS = ("native_gu_rows", "native_down_rows", "native_quant_act", "native_quant_h",
           "kq256_gu_rows", "kq256_rows", "iq512_gu_rows", "ExpertPool::drain")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binary", action="append", nargs=2, metavar=("NAME", "PATH"), required=True)
    p.add_argument("--output", type=Path, required=True)
    args=p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results={}
    for name, path in args.binary:
        names=subprocess.check_output(["nm", "--defined-only", str(path)], text=True)
        symbols=[line.split()[-1] for line in names.splitlines()
                 if len(line.split())>=3 and line.split()[-2] in ("t", "T", "w", "W")]
        demangled=subprocess.check_output(["c++filt"], input="\n".join(symbols), text=True).splitlines()
        result={}
        for raw, pretty in zip(symbols, demangled):
            if not pretty.startswith("guild::kernels::cpu::") or not any(pretty.split("(")[0].endswith(t) for t in TARGETS) or "[clone" in pretty:
                continue
            disassembly=subprocess.check_output(["objdump", "-d", "--no-show-raw-insn", f"--disassemble={raw}", path], text=True)
            leaf=next(t for t in TARGETS if t+"(" in pretty)
            (args.output/f"{name}-{leaf.replace('::','-')}.asm").write_text(disassembly)
            lines=[]
            for line in disassembly.splitlines():
                if not re.match(r"\s*[0-9a-f]+:\s+",line): continue
                instruction=re.sub(r"^\s*[0-9a-f]+:\s+", "", line).split("#")[0].strip()
                # Ignore relocation addresses while retaining instruction and
                # register choices. The raw .asm remains available for review.
                instruction=re.sub(r"[0-9a-f]+\s+<([^>]+)>",r"<\1>",instruction)
                instruction=re.sub(r"_ZN[^ >]+", "<cxx-symbol>",instruction)
                instruction=re.sub(r"\b[0-9a-f]+\(%rip\)","<relocation>(%rip)",instruction)
                lines.append(instruction)
            result[leaf]={"symbol":pretty,"instruction_count":len(lines),
                          "normalized_sha256":hashlib.sha256("\n".join(lines).encode()).hexdigest(),
                          "opcodes":dict(Counter(line.split()[0] for line in lines))}
            if not lines:
                raise RuntimeError(f"no disassembly for executable symbol {raw}")
        results[name]=result
    (args.output/"codegen.json").write_text(json.dumps(results,indent=2)+"\n")


if __name__=="__main__":main()
