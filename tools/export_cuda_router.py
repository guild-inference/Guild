#!/usr/bin/env python3
"""Export the validated router as the first CUDA-repository migration slice."""
import argparse
from pathlib import Path
import subprocess

p=argparse.ArgumentParser(description=__doc__)
p.add_argument("destination",type=Path)
args=p.parse_args()
root=Path(__file__).resolve().parents[1]
revision="42018db2e63e7febbdb7bcf58a6392166518cdc1"
source=subprocess.check_output(["git","show",f"{revision}:plugins/ornith_prefill_ext/ornith_prefill_plugin.cu"],cwd=root,text=True)
start=source.index("guild_status_t ext_finite_query_scratch")
end=source.index("guild_status_t plugin_init",start)
source=source[:start]+source[end:]
start=source.index("    // 1. Register CPU finite check op")
end=source.index("#if defined(GUILD_ENABLE_CUDA)",start)
source=source[:start]+"    guild_status_t st = GUILD_STATUS_OK;\n\n"+source[end:]
source=source.replace("ext.guild-kernels-ornith-prefill","org.guild.kernels.cuda")
source=source.replace("ext.cuda.moe.router_topk_bf16","external.cuda.router.bf16.v1")
source=source.replace("org.illatio.guild","org.guild-inference")
path=args.destination/"src/router.cu"
if path.exists():raise RuntimeError(f"refusing to overwrite {path}")
path.parent.mkdir(parents=True,exist_ok=True)
path.write_text(source)
(args.destination/"LICENSE").write_bytes(subprocess.check_output(["git","show",f"{revision}:LICENSE"],cwd=root))
