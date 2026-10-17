#!/usr/bin/env python3
"""Fresh-process signed runner integration against separately installed package inputs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import hashlib
import shutil
import csv
import statistics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build",type=Path,default=Path("build-cuda12"))
    parser.add_argument("--package",type=Path,required=True)
    parser.add_argument("--reference",type=Path)
    parser.add_argument("--additional-package",type=Path,action="append",default=[])
    parser.add_argument("--upgrade-package",type=Path)
    parser.add_argument("--verify-bundled-fallback",action="store_true")
    parser.add_argument("--artifact-output",type=Path)
    parser.add_argument("--benchmark-output",type=Path)
    parser.add_argument("--benchmark-prompts",type=int,nargs="+",default=[512,4096,8192,16384])
    args=parser.parse_args()
    build=args.build.resolve()
    core_digest=hashlib.sha256((build/"guild").read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory(prefix="external-runner-",dir=build) as work:
        work=Path(work)
        env=os.environ.copy();env["GUILD_DATA_DIR"]=str(work/"data")
        cli=str(build/"guild");test=str(build/"external_runner_parity_test")
        key=json.loads(subprocess.check_output([cli,"kernel","keygen","--json"],text=True,env=env))
        seed=work/"signing-seed"
        fd=os.open(seed,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,"w") as f:
            f.write(key["private_seed_hex"]+"\n")
        subprocess.run([cli,"kernel","trust-key","--publisher","org.guild-inference","--key-id","integration",
                        "--pubkey",key["public_key_hex"]],env=env,check=True)
        reference=args.reference.resolve() if args.reference else work/"reference.bin"
        if not args.reference:
            subprocess.run([test,"write",str(reference),"builtin.guild-kernels-ornith"],env=env,check=True)
        bundle=work/"qwen.gkernel"
        subprocess.run([cli,"kernel","sign",str(args.package.resolve()),"--key-id","integration",
                        "--seed-file",str(seed),"--bundle",str(bundle)],env=env,check=True)
        subprocess.run([cli,"kernel","install",str(bundle)],env=env,check=True)
        base_cpu_version=None
        for index, package in enumerate(args.additional_package):
            extra=work/f"extra-{index}.gkernel"
            subprocess.run([cli,"kernel","sign",str(package.resolve()),"--key-id","integration",
                            "--seed-file",str(seed),"--bundle",str(extra)],env=env,check=True)
            subprocess.run([cli,"kernel","install",str(extra)],env=env,check=True)
            metadata=json.loads((package/".gkmetadata").read_text())
            if metadata["package_id"]=="org.guild.kernels.cpu":
                env["GUILD_EXPECT_CPU_PACKAGE"]="org.guild.kernels.cpu"
                env["GUILD_EXPECT_CPU_VERSION"]=metadata["version"]
                base_cpu_version=metadata["version"]
            if metadata["package_id"]=="org.guild.kernels.cuda":
                env["GUILD_EXPECT_CUDA_PACKAGE"]="org.guild.kernels.cuda"
        # Installation, execution and inspection are separate processes.
        subprocess.run([test,"compare",str(reference),"org.guild.runners.qwen"],env=env,check=True)
        if args.upgrade_package:
            if base_cpu_version is None:
                raise RuntimeError("CPU upgrade verification requires an installed CPU base package")
            upgrade=work/"cpu-upgrade.gkernel"
            subprocess.run([cli,"kernel","sign",str(args.upgrade_package.resolve()),"--key-id","integration",
                            "--seed-file",str(seed),"--bundle",str(upgrade)],env=env,check=True)
            subprocess.run([cli,"kernel","install",str(upgrade)],env=env,check=True)
            env["GUILD_EXPECT_CPU_VERSION"]=json.loads((args.upgrade_package/".gkmetadata").read_text())["version"]
            subprocess.run([cli,"kernel","enable","org.guild.runners.qwen"],env=env,check=True)
            subprocess.run([test,"compare",str(reference),"org.guild.runners.qwen"],env=env,check=True)
            subprocess.run([cli,"kernel","rollback","org.guild.kernels.cpu"],env=env,check=True)
            env["GUILD_EXPECT_CPU_VERSION"]=base_cpu_version
            subprocess.run([cli,"kernel","enable","org.guild.runners.qwen"],env=env,check=True)
            subprocess.run([test,"compare",str(reference),"org.guild.runners.qwen"],env=env,check=True)
        if args.verify_bundled_fallback:
            if base_cpu_version is None:
                raise RuntimeError("bundled CPU fallback verification requires an external CPU base package")
            subprocess.run([cli,"kernel","disable","org.guild.kernels.cpu"],env=env,check=True)
            if "GUILD_EXPECT_CUDA_PACKAGE" in env:
                subprocess.run([cli,"kernel","disable","org.guild.kernels.cuda"],env=env,check=True)
            fallback_env=env.copy()
            for name in ("GUILD_EXPECT_CPU_PACKAGE","GUILD_EXPECT_CPU_VERSION","GUILD_EXPECT_CUDA_PACKAGE"):
                fallback_env.pop(name,None)
            fallback_env["GUILD_EXPECT_BUNDLED_CPU"]="1"
            subprocess.run([test,"compare",str(reference),"builtin.guild-kernels-ornith"],env=fallback_env,check=True)
            subprocess.run([cli,"kernel","enable","org.guild.kernels.cpu"],env=env,check=True)
            if "GUILD_EXPECT_CUDA_PACKAGE" in env:
                subprocess.run([cli,"kernel","enable","org.guild.kernels.cuda"],env=env,check=True)
        seed.unlink()
        if args.benchmark_output:
            subprocess.run([cli,"kernel","enable","org.guild.runners.qwen"],env=env,check=True)
            args.benchmark_output.mkdir(parents=True,exist_ok=True)
            results=[]
            for prompt in args.benchmark_prompts:
                dest=(args.benchmark_output/f"external-p{prompt}.csv").resolve()
                command=[str(build/"ornith_prefill_bench"),"--context","32768","--prompt",str(prompt),
                         "--generate","128","--batch","256","--runs","10","--full-warmup","--csv",str(dest)]
                with dest.with_suffix(".log").open("w") as log:
                    subprocess.run(command,env=env,check=True,stdout=log,stderr=log)
                with dest.open() as f: rows=list(csv.DictReader(f))
                assert len(rows)==10 and all(int(r["generated_tokens"])==128 for r in rows)
                results.append({"prompt":prompt,"prefill":statistics.median(float(r["prefill_tps"]) for r in rows),
                                "decode":statistics.median(float(r["decode_tps"]) for r in rows)})
                (args.benchmark_output/"summary.json").write_text(json.dumps(results,indent=2)+"\n")
        if args.artifact_output:
            args.artifact_output.mkdir(parents=True,exist_ok=True)
            for artifact in work.glob("*.gkernel"):
                shutil.copy2(artifact,args.artifact_output/artifact.name)
            (args.artifact_output/"verification.json").write_text(json.dumps({
                "publisher":"org.guild-inference","key_id":"integration","public_key_hex":key["public_key_hex"],
                "guild_sha256":core_digest,"test_only_signing_identity":True,
                "artifacts":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in work.glob("*.gkernel")}
            },indent=2)+"\n")
        assert core_digest==hashlib.sha256((build/"guild").read_bytes()).hexdigest(), "Core binary changed during package update"


if __name__=="__main__":
    main()
