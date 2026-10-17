#!/usr/bin/env python3
"""Stage a Linux CUDA distribution, its provenance and a reproducible archive.

Bundle the exact CUDA runtime/BLAS libraries resolved by the built executable.
The NVIDIA driver and the listed Linux system libraries remain prerequisites.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tarfile
import tempfile


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def linked_libraries(executable, env):
    result = subprocess.run(["ldd", str(executable)], env=env, check=True,
                            capture_output=True, text=True)
    if "not found" in result.stdout:
        raise RuntimeError("executable has unresolved runtime libraries: " + result.stdout)
    return dict(re.findall(r"^\s*(\S+) => (/\S+) \(", result.stdout, re.MULTILINE))


def write_archive(source, archive, epoch):
    def deterministic(member):
        member.uid = member.gid = 0
        member.uname = member.gname = ""
        member.mtime = epoch
        member.pax_headers = {}
        return member

    # tarfile's w:gz embeds the current time and output filename in the gzip header.
    with archive.open("wb") as output:
        with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=epoch) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as tar:
                tar.add(source, arcname=source.name, filter=deterministic)


def cuda_license_files(cuda_root, dependencies, docs_root=Path("/usr/share/doc")):
    bases = [cuda_root, cuda_root / "licenses", cuda_root / "doc", cuda_root / "DOCS",
             docs_root / "nvidia-cuda-toolkit"]
    # NVIDIA's official CUDA13 Debian image ships the redistribution terms in
    # each installed runtime package's copyright file, outside the toolkit tree.
    for dependency in dependencies:
        name = dependency["name"]
        major = name.rsplit(".", 1)[1]
        if name.startswith("libcudart.so."):
            bases.extend(docs_root.glob(f"cuda-cudart-{major}-*"))
        elif name.startswith("libcublas.so."):
            bases.extend(docs_root.glob(f"libcublas-{major}-*"))
    licenses = set()
    for base in bases:
        if base.is_dir():
            for pattern in ("*LICENSE*", "*EULA*", "copyright"):
                licenses.update(path.resolve() for path in base.glob(pattern) if path.is_file())
    return licenses


def stage(build, dest, cuda_root, allow_native_cpu=False):
    root = Path(__file__).resolve().parents[1]
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "AMD64"):
        raise RuntimeError("only Linux x86-64 distributions are implemented")
    with tempfile.TemporaryDirectory(prefix="distribution-data-", dir=build) as data:
        env = dict(os.environ, GUILD_DATA_DIR=data)
        env.pop("LD_LIBRARY_PATH", None)
        info = json.loads(subprocess.check_output([str(build / "guild"), "--version", "--json"],
                                                 env=env, text=True))
        if info["backend"] != "cuda" or not info["bundled_cpu"]["enabled"] or not info["bundled_cuda"]["enabled"]:
            raise RuntimeError("distribution requires CUDA with bundled CPU/CUDA baselines")
        if not info["bundled_cpu"]["pinned"] or info["external_packages"]:
            raise RuntimeError("distribution requires pinned CPU sources and an empty package store")
        if not info["portable_cpu_baseline"] and (not allow_native_cpu or info["channel"] != "development"):
            raise RuntimeError("public distributions require GUILD_PORTABLE=ON; local development may use --allow-native-cpu")
        resolved = linked_libraries(build / "guild", env)
        dependencies = []
        (dest / "lib").mkdir()
        (dest / "licenses").mkdir()
        for library in ("cudart", "cublas", "cublasLt"):
            matches = [(name, Path(path)) for name, path in resolved.items()
                       if re.fullmatch(rf"lib{library}\.so\.[0-9]+", name)]
            if len(matches) != 1:
                raise RuntimeError(f"expected one dynamically linked lib{library}, got {matches}")
            name, path = matches[0]
            shutil.copyfile(path.resolve(), dest / "lib" / name)
            (dest / "lib" / name).chmod(0o644)
            dependencies.append({"name": name, "sha256": sha(dest / "lib" / name)})

        licenses = cuda_license_files(cuda_root, dependencies)
        if not licenses:
            raise RuntimeError("CUDA redistribution license files must be included")
        for index, path in enumerate(sorted(licenses)):
            shutil.copyfile(path, dest / "licenses" / f"cuda-{index}-{path.name}")
        for source, name in ((root / "LICENSE", "Guild-Strata-MIT.txt"),
                             (root / "third_party/ggml/LICENSE", "GGML-MIT.txt"),
                             (build / "_deps/guild_cpu_backend-src/LICENSE", "CPU-MIT.txt")):
            if not source.is_file():
                raise RuntimeError(f"missing source license: {source}")
            shutil.copyfile(source, dest / "licenses" / name)
        shutil.copyfile(build / "guild", dest / "guild.bin")
        (dest / "guild.bin").chmod(0o755)
        (dest / "guild").write_text(
            '#!/bin/sh\nset -eu\nHERE=$(dirname "$(readlink -f "$0")")\n'
            'export LD_LIBRARY_PATH="$HERE/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"\n'
            'exec "$HERE/guild.bin" "$@"\n')
        (dest / "guild").chmod(0o755)
        info["runtime_libraries"] = dependencies
        info["system_library_prerequisites"] = sorted(set(resolved) - {d["name"] for d in dependencies})
        info["advertised_configuration"] = {
            "platform": "linux-x86_64",
            "build_os": platform.freedesktop_os_release().get("PRETTY_NAME", platform.platform()),
            "cpu": "AVX2, FMA, F16C, BMI2" if info["portable_cpu_baseline"] else "build-host-specific development artifact",
            "gpu": "NVIDIA driver compatible with the embedded CUDA toolchain and GPU architectures",
            "model_data": "GGUF and prepared Guild pack required; model data not included",
            "kernels": "bundled CPU and legacy CUDA baseline; external packages optional",
            "gpu_execution_in_this_build_job": False,
        }
        # Exercise the launcher and confirm CUDA resolves from the distribution.
        launch_env = dict(env, LD_LIBRARY_PATH=str(dest / "lib"))
        staged = linked_libraries(dest / "guild.bin", launch_env)
        for dependency in dependencies:
            if Path(staged[dependency["name"]]).resolve().parent != (dest / "lib").resolve():
                raise RuntimeError("staged CUDA dependency resolved outside the distribution")
        actual = json.loads(subprocess.check_output([str(dest / "guild"), "--version", "--json"], env=env, text=True))
        if actual != {key: value for key, value in info.items()
                      if key not in ("runtime_libraries", "system_library_prerequisites", "advertised_configuration")}:
            raise RuntimeError("staged executable provenance changed")
        info["executable_sha256"] = sha(dest / "guild.bin")
        (dest / "guild-release.json").write_text(json.dumps(info, indent=2) + "\n")
        return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cuda-root", type=Path, default=Path("/usr/local/cuda"))
    parser.add_argument("--allow-native-cpu", action="store_true", help="local development artifacts only")
    args = parser.parse_args()
    build, dest = args.build.resolve(), args.output.resolve()
    archive = dest.parent / (dest.name + ".tar.gz")
    checksums = archive.parent / (archive.name + ".sha256")
    if any(path.exists() for path in (dest, archive, checksums)):
        parser.error("refusing to replace an existing distribution or archive")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".guild-staging-", dir=dest.parent) as work:
        staged = Path(work) / dest.name
        staged.mkdir()
        stage(build, staged, args.cuda_root, args.allow_native_cpu)
        staged.rename(dest)
    write_archive(dest, archive, int(os.environ.get("SOURCE_DATE_EPOCH", "0")))
    checksums.write_text(sha(archive) + "  " + archive.name + "\n")
    print(json.dumps({"archive": str(archive), "sha256": sha(archive),
                      "manifest": str(dest / "guild-release.json")}, indent=2))


if __name__ == "__main__":
    main()
