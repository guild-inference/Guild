#!/usr/bin/env python3
"""Package Linux CPU and Windows CPU/CUDA preview variants and merge release manifests."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import zipfile
from package_linux_cuda import sha, write_archive


def write_deterministic_zip(source_dir: Path, archive_path: Path, epoch: int):
    dt = (2026, 1, 1, 0, 0, 0)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(source_dir.rglob("*")):
            if path.is_dir():
                continue
            rel = path.relative_to(source_dir.parent).as_posix()
            info = zipfile.ZipInfo(rel, date_time=dt)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o755 if path.suffix.lower() in (".exe", ".dll", ".bat", ".ps1") else 0o644) << 16
            zf.writestr(info, path.read_bytes())


def copy_core_licenses(root: Path, build: Path, dest_licenses: Path):
    dest_licenses.mkdir(parents=True, exist_ok=True)
    pairs = [
        (root / "LICENSE", "Guild-Strata-MIT.txt"),
        (root / "third_party/ggml/LICENSE", "GGML-MIT.txt"),
        (build / "_deps/guild_cpu_backend-src/LICENSE", "CPU-MIT.txt"),
    ]
    cuda_lic = build / "_deps/guild_cuda_backend-src/LICENSE"
    if cuda_lic.is_file():
        pairs.append((cuda_lic, "CUDA-Backend-MIT.txt"))
    for src, name in pairs:
        if src.is_file():
            shutil.copyfile(src, dest_licenses / name)


def stage_linux_cpu(build: Path, dest: Path):
    root = Path(__file__).resolve().parents[1]
    exe = build / "guild"
    if not exe.is_file():
        raise RuntimeError(f"missing built executable: {exe}")
    with tempfile.TemporaryDirectory(prefix="cpu-dist-data-", dir=build) as data:
        env = dict(os.environ, GUILD_DATA_DIR=data)
        info = json.loads(subprocess.check_output([str(exe), "--version", "--json"], env=env, text=True))
        if not info.get("portable_cpu_baseline"):
            raise RuntimeError("Linux CPU distribution requires GUILD_PORTABLE=ON")
        copy_core_licenses(root, build, dest / "licenses")
        shutil.copyfile(exe, dest / "guild.bin")
        (dest / "guild.bin").chmod(0o755)
        (dest / "guild").write_text(
            '#!/bin/sh\nset -eu\nHERE=$(dirname "$(readlink -f "$0")")\nexec "$HERE/guild.bin" "$@"\n'
        )
        (dest / "guild").chmod(0o755)
        info["executable_sha256"] = sha(dest / "guild.bin")
        info["variant_id"] = "linux-x86_64-cpu"
        info["advertised_configuration"] = {
            "platform": "linux-x86_64",
            "cpu": "AVX2, FMA, F16C, BMI2",
            "gpu": "none (management, inspection, GPACK conversion, and mock/preview TUI)",
        }
        (dest / "guild-release.json").write_text(json.dumps(info, indent=2) + "\n")
        return info


def stage_windows(build: Path, dest: Path, variant_id: str, cuda_bin_dir: Path | None = None,
                  extra_dll_dirs: list[Path] | None = None):
    root = Path(__file__).resolve().parents[1]
    candidates = [build / "Release" / "guild.exe", build / "guild.exe"]
    exe = next((c for c in candidates if c.is_file()), None)
    if exe is None:
        raise RuntimeError(f"missing built Windows executable in {build}")

    with tempfile.TemporaryDirectory(prefix="win-dist-data-") as data:
        env = dict(os.environ, GUILD_DATA_DIR=data)
        info = json.loads(subprocess.check_output([str(exe), "--version", "--json"], env=env, text=True))
        copy_core_licenses(root, build, dest / "licenses")
        shutil.copyfile(exe, dest / "guild.exe")

        runtime_dlls = []
        search_dirs = [exe.parent]
        if extra_dll_dirs:
            search_dirs.extend(extra_dll_dirs)
        if cuda_bin_dir and cuda_bin_dir.is_dir():
            search_dirs.append(cuda_bin_dir)

        for d in search_dirs:
            if not d.is_dir():
                continue
            for dll in sorted(d.glob("*.dll")):
                low = dll.name.lower()
                if any(low.startswith(prefix) for prefix in ("libcurl", "zlib", "cudart64_", "cublas64_", "cublaslt64_")):
                    target = dest / dll.name
                    if not target.exists():
                        shutil.copyfile(dll, target)
                        runtime_dlls.append({"name": dll.name, "sha256": sha(target)})

        # Verify staged guild.exe runs cleanly from the staged directory
        staged_info = json.loads(
            subprocess.check_output([str(dest / "guild.exe"), "--version", "--json"], env=env, text=True)
        )
        if staged_info.get("commit") != info.get("commit"):
            raise RuntimeError("staged Windows executable commit mismatch")

        info["runtime_libraries"] = runtime_dlls
        info["executable_sha256"] = sha(dest / "guild.exe")
        info["variant_id"] = variant_id
        info["advertised_configuration"] = {
            "platform": "windows-x86_64",
            "cpu": "AVX2, FMA, F16C, BMI2",
            "gpu": "NVIDIA CUDA" if "cuda" in variant_id else "CPU management & TUI",
        }
        (dest / "guild-release.json").write_text(json.dumps(info, indent=2) + "\n")
        return info


def merge_manifests(primary_manifest_path: Path, variant_specs: list[str], output_path: Path):
    base = json.loads(primary_manifest_path.read_text())
    variants = {}
    for spec in variant_specs:
        # Format: variant_id=archive_path:manifest_path
        vid, rest = spec.split("=", 1)
        arch_str, man_str = rest.split(":", 1)
        arch_path = Path(arch_str)
        man_path = Path(man_str)
        mdata = json.loads(man_path.read_text())
        variants[vid] = {
            "variant_id": vid,
            "archive": arch_path.name,
            "sha256": sha(arch_path),
            "size_bytes": arch_path.stat().st_size,
            "executable_sha256": mdata["executable_sha256"],
            "backend": mdata.get("backend", "none"),
            "gpu_toolchain": mdata.get("gpu_toolchain", "none"),
            "portable_cpu_baseline": mdata.get("portable_cpu_baseline", True),
        }
    base["variants"] = variants
    output_path.write_text(json.dumps(base, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_lcpu = sub.add_parser("linux-cpu")
    p_lcpu.add_argument("--build", type=Path, required=True)
    p_lcpu.add_argument("--output", type=Path, required=True)

    p_win = sub.add_parser("windows")
    p_win.add_argument("--build", type=Path, required=True)
    p_win.add_argument("--output", type=Path, required=True)
    p_win.add_argument("--variant-id", required=True)
    p_win.add_argument("--cuda-bin", type=Path)
    p_win.add_argument("--dll-dir", type=Path, action="append", default=[])

    p_merge = sub.add_parser("merge")
    p_merge.add_argument("--primary", type=Path, required=True)
    p_merge.add_argument("--variant", action="append", required=True)
    p_merge.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    epoch = int(os.environ.get("SOURCE_DATE_EPOCH", "0"))

    if args.cmd == "linux-cpu":
        dest = args.output.resolve()
        archive = dest.parent / (dest.name + ".tar.gz")
        checksum = archive.parent / (archive.name + ".sha256")
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True)
        stage_linux_cpu(args.build.resolve(), dest)
        write_archive(dest, archive, epoch)
        checksum.write_text(f"{sha(archive)}  {archive.name}\n")
        print(json.dumps({"archive": str(archive), "sha256": sha(archive)}, indent=2))
    elif args.cmd == "windows":
        dest = args.output.resolve()
        archive = dest.parent / (dest.name + ".zip")
        checksum = archive.parent / (archive.name + ".sha256")
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True)
        stage_windows(args.build.resolve(), dest, args.variant_id, args.cuda_bin, args.dll_dir)
        write_deterministic_zip(dest, archive, epoch)
        checksum.write_text(f"{sha(archive)}  {archive.name}\n")
        print(json.dumps({"archive": str(archive), "sha256": sha(archive)}, indent=2))
    elif args.cmd == "merge":
        merge_manifests(args.primary.resolve(), args.variant, args.output.resolve())


if __name__ == "__main__":
    main()
