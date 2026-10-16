#!/usr/bin/env python3
"""Create deterministic gzip archives for completed diagnostic traces and telemetry.

CSV and .command.json files are never modified. Originals remain on disk unless
--remove-originals is explicitly requested with --apply.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile

SCRIPT_VERSION = 1


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def read_stable_bytes(path):
    before = path.stat()
    data = path.read_bytes()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or len(data) != before.st_size:
        raise RuntimeError(f"input changed while reading: {path}")
    return data


def deterministic_gzip_bytes(data):
    return gzip.compress(data, compresslevel=6, mtime=0)


def classify_target(path):
    name = path.name
    if name.endswith(".telemetry.jsonl"):
        cmd_path = path.with_name(name[:-len(".telemetry.jsonl")] + ".command.json")
        if cmd_path.exists():
            try:
                meta = json.loads(cmd_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as e:
                return "telemetry_jsonl", False, f"unreadable command metadata: {e}"
            if meta.get("exit_code") is None or meta.get("state") == "running":
                return "telemetry_jsonl", False, "command still running or incomplete"
        return "telemetry_jsonl", True, None
    if path.suffix.lower() == ".tsv":
        return "activity_tsv", True, None
    if path.suffix.lower() == ".qdstrm":
        return "nsight_qdstrm", False, "unimported Nsight stream retained locally only"
    return None, False, None


def archive_directory(directory, apply=False, remove_originals=False, manifest_path=None):
    if remove_originals and not apply:
        raise ValueError("--remove-originals requires --apply")
    directory = directory.resolve()
    entries = []
    skipped = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        kind, eligible, reason = classify_target(path)
        if kind is None:
            continue
        rel = path.relative_to(directory).as_posix()
        if not eligible:
            skipped.append({"path": rel, "kind": kind, "reason": reason})
            continue
        raw = read_stable_bytes(path)
        raw_sha256 = sha256_bytes(raw)
        gz_path = path.with_name(path.name + ".gz")
        gz_rel = gz_path.relative_to(directory).as_posix()
        compressed = deterministic_gzip_bytes(raw)
        gz_sha256 = sha256_bytes(compressed)
        if gz_path.exists():
            existing = gz_path.read_bytes()
            if sha256_bytes(existing) != gz_sha256:
                raise RuntimeError(f"refusing to overwrite mismatched archive: {gz_path}")
        if apply:
            if not gz_path.exists():
                fd, tmp_name = tempfile.mkstemp(prefix=gz_path.name + ".tmp.", dir=str(gz_path.parent))
                tmp_path = Path(tmp_name)
                try:
                    with os.fdopen(fd, "wb") as f:
                        f.write(compressed)
                    tmp_path.replace(gz_path)
                except Exception:
                    tmp_path.unlink(missing_ok=True)
                    raise
            if remove_originals:
                path.unlink()
        entries.append({
            "path": rel,
            "archive_path": gz_rel,
            "kind": kind,
            "uncompressed_bytes": len(raw),
            "compressed_bytes": len(compressed),
            "sha256": raw_sha256,
            "archive_sha256": gz_sha256,
            "applied": bool(apply),
            "removed_original": bool(apply and remove_originals),
        })
    manifest = {
        "script_version": SCRIPT_VERSION,
        "directory": str(directory),
        "applied": bool(apply),
        "entries": entries,
        "skipped": skipped,
    }
    if manifest_path is not None:
        manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--apply", action="store_true", help="write .gz files (default is dry-run)")
    parser.add_argument("--remove-originals", action="store_true", help="delete original after verified .gz write")
    parser.add_argument("--manifest", type=Path, help="optional path to write JSON manifest")
    args = parser.parse_args()
    if not args.directory.is_dir():
        parser.error("directory does not exist")
    manifest = archive_directory(args.directory, apply=args.apply,
                                 remove_originals=args.remove_originals, manifest_path=args.manifest)
    total_raw = sum(e["uncompressed_bytes"] for e in manifest["entries"])
    total_gz = sum(e["compressed_bytes"] for e in manifest["entries"])
    print(f"entries={len(manifest['entries'])} skipped={len(manifest['skipped'])} "
          f"raw_bytes={total_raw} gz_bytes={total_gz} applied={manifest['applied']}")


if __name__ == "__main__":
    main()
