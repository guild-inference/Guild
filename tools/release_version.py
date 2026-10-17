#!/usr/bin/env python3
"""Guild software version/revision policy, shared by builds, releases and updater.

Revision is an independent integer, not generic SemVer precedence metadata.
"""
import argparse
from dataclasses import dataclass
from functools import total_ordering
import json
from pathlib import Path
import re

NUMBER = r"(0|[1-9][0-9]*)"
SOFTWARE = re.compile(rf"^{NUMBER}\.{NUMBER}\.{NUMBER}(?:-([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?$")
TAG = re.compile(r"^v(.+?)(?:\+r(0|[1-9][0-9]*))?$")


@total_ordering
@dataclass(frozen=True)
class ReleaseVersion:
    version: str
    revision: int = 0

    def __post_init__(self):
        match = SOFTWARE.fullmatch(self.version)
        if not match or isinstance(self.revision, bool) or not isinstance(self.revision, int) or self.revision < 0:
            raise ValueError("invalid software version or non-negative integer revision")
        for part in (match[4] or "").split("."):
            if part.isdigit() and len(part) > 1 and part.startswith("0"):
                raise ValueError("numeric prerelease identifiers cannot have leading zeros")

    @classmethod
    def from_tag(cls, tag):
        match = TAG.fullmatch(tag)
        if not match:
            raise ValueError(f"invalid immutable release tag: {tag}")
        version = cls(match[1], int(match[2] or 0))
        if version.tag != tag:
            raise ValueError("revision zero uses the base tag, without +r0")
        return version

    @property
    def software_key(self):
        match = SOFTWARE.fullmatch(self.version)
        base = tuple(int(match[i]) for i in (1, 2, 3))
        prerelease = match[4]
        # Stable is greater than any prerelease of the same numeric version.
        identifiers = tuple((0, int(p)) if p.isdigit() else (1, p) for p in prerelease.split(".")) if prerelease else ()
        return base, prerelease is None, identifiers

    def __lt__(self, other):
        if not isinstance(other, ReleaseVersion):
            return NotImplemented
        return (self.software_key, self.revision) < (other.software_key, other.revision)

    @property
    def tag(self):
        return "v" + self.version + (f"+r{self.revision}" if self.revision else "")

    @property
    def display(self):
        return "v" + self.version + (f", r{self.revision}" if self.revision else "")


def source_metadata(path, tag=None):
    source = json.loads(Path(path).read_text())
    base = source["version"]
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", base):
        raise ValueError("source version must contain only the numeric MAJOR.MINOR.PATCH")
    software = base + ("-" + source["prerelease"] if source.get("prerelease") else "")
    release = ReleaseVersion(software, source.get("revision", 0))
    if tag:
        requested = ReleaseVersion.from_tag(tag)
        if requested.version != release.version:
            raise ValueError("tag software version/prerelease disagrees with VERSION.json")
        # Revision tags can rebuild the same source version. Release validation
        # separately checks that revisions do not hide behavior-changing edits.
        release = requested
    return {"version": base, "software_version": release.version, "prerelease": source.get("prerelease", ""),
            "revision": release.revision, "display_version": release.display,
            "channel": ("preview" if source.get("prerelease") else "stable") if tag else "development",
            "release_tag": tag if tag else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    metadata = sub.add_parser("metadata")
    metadata.add_argument("--source", type=Path, required=True)
    metadata.add_argument("--tag")
    metadata.add_argument("--output", type=Path)
    compare = sub.add_parser("compare")
    compare.add_argument("left")
    compare.add_argument("right")
    args = parser.parse_args()
    if args.command == "compare":
        left, right = ReleaseVersion.from_tag(args.left), ReleaseVersion.from_tag(args.right)
        print(-1 if left < right else 1 if right < left else 0)
    else:
        output = json.dumps(source_metadata(args.source, args.tag), indent=2) + "\n"
        if args.output:
            args.output.write_text(output)
        else:
            print(output, end="")


if __name__ == "__main__":
    main()
