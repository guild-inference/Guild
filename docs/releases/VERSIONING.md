# Versions, revisions and build identity

Guild is a v0.x project. `VERSION.json` owns the Core software version and
prerelease identifier. `sdk/VERSION` owns the public SDK version; the frozen
Kernel ABI version is defined in `include/guild/abi/kernel_abi.h`. Kernel and
runner packages have their own implementation versions. None of these versions
is inferred from another.

## Software version and release revision

- Software tags use `vMAJOR.MINOR.PATCH`, optionally with a prerelease such as
  `v0.2.0-rc.1`.
- A release-engineering rebuild uses a separate non-negative integer revision:
  `v0.2.0+r1` displays as `v0.2.0, r1`. The initial release has revision zero and
  uses `v0.2.0`. Each revision requires the immediately preceding revision tag.
- Compare the software version first, including SemVer prerelease ordering,
  then compare the revision numerically. `v0.2.0+r10` follows `v0.2.0+r2`, but
  precedes `v0.2.1`. Generic SemVer build-metadata comparison is insufficient.
- Revision-only changes are limited to `.github/workflows/`, `packaging/`,
  `release/` and `docs/releases/`. Runtime, API, arithmetic, installer or other
  software changes require a new patch/minor/major version. Keep `VERSION.json`
  unchanged for a revision; the tag supplies its revision integer.
- Tags and published assets are immutable. Fix a published mistake with a new
  version or revision; do not replace a tag or upload over an existing release.

`tools/release_version.py` is shared by builds and the release checker.
`tools/validate_release.py --tag TAG --require-tag` verifies source/tag agreement,
the checked-out commit, a clean source tree, revision-only paths, and maturity
gates. A release build also runs this validation from CMake.

An untagged build is a development build even if its numeric software version
matches a released version. Release tags must match both the numeric source
version and its prerelease identifier. The current `0.2.0-dev` source version
does not constitute a stable `v0.2.0` release.

## Reporting and source pins

`guild --version` prints the human-readable identity; `guild --version --json`
includes software version, revision, channel, source SHA, tracked-source state,
build timestamp, compiler/toolchain, GPU architectures, Kernel ABI/SDK versions,
bundled source provenance, and installed external packages and registrations.
Model-specific selected implementations are separate from the process registry;
the report labels that distinction. A corrupt package store produces a nonzero
exit and an explicit package error.

`backends.lock.json` pins the bundled CPU source and all three independent
integration inputs. The CPU source must match its exact SHA and be unmodified.
`GUILD_CPU_SOURCE_DIR` is an explicit development override and is reported as
unpinned. Release builds reject this override and CPU revision overrides.
CUDA and bundled Qwen policy still come from Core: the report includes the
actual Core source SHA and labels their incomplete extraction. The older CUDA
`revision` in the lock records the baseline lineage, not the source compiled by
the current build.

Build identity is generated at configure time. Reconfigure after changing the
checkout to update its SHA and tracked-source state. Set `SOURCE_DATE_EPOCH` to
the commit timestamp for deterministic build timestamps. The archive writer
also normalizes tar ownership/time and gzip filename/time. Byte-identical
archives of a fixed staging tree are tested; whole-toolchain reproducible
executable builds remain a separate gate.

## v1.0 gate

`release/compatibility-gates.json` records the current maturity state. A v1.0
tag requires at least five genuinely distinct validated architecture families,
stable interfaces, a validated package ecosystem, mature GPACK and GEXEC,
supported-hardware inference, reproducible releases, compatibility guarantees,
and regression coverage. Renamed variants of one family do not count as
distinct families. The current gates remain false and block v1.0 publication.

## Checking for updates

```
python3 tools/check_guild_update.py --current v0.2.0+r1 --channel stable
```

The checker considers only Core releases carrying one matching
`guild-release.json`. It ignores unrelated SDK/kernel tags and drafts, excludes
prereleases from the stable channel, validates manifest/tag agreement, and
compares software version then integer revision. `--channel preview` includes
prereleases. It reports availability and does not replace a running executable.
The installer is not yet wired to this checker.
