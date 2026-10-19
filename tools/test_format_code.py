#!/usr/bin/env python3
"""Guards for the formatting tooling itself: the eligible file list, the ignore file, and the pinned rules.

The full-tree check is a separate CI step (`python tools/format_code.py --check`); these tests are cheap and run
with the other `tools/test_*.py` policy tests, without clang-format installed.
"""
from __future__ import annotations

import shutil
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import format_code

ROOT = format_code.ROOT


def read_config() -> dict[str, str]:
    """The `.clang-format` keys that are not a nested block, as raw strings."""
    lines = (ROOT / ".clang-format").read_text().splitlines()
    entries = {}
    for line in lines:
        if line.startswith("#") or not line.strip() or line.startswith("-"):
            continue
        key, _, value = line.partition(":")
        entries[key.strip()] = value.strip()
    return entries


class EligibleFiles(unittest.TestCase):
    def test_list_is_first_party_only(self):
        files = format_code.tracked_sources([])
        names = [str(f.relative_to(ROOT)) for f in files]
        self.assertGreater(len(names), 400)
        for name in names:
            self.assertFalse(name.startswith(format_code.EXCLUDED_PREFIXES), name)
            self.assertIn(Path(name).suffix, format_code.SOURCES, name)

    def test_list_covers_every_subsystem(self):
        names = {str(f.relative_to(ROOT)) for f in format_code.tracked_sources([])}
        for expected in ("include/guild/abi/kernel_abi.h", "src/runtime/model.cpp", "src/kernels/cpu/pool.cpp",
                         "tests/core/gexec_optimizer_test.cpp", "plugins/scan_module/scan.c",
                         "sycl/src/core/session.cpp"):
            self.assertIn(expected, names)

    def test_vendored_and_generated_trees_are_out(self):
        names = {str(f.relative_to(ROOT)) for f in format_code.tracked_sources([])}
        for excluded in ("third_party/llama.cpp/ggml/src/ggml.c",
                         "sycl/include/dpct/dpl_extras/algorithm.h",
                         "sycl/third_party/ggml/ggml-common.h"):
            self.assertNotIn(excluded, names)
        self.assertFalse([n for n in names if n.startswith("bench/results/")])

    def test_prefix_filter_and_typo_detection(self):
        self.assertTrue(format_code.tracked_sources(["src/gexec"]))
        self.assertTrue(all(str(f).startswith(str(ROOT / "src/gexec")) for f in format_code.tracked_sources(["src/gexec"])))
        self.assertEqual(format_code.missing_prefixes(["src/gexec"]), [])
        self.assertEqual(format_code.missing_prefixes(["src/not-a-directory"]), ["src/not-a-directory"])


class IgnoreFile(unittest.TestCase):
    def test_covers_the_tool_exclusions(self):
        patterns = [line.strip() for line in (ROOT / ".clang-format-ignore").read_text().splitlines()
                    if line.strip() and not line.startswith("#")]
        for prefix in format_code.EXCLUDED_PREFIXES:
            self.assertIn(f"{prefix}**", patterns)


class Config(unittest.TestCase):
    def test_pinned_conventions(self):
        config = read_config()
        expected = {
            "BasedOnStyle": "LLVM",
            "ColumnLimit": "120",
            "IndentWidth": "4",
            "TabWidth": "4",
            "UseTab": "Never",
            "SortIncludes": "Never",
            "BreakBeforeBraces": "Attach",
            "PointerAlignment": "Left",
            "SpaceAfterCStyleCast": "true",
            "AllowShortIfStatementsOnASingleLine": "Never",
            "ReflowComments": "true",
        }
        for key, value in expected.items():
            self.assertEqual(config.get(key), value, key)

    def test_expected_major_version(self):
        self.assertEqual(format_code.EXPECTED_MAJOR, 18)


@unittest.skipUnless(shutil.which("clang-format") or shutil.which("clang-format-18"),
                     "clang-format is not installed")
class ToolBehaviour(unittest.TestCase):
    def test_unformatted_tmp_file_is_detected_and_formatting_is_idempotent(self):
        import tempfile

        binary = format_code.find_clang_format()
        with tempfile.TemporaryDirectory() as tmp:
            scratch = Path(tmp) / "scratch.cpp"
            scratch.write_text("int  f(int x){if(x)return 1;return 0;}\n")
            _, before, formatted = format_code.format_one(binary, scratch)
            self.assertNotEqual(before, formatted)
            scratch.write_bytes(formatted)
            _, once, twice = format_code.format_one(binary, scratch)
            self.assertEqual(once, twice)


if __name__ == "__main__":
    unittest.main()
