"""Adaptive decimal download units share the transfer's scale."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import setup


class DownloadProgress(unittest.TestCase):
    def test_examples(self):
        self.assertEqual(setup.format_download_progress(7_700_000, 10_000_000), "7.7 / 10.0 MB (77%)")
        self.assertEqual(setup.format_download_progress(32_110_000_000, 49_840_000_000), "32.11 / 49.84 GB (64%)")

    def test_boundaries_and_unknown_size(self):
        for n, expected in [(0, "0 B"), (999, "999 B"), (1000, "1.0 KB"),
                            (1_000_000, "1.0 MB"), (1_000_000_000, "1.00 GB")]:
            with self.subTest(n=n):
                self.assertEqual(setup.format_size(n), expected)
                self.assertEqual(setup.format_download_progress(n), expected)
        self.assertEqual(setup.format_download_progress(770, 1000), "0.8 / 1.0 KB (77%)")
        self.assertEqual(setup.format_download_progress(77, 100), "77 / 100 B (77%)")
        self.assertEqual(setup.format_download_progress(999_000, 1_000_000), "1.0 / 1.0 MB (100%)")


if __name__ == "__main__":
    unittest.main()
