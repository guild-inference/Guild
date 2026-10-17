#!/usr/bin/env python3
"""Check byte reproducibility, permissions and long filenames in release archives."""
import os
from pathlib import Path
import tarfile
import tempfile
import unittest
from package_linux_cuda import write_archive


class Archive(unittest.TestCase):
    def test_archive_ignores_source_times_and_output_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            source = work / "guild-linux"
            source.mkdir()
            executable = source / "guild"
            executable.write_text("#!/bin/sh\nexit 0\n")
            executable.chmod(0o755)
            long_file = source / ("x" * 130)
            long_file.write_bytes(b"long-path payload")
            first, second = work / "first.tar.gz", work / "second.tar.gz"
            write_archive(source, first, 1234567890)
            os.utime(executable, (1600000000, 1600000000))
            os.utime(long_file, (1700000000, 1700000000))
            write_archive(source, second, 1234567890)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            with tarfile.open(first) as archive:
                self.assertEqual(archive.getmember("guild-linux/guild").mode, 0o755)
                self.assertEqual(archive.extractfile("guild-linux/" + long_file.name).read(), b"long-path payload")
                self.assertTrue(all(m.mtime == 1234567890 and m.uid == m.gid == 0 for m in archive))


if __name__ == "__main__":
    unittest.main()
