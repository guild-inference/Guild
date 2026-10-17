#!/usr/bin/env python3
import json
from pathlib import Path
import tempfile
import unittest
from release_version import ReleaseVersion, source_metadata


class ReleasePolicy(unittest.TestCase):
    def test_revision_is_explicit_order(self):
        versions=["v0.2.0", "v0.2.0+r1", "v0.2.0+r2", "v0.2.0+r10", "v0.2.1", "v0.3.0"]
        self.assertEqual(sorted(ReleaseVersion.from_tag(v) for v in reversed(versions)),
                         [ReleaseVersion.from_tag(v) for v in versions])
        self.assertEqual(ReleaseVersion.from_tag("v0.2.0+r2").display,"v0.2.0, r2")

    def test_software_precedes_revision(self):
        self.assertLess(ReleaseVersion.from_tag("v0.2.0+r99"),ReleaseVersion.from_tag("v0.2.1"))
        self.assertLess(ReleaseVersion.from_tag("v0.2.0-rc.10+r99"),ReleaseVersion.from_tag("v0.2.0"))
        self.assertLess(ReleaseVersion.from_tag("v0.2.0-rc.2"),ReleaseVersion.from_tag("v0.2.0-rc.10"))

    def test_invalid_tags(self):
        for value in ("0.2.0", "v00.2.0", "v0.2", "v0.2.0+r01", "v0.2.0+rebuild",
                      "v0.2.0-rc.01", "v0.2.0+r-1", "v0.2.0+g123", "v0.2.0+r0"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ReleaseVersion.from_tag(value)

    def test_source_and_tag_agree(self):
        with tempfile.TemporaryDirectory() as work:
            path=Path(work)/"VERSION.json"
            path.write_text(json.dumps({"version":"0.2.0","revision":0,"prerelease":"rc.1","channel":"preview"}))
            result=source_metadata(path,"v0.2.0-rc.1+r2")
            self.assertEqual(result["revision"],2)
            self.assertEqual(result["channel"],"preview")
            self.assertEqual(result["software_version"],"0.2.0-rc.1")
            with self.assertRaises(ValueError):source_metadata(path,"v0.2.0")
            with self.assertRaises(ValueError):source_metadata(path,"v0.2.1-rc.1")


if __name__=="__main__":unittest.main()
