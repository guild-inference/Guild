#!/usr/bin/env python3
import unittest
from release_version import ReleaseVersion
from validate_release import validate_revision_changes,validate_v1_gate


class Validation(unittest.TestCase):
    def test_revision_packaging_only(self):
        validate_revision_changes(["packaging/manifest.json",".github/workflows/release.yml"])
        for path in ("src/runtime/model.cpp","include/guild/abi/kernel_abi.h","serve/app.py","CMakeLists.txt","setup.py"):
            with self.subTest(path=path),self.assertRaises(ValueError):validate_revision_changes([path])

    def test_v1_requires_real_maturity(self):
        validate_v1_gate(ReleaseVersion("0.99.0"),{})
        with self.assertRaises(ValueError):validate_v1_gate(ReleaseVersion("1.0.0"),{})
        gates={key:True for key in ("stable_interfaces","validated_package_ecosystem","mature_gpack","mature_gexec",
                                   "supported_hardware_inference","reproducible_releases","compatibility_guarantees","regression_coverage")}
        gates["validated_architecture_families"]=["qwen","qwen","qwen","qwen","qwen"]
        with self.assertRaises(ValueError):validate_v1_gate(ReleaseVersion("1.0.0"),gates)
        gates["validated_architecture_families"]=["a","b","c","d","e"]
        validate_v1_gate(ReleaseVersion("1.0.0"),gates)


if __name__=="__main__":unittest.main()
