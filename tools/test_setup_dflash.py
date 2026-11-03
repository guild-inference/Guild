"""The DFlash 2 drafter in setup (no GPU, no downloads): fetched and written into the run config (--dflash, windows of
8) on one NVIDIA card with an engine that has --dflash; the MTP layer alone with --no-dflash, an older engine, or
several GPUs.

    python -m unittest tools.test_setup_dflash
"""
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import setup  # noqa: E402
from tools.test_setup_golden import card, install  # noqa: E402


class DFlashDrafter(unittest.TestCase):
    def install(self, flags=(), cards=None, dflash_engine=True):
        def supports(engine, option, env=None):
            return option != "--dflash" or dflash_engine
        fetched = []

        def fake_download(url, dst, what=None):
            fetched.append((url, dst))
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(b"")
            setup.mark(dst)
        code, out, cfg, _ = install(127.8, cards or [card(0, "NVIDIA GeForce RTX 4090", 24.0, "89")],
                                    ["--family", "qwen", "--model", "IQ3_S", "--no-start", *flags],
                                    extra=[mock.patch.object(setup, "engine_supports_option", side_effect=supports),
                                           mock.patch.object(setup, "download", fake_download)])
        return code, out, cfg, fetched

    def test_one_card_drafts_with_dflash(self):
        code, out, cfg, fetched = self.install()
        self.assertEqual(code, 0, out)
        args = cfg["args"]
        self.assertIn("--dflash", args)
        self.assertTrue(args[args.index("--dflash") + 1].replace("\\", "/").endswith("/mtp/dflash"))
        self.assertEqual(args[args.index("--spec") + 1], "8")
        self.assertIn("--mtp", args)                                  # still there: the engine falls back to it
        names = sorted(dst.name for url, dst in fetched if "DFlash2" in url)
        self.assertEqual(names, sorted(setup.DFLASH_FILES))
        for url, dst in fetched:
            if "DFlash2" in url:
                self.assertIn(setup.HF_REVISIONS[setup.DFLASH_REPO], url)   # the pinned revision
        self.assertIn("DFlash drafter:", out)

    def test_no_dflash_keeps_the_mtp_layer(self):
        code, out, cfg, fetched = self.install(["--no-dflash"])
        self.assertEqual(code, 0, out)
        self.assertNotIn("--dflash", cfg["args"])
        self.assertEqual(cfg["args"][cfg["args"].index("--spec") + 1], "4")
        self.assertFalse(any("DFlash2" in url for url, _ in fetched))

    def test_an_engine_without_dflash(self):
        code, out, cfg, fetched = self.install(dflash_engine=False)
        self.assertEqual(code, 0, out)
        self.assertNotIn("--dflash", cfg["args"])
        self.assertEqual(cfg["args"][cfg["args"].index("--spec") + 1], "4")
        self.assertIn("no --dflash yet", out)

    def test_two_gpus_keep_the_mtp_layer(self):
        code, out, cfg, fetched = self.install(["--gpus", "0,1"],
                                               cards=[card(i, "NVIDIA GeForce RTX 3090", 24.0, "86") for i in range(2)])
        self.assertEqual(code, 0, out)
        self.assertEqual(cfg["gpu"], [0, 1])
        self.assertNotIn("--dflash", cfg["args"])
        self.assertIn("across several GPUs", out)

    def test_pinned_files(self):
        self.assertEqual(set(setup.DFLASH_FILES), {"config.json", "model.safetensors"})
        for size, sha in setup.DFLASH_FILES.values():
            self.assertGreater(size, 0)
            self.assertEqual(len(sha), 64)


if __name__ == "__main__":
    unittest.main()
