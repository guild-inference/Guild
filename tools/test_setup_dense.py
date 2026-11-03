"""Qwen's dense matrices in setup (no GPU, no downloads): Unsloth's UD-IQ4_XS on an NVIDIA card is written with
--dense q5_k --hc q8_0 (its dense and hyper-connection matrices are all Q8_0); the GSQ-RCO quants and an engine
without the options keep the GGUF's formats.

    python -m unittest tools.test_setup_dense
"""
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import setup  # noqa: E402
from tools.test_setup_golden import card, install  # noqa: E402


class DenseFormat(unittest.TestCase):
    def install(self, family, model, engine_has=True):
        def supports(engine, option, env=None):
            return engine_has or option not in ("--dense", "--hc")
        code, out, cfg, _ = install(165.0, [card(0, "NVIDIA GeForce RTX 4090", 24.0, "89")],
                                    ["--family", family, "--model", model, "--context", "32768", "--no-start"],
                                    extra=[mock.patch.object(setup, "engine_supports_option", side_effect=supports),
                                           # the harness's engine (0.1.32) as new enough for UD-IQ4_XS
                                           mock.patch.dict(setup.MODELS["UD-IQ4_XS"], {"engine": (0, 1, 32)})])
        self.assertEqual(code, 0, out)
        return cfg["args"]

    def test_ud_iq4_xs_takes_q5_k_and_the_q8_0_hyper_connection(self):
        args = self.install("unsloth", "UD-IQ4_XS")
        self.assertEqual(args[args.index("--dense") + 1], "q5_k")
        self.assertEqual(args[args.index("--hc") + 1], "q8_0")

    def test_gsq_rco_keeps_its_formats(self):
        args = self.install("qwen", "IQ3_S")
        self.assertNotIn("--dense", args)
        self.assertNotIn("--hc", args)

    def test_an_engine_without_the_options(self):
        args = self.install("unsloth", "UD-IQ4_XS", engine_has=False)
        self.assertNotIn("--dense", args)
        self.assertNotIn("--hc", args)


if __name__ == "__main__":
    unittest.main()
