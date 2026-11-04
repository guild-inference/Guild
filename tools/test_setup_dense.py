"""Qwen's dense matrices in setup (no GPU, no downloads): Unsloth's UD-IQ4_XS on an NVIDIA card is written with
--dense q4_k --dense-map ssm_out=q5_k,attn_output=q8_0, Unsloth's imatrix and --hc q8_0 (its dense and
hyper-connection matrices are all Q8_0; an engine without the map: --dense q5_k), and on Pascal --prefill-mmq dense;
the GSQ-RCO quants and an engine without the options keep the GGUF's formats.

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

    def test_ud_iq4_xs_takes_the_mixed_map_and_the_q8_0_hyper_connection(self):
        args = self.install("unsloth", "UD-IQ4_XS")
        self.assertEqual(args[args.index("--dense") + 1], "q4_k")
        self.assertEqual(args[args.index("--dense-map") + 1], "ssm_out=q5_k,attn_output=q8_0")
        self.assertTrue(args[args.index("--dense-imatrix") + 1].endswith("imatrix_unsloth.gguf"))
        self.assertEqual(args[args.index("--hc") + 1], "q8_0")
        self.assertNotIn("--prefill-mmq", args)  # an RTX 4090: not measured

    def test_an_engine_without_the_map_keeps_q5_k(self):
        def supports(engine, option, env=None):
            return option not in ("--dense-map", "--dense-imatrix")
        code, out, cfg, _ = install(165.0, [card(0, "NVIDIA GeForce RTX 4090", 24.0, "89")],
                                    ["--family", "unsloth", "--model", "UD-IQ4_XS", "--context", "32768",
                                     "--no-start"],
                                    extra=[mock.patch.object(setup, "engine_supports_option", side_effect=supports),
                                           mock.patch.dict(setup.MODELS["UD-IQ4_XS"], {"engine": (0, 1, 32)})])
        self.assertEqual(code, 0, out)
        args = cfg["args"]
        self.assertEqual(args[args.index("--dense") + 1], "q5_k")
        self.assertNotIn("--dense-map", args)
        self.assertNotIn("--dense-imatrix", args)

    def test_pascal_reads_the_prompt_through_mmq(self):
        code, out, cfg, _ = install(165.0, [card(0, "NVIDIA GeForce GTX 1080 Ti", 11.0, "61")],
                                    ["--family", "unsloth", "--model", "UD-IQ4_XS", "--context", "32768",
                                     "--no-start"],
                                    extra=[mock.patch.object(setup, "engine_supports_option", return_value=True),
                                           mock.patch.dict(setup.MODELS["UD-IQ4_XS"], {"engine": (0, 1, 32)})])
        self.assertEqual(code, 0, out)
        args = cfg["args"]
        self.assertEqual(args[args.index("--prefill-mmq") + 1], "dense")

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
