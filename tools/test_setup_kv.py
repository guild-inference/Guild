"""KV formats, long FP16 contexts and RAM-heavy small-card setup (no GPU/downloads)."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import setup
from tools.test_setup_golden import card, install


class KvFormats(unittest.TestCase):
    def test_labels_and_layout_bytes(self):
        expected = {"fp16": ("FP16", 2048, 2048), "int8": ("8-bit", 1056, 1056),
                    "q4_0": ("4-bit (Hadamard-rotated)", 576, 576),
                    "k8v4": ("INT8 K + 4-bit V", 816, 1056)}
        for kv, (label, main, draft) in expected.items():
            with self.subTest(kv=kv):
                self.assertEqual(setup.kv_format_label(kv), label)
                self.assertEqual(setup.kv_bytes_per_token(kv), 12 * main + draft)
                self.assertEqual(setup.kv_size_gb(262144, kv), 262144 * (12 * main + draft) / 1e9)
        self.assertEqual(setup.kv_size_gb(262144, "fp16") * 1e9, 6.5 * 1024**3)
        with self.assertRaises(KeyError):
            setup.kv_format_label("unknown")

    def test_ram_warning_uses_selected_format(self):
        fp16 = setup.ctx_ram_need("IQ3_S", 262144, kv="fp16")
        int8 = setup.ctx_ram_need("IQ3_S", 262144, kv="int8")
        self.assertAlmostEqual(fp16 - int8, 262144 * 13 * (2048 - 1056) / 1e9)

    def test_vram_pool_estimates(self):
        self.assertEqual(setup.main_kv_vram_gb(262144, "fp16") * 1e9, 6 * 1024**3)
        self.assertEqual(setup.main_kv_vram_gb(262144, "fp16", streaming=True) * 1e9, 768 * 1024**2)
        self.assertEqual(setup.main_kv_vram_gb(262144, "fp16", host_only=True) * 1e9, 32.125 * 1024**2)
        self.assertEqual(setup.main_kv_vram_gb(262144, "fp16", host_only=True),
                         setup.main_kv_vram_gb(8192, "fp16", host_only=True))

    def test_budgeted_experts_do_not_require_whole_model_ram(self):
        self.assertLess(setup.kv_host_ram_need_gb("UD-IQ4_XS", 262144, "fp16", 40), 64)
        self.assertGreater(setup.kv_host_ram_need_gb("UD-IQ4_XS", 262144, "fp16", 55), 64)
        self.assertLess(setup.parallel_slot_gb(262144, "fp16", True, host_only=True),
                        setup.parallel_slot_gb(262144, "fp16", True))

    def install(self, flags, vram=24, answers=None, supports=True):
        return install(377, [card(0, "Test GPU", vram, "61")],
                       ["--model", "Q2_0", "--no-start", "--context", "262144", *flags], answers=answers,
                       extra=[mock.patch.object(setup, "engine_supports_option", return_value=supports)])

    def test_explicit_fp16_config_and_start_script(self):
        code, out, cfg, _ = self.install(["--kv", "fp16"])
        self.assertEqual(code, 0, out)
        args = cfg["args"]
        self.assertEqual(args[args.index("--kv") + 1], "fp16")
        self.assertEqual(args[args.index("--max-context") + 1], "262144")
        self.assertIn("FP16", out)
        self.assertIn("7.0 GB", out)
        # Scripts delegate to the config: they must point to the same file and not inject a replacement --kv.
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(setup, "ROOT", Path(tmp)):
            path = Path(tmp) / "strata-q2_0.json"
            path.write_text(json.dumps(cfg))
            script = setup.write_run_script("q2_0", path, 8080, False)
            self.assertIn(str(path), script.read_text())
            self.assertNotIn("--kv int8", script.read_text())
            self.assertEqual(setup.choices_from_config(path)["kv"], "fp16")

    def test_interactive_long_fp16(self):
        code, out, cfg, _ = self.install([], answers={"KV": "3"})
        self.assertEqual(code, 0, out)
        self.assertEqual(cfg["args"][cfg["args"].index("--kv") + 1], "fp16")

    def test_long_default_stays_int8(self):
        code, out, cfg, _ = self.install([])
        self.assertEqual(code, 0, out)
        self.assertEqual(cfg["args"][cfg["args"].index("--kv") + 1], "int8")
        self.assertNotIn("--mtp-optional", cfg["args"])

    def test_eight_gb_plan_preserves_explicit_choices(self):
        code, out, cfg, _ = self.install(["--kv", "fp16", "--draft-vocab", "cjk"], vram=8)
        self.assertEqual(code, 0, out)
        args = cfg["args"]
        self.assertIn("--kv-host-only", args)
        self.assertNotIn("--kv-resident", args)
        self.assertIn("--mtp-optional", args)
        self.assertEqual(args[args.index("--prefill") + 1], "512")
        self.assertEqual(args[args.index("--kv") + 1], "fp16")
        self.assertEqual(args[args.index("--max-context") + 1], "262144")
        self.assertEqual(cfg["draft_vocab"], "cjk")
        self.assertIn("MTP is optional", out)

    def test_old_engine_is_not_given_unsupported_flags(self):
        code, out, cfg, _ = self.install(["--kv", "fp16"], vram=8, supports=False)
        self.assertNotEqual(code, 0)
        self.assertIsNone(cfg)
        self.assertIn("build/update", out)

    def test_no_impossible_full_vram_fp16_plan(self):
        code, out, cfg, _ = self.install(["--kv", "fp16", "--kv-streaming", "off"], vram=8)
        self.assertNotEqual(code, 0)
        self.assertIsNone(cfg)
        self.assertIn("no runnable", out)

    def test_explicit_host_only_and_invalid_hybrid(self):
        code, out, cfg, _ = self.install(["--kv", "fp16", "--kv-host-only"])
        self.assertEqual(code, 0, out)
        self.assertIn("--kv-host-only", cfg["args"])
        code, out, cfg, _ = self.install(["--kv", "k8v4", "--kv-host-only"])
        self.assertNotEqual(code, 0)
        self.assertIsNone(cfg)
        self.assertIn("not supported", out)


if __name__ == "__main__":
    unittest.main()
