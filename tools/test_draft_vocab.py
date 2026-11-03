"""Tests for tools/draft_vocab.py's corpus coverage mode (#597): the most frequent tokens of a corpus that together cover
a given share of its occurrences; the DFlash head's options (one share a file, JSON lines, punct / caps / single,
--size and --multiple).  No model, no GPU.

    python -m unittest tools.test_draft_vocab
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import draft_vocab as DV  # noqa: E402


class Coverage(unittest.TestCase):
    def test_most_frequent_first_until_the_share_is_covered(self):
        c = Counter({7: 50, 3: 30, 9: 15, 4: 5})              # 100 occurrences
        self.assertEqual(DV.covering(c, 0.5), [7])
        self.assertEqual(DV.covering(c, 0.8), [7, 3])
        self.assertEqual(DV.covering(c, 0.81), [7, 3, 9])
        self.assertEqual(DV.covering(c, 1.0), [7, 3, 9, 4])

    def test_ties_take_the_lower_id_first(self):
        self.assertEqual(DV.covering(Counter({5: 10, 2: 10, 8: 10}), 0.5), [2, 5])

    def test_share_outside_a_subset(self):
        c = Counter({1: 60, 2: 30, 3: 10})
        self.assertAlmostEqual(DV.outside(c, [1]), 0.4)
        self.assertAlmostEqual(DV.outside(c, [1, 2, 3]), 0.0)
        self.assertEqual(DV.outside(Counter(), [1]), 0.0)

    def test_corpus_counts_reads_every_file(self):
        class Tok:                                            # one id per character
            def encode(self, text):
                return [ord(ch) for ch in text]
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d) / "a.txt", Path(d) / "b.txt"
            a.write_text("été", encoding="utf-8")
            b.write_text("eté", encoding="utf-8")
            self.assertEqual(DV.corpus_counts(Tok(), [a, b]), Counter({ord("é"): 3, ord("t"): 2, ord("e"): 1}))


class TokenClasses(unittest.TestCase):
    class Tok:                                                # token_types: 1 normal, 3 control, 4 user-defined
        token_types = [1, 1, 1, 3, 4, 1]

    def test_classes(self):
        tok = self.Tok()
        self.assertEqual(DV.classes_of(tok, 0, "123"), {"digits"})
        self.assertEqual(DV.classes_of(tok, 1, " 12"), set())       # a space and digits: an ordinary token
        self.assertEqual(DV.classes_of(tok, 2, None), {"bytes"})   # a piece of a character
        self.assertEqual(DV.classes_of(tok, 3, "<|user|>"), {"special"})
        self.assertEqual(DV.classes_of(tok, 4, "</think>"), {"special"})
        self.assertEqual(DV.classes_of(tok, 5, "abc"), set())

    def test_punctuation_capitals_and_single_bytes(self):
        class Tok:
            token_types = [1] * 8
            data = [b" */", b"));", b" VARCHAR", b"MAX_SIZE", b"A", b"!", b"\xe4", b" Var"]

            def token_bytes(self, i):
                return self.data[i]
        tok = Tok()
        kinds = [DV.classes_of(tok, i, DV.token_text(tok, i)) for i in range(8)]
        self.assertEqual(kinds, [{"punct"}, {"punct"}, {"caps"}, {"caps"}, {"single"}, {"single", "punct"},
                                 {"single", "bytes"}, set()])


class TokenizerDir(unittest.TestCase):
    def test_reads_a_pack_tokenizer_export(self):
        import json
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "vocab.json").write_text(json.dumps({"a": 0, "b": 1, "ab": 2, "<|user|>": 3}), encoding="utf-8")
            (p / "merges.txt").write_text("a b\n", encoding="utf-8")
            (p / "token_type.json").write_text(json.dumps([1, 1, 1, 3]), encoding="utf-8")
            (p / "tokenizer.json").write_text(json.dumps({"pre": "glm4", "special_ids": {"x": 3}}), encoding="utf-8")
            tok = DV.load_tokenizer_dir(p)
            self.assertEqual(tok.pre, "glm4")
            self.assertEqual(tok.encode("ab"), [2])
            self.assertEqual(DV.classes_of(tok, 3, DV.token_text(tok, 3)), {"special"})


class CorpusFiles(unittest.TestCase):
    def test_json_lines_fields_and_chat_messages(self):
        import json
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "c.jsonl"
            rows = [{"problem": "p1", "solution": "s1", "other": "x"},
                    {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]}]
            f.write_text("\n".join(json.dumps(r) for r in rows) + "\nnot json\n", encoding="utf-8")
            self.assertEqual(list(DV.corpus_texts(f, ["problem", "solution", "messages"])), ["p1", "s1", "hi", "yo"])
            self.assertEqual(list(DV.corpus_texts(f, ["problem", "solution", "messages"], max_chars=3)), ["p1", "s1"])

    def test_parquet_columns(self):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest("no pyarrow")
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "t.parquet"
            pq.write_table(pa.table({"text": ["a b", "c"], "id": [1, 2]}), f)
            self.assertEqual(list(DV.corpus_texts(f, ["text", "missing"])), ["a b", "c"])

    def test_mean_share_ranking(self):
        a = Counter({1: 900, 2: 100})                         # shares 0.9, 0.1
        b = Counter({3: 2, 4: 1})                             # shares 2/3, 1/3: a small file weighs as much
        self.assertEqual(DV.ranked([a, b]), [1, 3, 4, 2])


class WholeSubsets(unittest.TestCase):
    """main() on a tokenizer of one id a character: a share a file, then --size and --multiple from the ranking"""
    def run_main(self, d, *args):
        import contextlib
        import io
        import json
        p = Path(d)
        vocab = {ch: i for i, ch in enumerate("abcdefgh")}
        (p / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
        (p / "merges.txt").write_text("", encoding="utf-8")
        (p / "tokenizer.json").write_text(json.dumps({"pre": "glm4"}), encoding="utf-8")
        (p / "one.txt").write_text("aaaaaaab", encoding="utf-8")       # a 7/8, b 1/8
        (p / "two.txt").write_text("ccccdddde", encoding="utf-8")      # c 4/9, d 4/9, e 1/9
        out = p / "out.bin"
        argv = ["draft_vocab.py", "--tokenizer", str(p), "--per-file", "--corpus", str(p / "one.txt"),
                str(p / "two.txt"), *args, "--out", str(out)]
        with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(DV.main(), 0)
        from array import array
        return sorted(array("i", out.read_bytes()))

    def test_a_share_a_file(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(self.run_main(d, "--coverage", "0.8", "0.4"), [0, 2])         # a; c
            self.assertEqual(self.run_main(d, "--coverage", "0.8", "0.9"), [0, 2, 3, 4])   # a; c, d, e
            self.assertEqual(self.run_main(d, "--coverage", "0.9"), [0, 1, 2, 3, 4])       # a, b; c, d, e

    def test_size_and_multiple(self):
        with tempfile.TemporaryDirectory() as d:
            # mean shares: a 7/16, c and d 2/9 (ties: the lower id), b 1/16, e 1/18
            self.assertEqual(self.run_main(d, "--coverage", "0.8", "0.4", "--size", "4"), [0, 1, 2, 3])
            self.assertEqual(self.run_main(d, "--coverage", "0.8", "0.4", "--multiple", "3"), [0, 2, 3])
            self.assertEqual(self.run_main(d, "--coverage", "0.8", "0.9", "--multiple", "4"), [0, 2, 3, 4])


if __name__ == "__main__":
    unittest.main()
