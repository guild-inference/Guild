"""The draft head's token subset (rt/draft_vocab.bin): the vocabulary rows the MTP draft layer may propose.

The engine reads rt/draft_vocab.bin (int32 token ids) when the draft layer binds and computes the draft logits over
those rows only: a smaller head is faster and takes less VRAM, but a token outside the subset can never be drafted.
The shipped subset was built from English and code and had 27 of the vocabulary's 55,328 Han tokens, so an answer in
Chinese drafted almost nothing (#137).  This adds whole scripts to a subset:

    python tools/draft_vocab.py --gguf <model>-00001-of-0000N.gguf --base data/draft_vocab.bin --add cjk \
        --out data/draft_vocab.bin

--add takes han, kana, hangul, cjk_punct, cjk (those four) or cyrillic.  The base ids keep their order; the added ones follow in
id order.  --stats prints what a subset holds.

A language written in a script the base already holds (French, Spanish, German... in Latin letters) needs specific
words and word pieces, not a whole script.  --corpus adds the tokens of a text corpus in that language: the most
frequent ones that together cover --coverage (default 0.99) of the corpus's token occurrences (#597):

    python tools/draft_vocab.py --gguf <model>-00001-of-0000N.gguf --base data/draft_vocab_en.bin \
        --corpus fr1.txt fr2.txt ... --coverage 0.99 --out data/draft_vocab_fr.bin

It prints the share of the corpus's occurrences outside the base and outside the new subset.  --add and --corpus
can be combined.

GLM-5.3-Flash drafts with its NextN block over the same kind of subset (draft_vocab.bin in the model's tokenizer
directory, glm5_run --draft-vocab, or GUILD_GLM_DRAFT_VOCAB).  Its tokenizer is the pack's tokenizer/ export, which
--tokenizer reads in place of --gguf.  GLM's vocabulary has no subset to extend, so the subset is built whole: the
tokens covering 99.9% of each of several corpora of English prose and code (--per-file: each file a kind of text), the
whole CJK and Cyrillic scripts, every token of digits, the control tokens (special) and the pieces of characters
(bytes: what a rare character or an emoji is spelled with):

    python tools/draft_vocab.py --tokenizer <GLM pack>/tokenizer --corpus code_and_docs.txt wiki.txt books.txt \
        --per-file --coverage 0.999 --add cjk,cyrillic,digits,special,bytes --out draft_vocab.bin

The DFlash 2 drafter's head (--dflash-vocab) takes such a subset too, built whole from Qwen3.8-Flash-Next's
vocabulary (bench/results/2026-11-03-decode-round9).  data/dflash_vocab.bin covers each kind of text to its own
share (--coverage, one a file, with --per-file): English chat, encyclopedic English, code, multilingual chat and
Chinese, Japanese, Korean and Russian encyclopedic text; punct (every token of ASCII punctuation), caps (every
all-caps ASCII word: SQL's keywords, constants), single (the 256 one-byte tokens) and special are added, and
--multiple 2 pads the count with the next most frequent tokens (an even count of rows takes the head kernel's
two-rows-a-task path).  --fields names the texts of JSON lines (.jsonl) and Parquet (.parquet) files, --max-chars
caps what is read of each file and --jobs tokenizes in several processes:

    python tools/draft_vocab.py --gguf <Qwen3.8-Flash-Next>-00001-of-0000N.gguf --per-file --jobs 32 \
        --max-chars 40000000 --fields messages,text,problem,solution,query,answer,inputs,targets \
        --corpus ultrachat.parquet wikitext.parquet magicoder.jsonl codefeedback.jsonl aya.parquet zh.parquet \
            ja.parquet ko.parquet ru.parquet \
        --coverage 0.9995 0.999 0.9995 0.9995 0.99 0.99 0.98 0.98 0.99 \
        --add special,single,punct,caps --multiple 2 --out data/dflash_vocab.bin

data/dflash_vocab_en.bin takes English and code only, filled to a size (--size: the most frequent tokens by their
mean share of the files' occurrences):

    python tools/draft_vocab.py --gguf <Qwen3.8-Flash-Next>-00001-of-0000N.gguf --per-file --jobs 32 \
        --max-chars 40000000 --fields messages,text,problem,solution,query,answer \
        --corpus ultrachat.parquet wikitext.parquet magicoder.jsonl codefeedback.jsonl --coverage 0.5 \
        --add special,single,punct,caps --size 65536 --out data/dflash_vocab_en.bin

The files, all from the Hugging Face Hub: HuggingFaceH4/ultrachat_200k data/train_sft-00000-of-00003-*.parquet
(8049631), Salesforce/wikitext wikitext-103-raw-v1/train-00000-of-00002.parquet (b08601e),
ise-uiuc/Magicoder-OSS-Instruct-75K data-oss_instruct-decontaminated.jsonl (5f839b1),
m-a-p/CodeFeedback-Filtered-Instruction CodeFeedback-Filtered-Instruction.jsonl (a08c213), CohereForAI/aya_dataset
data/train-00000-of-00001.parquet (f9ea045) and wikimedia/wikipedia (b04c8d1) 20231101.zh/train-00002-of-00006,
20231101.ja/train-00014-of-00015, 20231101.ko/train-00002-of-00003 and 20231101.ru/train-00007-of-00021.
"""
from __future__ import annotations

import argparse
import json
import re
import string
import sys
import unicodedata
from array import array
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import guild_tokenizer as ST  # noqa: E402

SCRIPTS = {
    "han": [(0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF), (0x20000, 0x2FA1F)],
    "kana": [(0x3040, 0x30FF), (0x31F0, 0x31FF), (0xFF66, 0xFF9F)],
    "hangul": [(0x1100, 0x11FF), (0x3130, 0x318F), (0xAC00, 0xD7AF)],
    "cjk_punct": [(0x3000, 0x303F), (0xFF00, 0xFF65), (0xFFA0, 0xFFEF)],
    # Ukrainian, Russian, Bulgarian, Serbian...: the shipped subset held 142 of the vocabulary's 18,580 Cyrillic tokens
    "cyrillic": [(0x0400, 0x04FF), (0x0500, 0x052F), (0x1C80, 0x1C8F), (0x2DE0, 0x2DFF), (0xA640, 0xA69F)],
}
GROUPS = {"cjk": ["han", "kana", "hangul", "cjk_punct"]}
# token classes --add also takes: digits (every character a number), special (the control and user-defined tokens:
# <|user|>, </think>, <tool_call>...), bytes (pieces of a character: not whole UTF-8 characters, not special), single
# (one byte: what any text can be spelled with), punct (ASCII punctuation, after any leading spaces: " */", "));") and
# caps (an all-caps ASCII word of two or more characters: " VARCHAR", "MAX_SIZE")
CLASSES = ["digits", "special", "bytes", "single", "punct", "caps"]
CAPS = re.compile(r"[A-Z][A-Z0-9_]+")


def scripts_of(text: str) -> set[str]:
    found = set()
    for ch in text:
        c = ord(ch)
        for name, ranges in SCRIPTS.items():
            if any(a <= c <= b for a, b in ranges):
                found.add(name)
    return found


def classes_of(tok, i: int, text: str | None) -> set[str]:
    special = bool(tok.token_types) and tok.token_types[i] in (3, 4)
    if special:
        return {"special"}
    found = {"single"} if one_byte(tok, i) else set()
    if text is None:
        return found | {"bytes"}
    if text and all(unicodedata.category(ch).startswith("N") for ch in text):
        found.add("digits")
    t = text.lstrip(" ")
    if t and all(ch in string.punctuation for ch in t):
        found.add("punct")
    if CAPS.fullmatch(t):
        found.add("caps")
    return found


def one_byte(tok, i: int) -> bool:
    try:
        return len(tok.token_bytes(i)) == 1
    except (AttributeError, KeyError, IndexError):
        return False


def load_tokenizer_dir(path) -> ST.Tokenizer:
    """A pack's tokenizer/ export (guild_tokenizer.extract): vocab.json, merges.txt, token_type.json and
    tokenizer.json (its pre-tokenizer)."""
    p = Path(path)
    cfg = json.loads((p / "tokenizer.json").read_text(encoding="utf-8"))
    vocab = json.loads((p / "vocab.json").read_text(encoding="utf-8"))
    tokens = [None] * len(vocab)
    for t, i in vocab.items():
        tokens[i] = t
    if any(t is None for t in tokens):
        raise ValueError(f"{p / 'vocab.json'}: the ids are not 0 .. {len(vocab) - 1}")
    merges = [m for m in (p / "merges.txt").read_text(encoding="utf-8").split("\n") if m]
    types = json.loads((p / "token_type.json").read_text(encoding="utf-8")) if (p / "token_type.json").exists() else None
    special = {k: int(v) for k, v in (cfg.get("special_ids") or {}).items()}
    return ST.Tokenizer(tokens, merges, types, cfg.get("pre", "qwen35"), special)


def token_text(tok, i: int) -> str | None:
    """A token's text, or None when its bytes are not whole UTF-8 characters (a piece of one) or it is special."""
    try:
        return tok.token_bytes(i).decode("utf-8")
    except (KeyError, UnicodeDecodeError, IndexError):
        return None


def strings(v):
    """A field's texts: a string, or a list of strings or of chat messages ({"content": ...})."""
    if isinstance(v, str):
        yield v
    elif isinstance(v, list):
        for m in v:
            if isinstance(m, str):
                yield m
            elif isinstance(m, dict) and isinstance(m.get("content"), str):
                yield m["content"]


def corpus_texts(path, fields=(), max_chars: int = 0):
    """A corpus file's texts: UTF-8 text in pieces of whole lines (about 1 MiB, so a large file does not have to be
    held as one string's tokens), or the `fields` of JSON lines (.jsonl) or the columns of a Parquet table (.parquet,
    with pyarrow); up to `max_chars` characters (0: the whole file)."""
    path = Path(path)
    read = 0

    def take(text):
        nonlocal read
        read += len(text)
        return text

    if path.suffix == ".jsonl":
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                for k in fields:
                    for t in strings(d.get(k) if isinstance(d, dict) else None):
                        yield take(t)
                if max_chars and read >= max_chars:
                    return
    elif path.suffix == ".parquet":
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        cols = [c for c in fields if c in pf.schema_arrow.names]
        for batch in pf.iter_batches(columns=cols, batch_size=256):
            for row in batch.to_pylist():
                for c in cols:
                    for t in strings(row[c]):
                        yield take(t)
                if max_chars and read >= max_chars:
                    return
    else:
        with open(path, encoding="utf-8") as f:
            piece: list[str] = []
            size = 0
            for line in f:
                piece.append(line)
                size += len(line)
                if size >= 1 << 20:
                    yield take("".join(piece))
                    piece, size = [], 0
                    if max_chars and read >= max_chars:
                        return
            if piece:
                yield take("".join(piece))


_worker_tok = None


def _load(spec):
    kind, path = spec
    return ST.Tokenizer.from_gguf(path) if kind == "gguf" else load_tokenizer_dir(path)


def _init_worker(spec):
    global _worker_tok
    _worker_tok = _load(spec)


def _count(texts) -> Counter:
    c: Counter = Counter()
    for t in texts:
        c.update(_worker_tok.encode(t))
    return c


def batches(texts, size: int = 200_000):
    b, n = [], 0
    for t in texts:
        b.append(t)
        n += len(t)
        if n >= size:
            yield b
            b, n = [], 0
    if b:
        yield b


def corpus_counts(tok, paths, fields=(), max_chars: int = 0, jobs: int = 1, spec=None) -> Counter:
    """Token occurrences over the corpus files (tokenized as plain text: no special tokens); with jobs > 1, in that
    many processes, each loading the tokenizer from `spec` (("gguf", path) or ("dir", path))."""
    counts: Counter = Counter()
    if jobs > 1 and spec is not None:
        import multiprocessing as mp
        with mp.Pool(jobs, initializer=_init_worker, initargs=(spec,)) as pool:
            for path in paths:
                for c in pool.imap_unordered(_count, batches(corpus_texts(path, fields, max_chars))):
                    counts.update(c)
        return counts
    for path in paths:
        for t in corpus_texts(path, fields, max_chars):
            counts.update(tok.encode(t))
    return counts


def covering(counts: Counter, coverage: float) -> list[int]:
    """The most frequent ids (ties: the lower id first) that together cover `coverage` of the occurrences."""
    total = sum(counts.values())
    picked, covered = [], 0
    for i, c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        if covered >= coverage * total:
            break
        picked.append(i)
        covered += c
    return picked


def ranked(per_file: list[Counter]) -> list[int]:
    """Every id the files hold, by its mean share of each file's occurrences (the most first; ties: the lower id)."""
    score: dict[int, float] = {}
    for c in per_file:
        total = sum(c.values())
        for i, k in c.items():
            score[i] = score.get(i, 0.0) + k / total
    return [i for i, _ in sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))]


def outside(counts: Counter, ids) -> float:
    """The share of the occurrences whose token is not in `ids`."""
    total = sum(counts.values())
    have = set(ids)
    return sum(c for i, c in counts.items() if i not in have) / total if total else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--gguf", help="the model's (first) GGUF file: its vocabulary")
    src.add_argument("--tokenizer", help="a pack's tokenizer/ directory (vocab.json, merges.txt, ...) instead")
    ap.add_argument("--base", help="the subset to extend (int32 ids)")
    ap.add_argument("--add", default="",
                    help="comma list: han, kana, hangul, cjk_punct, cjk, cyrillic, digits, special, bytes, single, "
                         "punct, caps")
    ap.add_argument("--corpus", nargs="+", default=[],
                    help="UTF-8 text files in the language to add (#597); .jsonl and .parquet files with --fields")
    ap.add_argument("--coverage", type=float, nargs="+", default=[0.99],
                    help="with --corpus: the share of its token occurrences the added tokens cover (default 0.99); "
                         "with --per-file, one share for every file or one a file")
    ap.add_argument("--per-file", action="store_true",
                    help="with --corpus: cover --coverage of each file's occurrences (each file a kind of text), "
                         "not of all of them together")
    ap.add_argument("--fields", default="",
                    help="comma list: the keys (.jsonl) or columns (.parquet) whose texts the corpus files hold")
    ap.add_argument("--max-chars", type=int, default=0, help="read up to this many characters of each corpus file")
    ap.add_argument("--jobs", type=int, default=1, help="tokenize the corpus in this many processes")
    ap.add_argument("--size", type=int, default=0,
                    help="fill the subset up to this many ids with the corpus's most frequent other tokens")
    ap.add_argument("--multiple", type=int, default=1,
                    help="pad the subset to a multiple of this many ids with the corpus's most frequent other tokens")
    ap.add_argument("--out", help="where to write the new subset")
    ap.add_argument("--stats", action="store_true", help="print what the base (and the new) subset holds")
    a = ap.parse_args()

    tok = ST.Tokenizer.from_gguf(a.gguf) if a.gguf else load_tokenizer_dir(a.tokenizer)
    n = len(tok.tokens)
    kinds = [(scripts_of(t) if t is not None else set()) | classes_of(tok, i, t)
             for i, t in ((i, token_text(tok, i)) for i in range(n))]
    base = list(array("i", Path(a.base).read_bytes())) if a.base else []
    if any(i < 0 or i >= n for i in base):
        sys.exit(f"{a.base} has ids outside the vocabulary ({n})")

    want = set()
    for x in filter(None, (s.strip() for s in a.add.split(","))):
        names = GROUPS.get(x, [x])
        if any(nm not in SCRIPTS and nm not in CLASSES for nm in names):
            sys.exit(f"--add {x}: expected one of {', '.join([*SCRIPTS, *GROUPS, *CLASSES])}")
        want.update(names)
    if any(not 0 < c <= 1 for c in a.coverage):
        sys.exit("--coverage: expected shares in (0, 1]")
    if len(a.coverage) > 1 and not (a.per_file and len(a.coverage) == len(a.corpus)):
        sys.exit("--coverage: one share, or with --per-file one a corpus file")
    if (a.size or a.multiple > 1) and not a.corpus:
        sys.exit("--size and --multiple take their tokens from --corpus")
    fields = [f.strip() for f in a.fields.split(",") if f.strip()]
    spec = ("gguf", a.gguf) if a.gguf else ("dir", a.tokenizer)

    def count(paths):
        return corpus_counts(tok, paths, fields, a.max_chars, a.jobs, spec)

    per_file = [count([f]) for f in a.corpus] if a.per_file else []
    counts = sum(per_file, Counter()) if a.per_file else count(a.corpus) if a.corpus else Counter()
    shares = a.coverage * len(a.corpus) if len(a.coverage) == 1 else a.coverage
    if a.per_file:
        from_corpus = set().union(*(covering(c, sh) for c, sh in zip(per_file, shares)))
    else:
        from_corpus = set(covering(counts, a.coverage[0])) if counts else set()
    have = set(base)
    added = [i for i in range(n) if i not in have and (kinds[i] & want or i in from_corpus)]
    ids = base + added
    if a.size or a.multiple > 1:  # the most frequent other tokens, up to --size, then to a multiple of --multiple
        taken = set(ids)
        rest = (i for i in ranked(per_file if a.per_file else [counts]) if i not in taken)
        fill = []
        while len(ids) + len(fill) < a.size or (len(ids) + len(fill)) % a.multiple:
            i = next(rest, None)
            if i is None:
                break
            fill.append(i)
        added += fill
        ids += fill

    def stats(label, sel):
        counts = {nm: sum(1 for i in sel if nm in kinds[i]) for nm in [*SCRIPTS, *CLASSES]}
        full = {nm: sum(1 for k in kinds if nm in k) for nm in [*SCRIPTS, *CLASSES]}
        print(f"{label}: {len(sel)} ids; " + ", ".join(f"{nm} {counts[nm]}/{full[nm]}" for nm in counts))

    if a.stats or not a.out:
        if base:
            stats("base", base)
        if added:
            stats("new", ids)
    if counts:
        print(f"corpus: {sum(counts.values())} token occurrences, {len(counts)} distinct; "
              f"{len(from_corpus)} tokens cover {'/'.join(f'{c:g}' for c in a.coverage)}, "
              f"{len(from_corpus - have)} of them not in the base")
        for f, c in zip(a.corpus, per_file):
            print(f"  {Path(f).name}: {sum(c.values())} occurrences, outside the new subset {outside(c, ids):.3%}")
        print(f"corpus occurrences outside the base: {outside(counts, base):.2%}, outside the new subset: "
              f"{outside(counts, ids):.2%}")
    if a.out:
        Path(a.out).write_bytes(array("i", ids).tobytes())
        print(f"wrote {a.out}: {len(ids)} ids ({len(added)} added)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
