"""tools/nemotron_draft_vocab.py - the draft head's token subset of Nemotron 3.5 Transcribe (data/nemotron_draft_vocab.bin).

transcribe_run drafts with the MTP block over these rows of the head only (--draft-vocab, GUILD_NEM_DRAFT_VOCAB);
verify multiplies the whole head, so the subset changes which drafts are offered and how many are accepted, never
the transcript.  The subset: the control tokens (ids 0-999, end of turn 11 among them) and the 256 one-byte tokens
(1000-1255), then for each corpus the most frequent tokens that together cover --coverage of its token occurrences:

    python tools/nemotron_draft_vocab.py --tokenizer <nemotron-3.5-transcribe>/tokenizer.json \
        --wikitext wikitext-103-raw-v1/train-00000-of-00002.parquet \
        --ultrachat ultrachat/data/train_sft-00000-of-00003-a3ecf92756993583.parquet \
        --out data/nemotron_draft_vocab.bin [--coverage 0.9995] [--max-chars 8000000] [--check ref.jsonl ...]

Inputs: the model's tokenizer.json; the first --max-chars characters of wikitext-103-raw-v1's first train shard
(its "text" column, empty lines skipped) and of UltraChat's first train_sft shard (every message's "content").
With the defaults it writes the shipped file: 44,602 ids (34% of the 131,072 rows; md5 3685b6e5...).  --check
prints how many token ids of transcripts (JSON lines with "ids", such as the gate's reference_fp32.jsonl) fall
outside the subset (16 of 3,369 for the gate's fp32 and bf16 references).  Needs pyarrow and tokenizers.
The output is int32 little-endian token ids, ascending.
"""
import argparse
import collections
import json
import struct
import sys


def texts_wikitext(path, max_chars):
    import pyarrow.parquet as pq

    n = 0
    for s in pq.read_table(path, columns=["text"]).column("text").to_pylist():
        if s and s.strip():
            n += len(s)
            yield s
            if n > max_chars:
                return


def texts_ultrachat(path, max_chars):
    import pyarrow.parquet as pq

    n = 0
    for msgs in pq.read_table(path, columns=["messages"]).column("messages").to_pylist():
        for m in msgs:
            s = m["content"]
            n += len(s)
            yield s
        if n > max_chars:
            return


def count(tk, texts):
    c = collections.Counter()
    batch = []
    for s in texts:
        batch.append(s)
        if len(batch) == 512:
            for e in tk.encode_batch(batch, add_special_tokens=False):
                c.update(e.ids)
            batch = []
    for e in tk.encode_batch(batch, add_special_tokens=False):
        c.update(e.ids)
    return c


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--wikitext", required=True)
    ap.add_argument("--ultrachat", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--coverage", type=float, default=0.9995)
    ap.add_argument("--max-chars", type=int, default=8_000_000)
    ap.add_argument("--check", nargs="*", default=[])
    a = ap.parse_args()
    from tokenizers import Tokenizer

    tk = Tokenizer.from_file(a.tokenizer)
    sub = set(range(0, 1256))  # control tokens 0-999, one-byte tokens 1000-1255
    for c in (count(tk, texts_wikitext(a.wikitext, a.max_chars)), count(tk, texts_ultrachat(a.ultrachat, a.max_chars))):
        total, acc = sum(c.values()), 0
        for i, n in c.most_common():
            if acc >= a.coverage * total:
                break
            sub.add(i)
            acc += n
    ids = sorted(sub)
    with open(a.out, "wb") as f:
        f.write(struct.pack("<%di" % len(ids), *ids))
    print(f"{a.out}: {len(ids)} token ids ({len(ids) / tk.get_vocab_size():.3f} of the head)")
    for path in a.check:
        ref = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    ref += json.loads(line)["ids"]
        print(f"{path}: {sum(i not in sub for i in ref)} of {len(ref)} token ids outside the subset")
    return 0


if __name__ == "__main__":
    sys.exit(main())
