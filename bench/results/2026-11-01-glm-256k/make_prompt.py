"""make_prompt.py: a ~15K-token real-text GLM-5.3-Flash prompt (chat-templated) from corpus.txt, written as the
comma-separated ids glm5_run takes (ctx15k.ids) and the rendered text (ctx15k.txt)."""
import json, sys
from pathlib import Path
ROOT = Path('/home/ubuntu/Guild')
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'tools'))
from guild_tokenizer import Tokenizer
from serve.frontend import ChatTemplate
D = Path('/home/ubuntu/models/GLM-5.3-Flash/pack/tokenizer')
meta = json.loads((D / 'tokenizer.json').read_text())
vocab = json.loads((D / 'vocab.json').read_text())
tokens = [None] * len(vocab)
for t, i in vocab.items():
    tokens[i] = t
merges = [l for l in (D / 'merges.txt').read_text(encoding='utf-8').split('\n') if l]
types = json.loads((D / 'token_type.json').read_text())
tok = Tokenizer(tokens, merges, types, meta['pre'], meta['special_ids'])
tmpl = ChatTemplate(D / 'chat_template.jinja')
corpus = Path('/home/ubuntu/lanes/prompts/corpus.txt').read_text()
ask = '\n\n---\n\nSummarize the documents and code above: list the main components, what each one does, and how they fit together.'
def ids_for(nchars):
    text = corpus[:nchars]
    cut = text.rfind('\n')
    text = text[:cut] if cut > 0 else text
    r = tmpl.render([{'role': 'user', 'content': text + ask}])
    return r, tok.encode(r, parse_special=True)
target = int(sys.argv[1]) if len(sys.argv) > 1 else 15000
lo, hi = 1000, 1750000
while hi - lo > 100:
    mid = (lo + hi) // 2
    if len(ids_for(mid)[1]) <= target: lo = mid
    else: hi = mid
r, ids = ids_for(lo)
out = Path('/home/ubuntu/work/nemotron/opt/glm128k')
(out / f'ctx{target}.ids').write_text(','.join(map(str, ids)))
(out / f'ctx{target}.txt').write_text(r)
print(len(ids), 'tokens from', lo, 'chars; head', r[:80].replace('\n', '|'), '... tail', r[-160:].replace('\n', '|'))
print('decode check', tok.decode(ids) == r)
