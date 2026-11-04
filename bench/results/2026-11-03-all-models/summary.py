"""The all-models bench's numbers from opt/all's outputs, as Markdown tables."""
import glob, json, re, statistics as st
O = '/home/ubuntu/work/nemotron/opt/all'

def qwen(label):
    r = json.load(open('%s/b-%s.json' % (O, label)))
    by = {}
    for run in r['runs']:
        by.setdefault(run['prompt'], []).append(run)
    out = {}
    for p, runs in by.items():
        out[p] = {'tok_s': [round(x['decode_tok_s'], 2) for x in runs],
                  'ttft_s': [round(x['ttft_s'], 2) for x in runs],
                  'prompt_tokens': runs[0]['prompt_tokens'], 'prompt_ms': [round(x['prompt_ms']) for x in runs],
                  'drafts': '%s/%s' % (runs[0].get('drafts_accepted'), runs[0].get('drafts_offered')),
                  'same_text': len({tuple(x['output_ids']) for x in runs}) == 1}
    return out

print('## Qwen3.8-Flash-Next')
for lab in ('qwen_short', 'qwen_more', 'qwen_4k'):
    try:
        for p, v in qwen(lab).items():
            print(lab, p, v)
    except Exception as e:
        print(lab, 'missing', e)

print('## GLM-5.3-Flash')
for f in ('glm_chat', 'glm_prompt'):
    try:
        for l in open('%s/%s.out' % (O, f)):
            if l.startswith('gen ') or 'prompt ' in l and 'tokens in' in l and f == 'glm_prompt' or l.startswith('decode'):
                print(f, re.sub(r'\(logits [0-9a-f]+\); ', '', l.strip())[:330])
    except Exception as e:
        print(f, 'missing', e)

print('## Ornith-1.5-35B')
for s in (0, 4):
    try:
        j = json.load(open('%s/ornith_spec%d.json' % (O, s)))
        print('spec', s, {k: j[k] for k in ('prompt_tokens', 'generated_tokens', 'prefill_tps_p50', 'decode_tps_p50',
                                             'ttft_ms_p50', 'deterministic_token_parity') if k in j})
    except Exception as e:
        print('spec', s, 'missing', e)

print('## StartLux-Decision')
try:
    for l in open('%s/startlux.jsonl' % O):
        d = json.loads(l)
        print(d['id'], d['usage']['input_tokens'], 'tokens', round(d['timing']['mean_ms']), 'ms mean of',
              len(d['timing']['ms']), [round(x) for x in d['timing']['ms']])
except Exception as e:
    print('startlux missing', e)

print('## Nemotron 3.5 Transcribe')
for mode in ('greedy', 'mtp3'):
    try:
        rows = [json.loads(l) for l in open('%s/nemotron_%s.jsonl' % (O, mode))]
        by = {}
        for r in rows:
            if r.get('run', 0) > 0:
                by.setdefault(r['id'], []).append(r)
        for cid, rs in by.items():
            tot = st.median(r['total_ms'] for r in rs)
            print(mode, cid, '%.1f s audio' % rs[0]['seconds'], 'total %.0f ms (prefill %.0f, decode %.0f), RTF %.3f'
                  % (tot, st.median(r['prefill_ms'] for r in rs), st.median(r['decode_ms'] for r in rs),
                     tot / 1000 / rs[0]['seconds']), 'same ids' if len({tuple(r['ids']) for r in rs}) == 1 else 'IDS DIFFER')
    except Exception as e:
        print(mode, 'missing', e)

print('## Drama 3')
for f in sorted(glob.glob('%s/drama3_*.log' % O)):
    last = [l.strip() for l in open(f) if l.startswith('guild-generate:')]
    print(f.rsplit('/', 1)[-1], last[-1] if last else 'no result line')
