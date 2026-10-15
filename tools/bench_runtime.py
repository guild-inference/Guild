"""Repeat frozen token-ID requests against a Guild-compatible engine pipe.

The same runner accepts model/pack/tokenizer/profile/draft paths and engine flags;
it has no model- or machine-specific paths. A JSON prompt file can contain text or
explicit token_ids, allowing exact comparisons between binaries and configurations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import shutil
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
from guild_tokenizer import Tokenizer
from serve.frontend import ChatTemplate


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--exe', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--pack', required=True)
    ap.add_argument('--tokenizer')
    ap.add_argument('--profile')
    ap.add_argument('--mtp')
    ap.add_argument('--context', type=int, default=4096)
    ap.add_argument('--spec', type=int, default=4)
    ap.add_argument('--spec-min-p', type=float, default=0.5)
    ap.add_argument('--cache', default='auto')
    ap.add_argument('--pcie-frac', type=float)
    ap.add_argument('--prefill', type=int, default=512)
    ap.add_argument('--resident-budget-gib', type=int)
    ap.add_argument('--pool-workers', type=int, default=0)
    ap.add_argument('--runs', type=int, default=3)
    ap.add_argument('--max-new', type=int, default=256)
    ap.add_argument('--warmup', type=int, default=16)
    ap.add_argument('--prompts', help='JSON list: {name, text} or {name, token_ids}')
    ap.add_argument('--timeout', type=float, default=600)
    ap.add_argument('--out', required=True)
    ap.add_argument('--extra-args', nargs=argparse.REMAINDER, default=[])
    a = ap.parse_args()
    if min(a.context, a.runs, a.max_new, a.warmup) < 1:
        ap.error('context, runs, max-new and warmup must be positive')
    exe, model, pack = (Path(p).resolve() for p in (a.exe, a.model, a.pack))
    tokenizer = Path(a.tokenizer).resolve() if a.tokenizer else pack / 'tokenizer'
    tok = Tokenizer.from_gguf(model)
    template_file = tokenizer / 'chat_template.jinja'
    template = ChatTemplate(template_file) if template_file.exists() else None
    definitions = json.loads(Path(a.prompts).read_text()) if a.prompts else [
        {'name': 'prose', 'text': 'Explain how rain forms in at least 1000 words, covering the complete water cycle in detail.'},
        {'name': 'code', 'text': 'Write a complete Python implementation of a least recently used cache with type hints, detailed comments, examples, and an explanation of its time complexity.'},
    ]
    if not isinstance(definitions, list) or not definitions:
        ap.error('prompts must be a nonempty JSON list')
    prompts = {}
    for definition in definitions:
        name = definition['name']
        if name in prompts:
            ap.error(f'duplicate prompt name: {name}')
        ids = definition.get('token_ids')
        if ids is None:
            text = definition['text']
            if template:
                text = template.render([{'role': 'user', 'content': text}], enable_thinking=False)
            ids = tok.encode(text, parse_special=True)
        if not ids or any(not isinstance(t, int) or t < 0 for t in ids):
            ap.error(f'invalid token IDs: {name}')
        if len(ids) + max(a.max_new, a.warmup) + 8 > a.context:
            ap.error(f'prompt and output do not fit context: {name}')
        prompts[name] = ids
    command = [str(exe), '--serve', '--native', str(model), '--pack', str(pack),
               '--tokenizer', str(tokenizer), '--max-context', str(a.context), '--kv', 'fp16', '--kv-host-only',
               '--spec', str(a.spec), '--spec-min-p', str(a.spec_min_p), '--expert-cache', a.cache,
               '--prefill', str(a.prefill), '--pool-workers', str(a.pool_workers)]
    for flag, value in [('--expert-profile', a.profile), ('--mtp', a.mtp),
                        ('--pcie-frac', a.pcie_frac), ('--resident-budget-gib', a.resident_budget_gib)]:
        if value is not None:
            command += [flag, str(Path(value).resolve()) if flag in ('--expert-profile', '--mtp') else str(value)]
    command += a.extra_args
    env = dict(os.environ, GUILD_DECODE_TIMING='1')
    report = {'date': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'command': command,
              'executable_sha256': hashlib.sha256(exe.read_bytes()).hexdigest(), 'prompts': prompts,
              'environment': {k: env[k] for k in ('GUILD_DECODE_TIMING', 'GUILD_LOGITS_VALIDATION_HOST',
                  'GUILD_COMMIT_SYNC', 'GUILD_POOL_WORKERS', 'GUILD_IQ_MT_MIN', 'GUILD_FORCE_ISA',
                  'GUILD_NO_IQ256', 'GUILD_NO_IQ512', 'GUILD_PREFILL_BATCH', 'GUILD_PREFILL_SEQUENTIAL',
                  'CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'STRATA_IQ_MT_MIN') if k in env},
              'runs': []}
    output = Path(a.out)
    output.write_text(json.dumps(report, indent=2))
    lines = queue.Queue()
    print('START ' + json.dumps(command), flush=True)
    started = time.monotonic()
    with output.with_suffix('.log').open('w') as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=log, text=True, bufsize=1)

        def reader():
            for line in process.stdout:
                lines.put(line.strip())
            lines.put(None)

        threading.Thread(target=reader, daemon=True).start()

        def next_line():
            line = lines.get(timeout=a.timeout)
            if line is None or line.startswith('ERR'):
                raise RuntimeError(f'engine failed: {line}; see {output.with_suffix(".log")}')
            return line

        def generate(ids, limit):
            process.stdin.write(f'GEN {limit} ' + ','.join(map(str, ids)) + '\n')
            process.stdin.flush()
            start = time.monotonic()
            first = None
            tokens = []
            while True:
                line = next_line()
                if line.startswith('T '):
                    first = first if first is not None else time.monotonic() - start
                    tokens.append(int(line.split()[1]))
                elif line.startswith('DONE '):
                    f = line.split()
                    count, prompt_n, prompt_ms, decode_ms = int(f[1]), int(f[2]), float(f[3]), float(f[4])
                    if count != len(tokens) or count < 1:
                        raise RuntimeError(f'completion count mismatch: DONE={count}, received={len(tokens)}')
                    row = {'generated': count, 'prompt_tokens': prompt_n, 'prompt_ms': prompt_ms,
                           'decode_ms': decode_ms, 'decode_tok_s': count * 1000 / decode_ms,
                           'ttft_s': first, 'wall_s': time.monotonic() - start, 'finish': f[5],
                           'output_ids': tokens, 'text': tok.decode(tokens), 'done': line}
                    for index, name in [(6, 'drafts_accepted'), (7, 'drafts_offered'), (8, 'reused'),
                                        (9, 'gpu_hits'), (10, 'cache_lookups'), (11, 'ram_blobs'),
                                        (12, 'file_blobs'), (14, 'prompt_read'), (15, 'offloaded')]:
                        if len(f) > index:
                            row[name] = int(f[index])
                    return row

        try:
            while True:
                line = next_line()
                if line.startswith('READY '):
                    report['ready'], report['load_s'] = line, time.monotonic() - started
                    print(f'READY load_s={report["load_s"]:.2f} {line}', flush=True)
                    break
            generate(next(iter(prompts.values())), a.warmup)
            for name, ids in prompts.items():
                for repeat in range(a.runs):
                    row = generate(ids, a.max_new)
                    row.update(prompt=name, repeat=repeat + 1)
                    report['runs'].append(row)
                    output.write_text(json.dumps(report, indent=2))
                    print('RESULT ' + json.dumps({k: row[k] for k in ('prompt', 'repeat', 'generated', 'decode_tok_s', 'ttft_s')}), flush=True)
            report['median_by_prompt'] = {name: statistics.median(r['decode_tok_s'] for r in report['runs']
                                                                 if r['prompt'] == name) for name in prompts}
            if shutil.which('nvidia-smi'):
                try:
                    report['gpu_after'] = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,temperature.gpu,power.draw',
                                                                   '--format=csv,noheader'], text=True, stderr=subprocess.DEVNULL).strip()
                except subprocess.CalledProcessError:
                    pass
            output.write_text(json.dumps(report, indent=2))
            print('SUMMARY ' + json.dumps(report['median_by_prompt']), flush=True)
        finally:
            if process.poll() is None:
                try:
                    process.stdin.write('QUIT\n')
                    process.stdin.flush()
                except BrokenPipeError:
                    pass
                try:
                    process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    process.wait(timeout=30)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
