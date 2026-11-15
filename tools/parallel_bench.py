#!/usr/bin/env python3
"""Concurrent requests through the HTTP server: what "parallel": N buys a client (#465).

Starts serve/server.py with a run config (its "parallel" set or removed by --parallel), sends C different chat
requests at once (streamed, so each one's first token is timed), and reports per round: the aggregate decode rate
(every completion token over the round's wall time), each request's own rate, and the first-token times (median and
the last request's).  Rounds repeat for every C of --concurrency, --rounds times each, the median round reported.

  python tools/parallel_bench.py --config guild-qwen.json --parallel 3 --concurrency 1,2,3 --max-tokens 256

Greedy (temperature 0) and thinking off by default, so the texts are comparable across settings.
"""
import argparse, json, os, signal, statistics, subprocess, sys, tempfile, threading, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

QUESTIONS = [
    "Explain in detail how a B-tree works, with an example of an insertion that splits a node.",
    "Write a Python function that merges overlapping intervals, then explain its complexity.",
    "Compare TCP and QUIC: handshake, congestion control, head-of-line blocking and multiplexing.",
    "Tell the history of the computation of pi, from Archimedes to the present day.",
    "What were the main causes of the French Revolution? Give at least five.",
    "Explain how a transformer's attention works, step by step, with the shapes of every matrix.",
    "Give a recipe for an apple pie, step by step, with quantities.",
    "Describe the life cycle of a star like the Sun, from the nebula to the white dwarf.",
]


def post_stream(url, body, timeout=1800):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    first, n, chunks, text = None, 0, 0, []
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                ev = json.loads(data)
            except ValueError:
                continue
            for ch in ev.get("choices", []):
                piece = (ch.get("delta") or {}).get("content")
                if piece:
                    if first is None:
                        first = time.monotonic() - t0
                    text.append(piece)
                    chunks += 1
            usage = ev.get("usage")
            if usage and usage.get("completion_tokens"):
                n = int(usage["completion_tokens"])
    return {"ttft": first if first is not None else float("nan"), "total": time.monotonic() - t0, "tokens": n or chunks,
            "text": "".join(text)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="a run config (the JSON serve/server.py --config reads)")
    ap.add_argument("--parallel", type=int, default=None, help="the config's \"parallel\" for this run (0: off)")
    ap.add_argument("--concurrency", default="1,2,3", help="requests sent at once, a comma-separated list")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--port", type=int, default=4799)
    ap.add_argument("--out", help="write every round's numbers here (JSON)")
    ap.add_argument("--env", action="append", default=[], help="VAR=value for the server (and so the engine)")
    a = ap.parse_args()

    cfg = json.loads(Path(a.config).read_text())
    if a.parallel is not None:
        if a.parallel > 1:
            cfg["parallel"] = a.parallel
        else:
            cfg.pop("parallel", None)
    cfg["port"] = a.port
    cfg["host"] = "127.0.0.1"
    log_dir = Path(tempfile.mkdtemp(prefix="parallel_bench_"))
    cfg["log"] = str(log_dir / "engine.log")
    cfg_path = log_dir / "config.json"
    cfg_path.write_text(json.dumps(cfg))
    env = dict(os.environ)
    for kv in a.env:
        k, _, v = kv.partition("=")
        env[k] = v
    srv = subprocess.Popen([sys.executable, str(ROOT / "serve" / "server.py"), "--engine", "guild", "--config",
                            str(cfg_path), "--port", str(a.port)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, env=env, start_new_session=True)
    ready = threading.Event()
    server_log = []

    def pump():
        for line in srv.stdout:
            server_log.append(line)
            if line.startswith("ready:"):
                ready.set()
    threading.Thread(target=pump, daemon=True).start()
    base = f"http://127.0.0.1:{a.port}"
    try:
        # the server answers before the engine has loaded: wait for the model
        t_load = time.monotonic()
        while not ready.is_set():
            if srv.poll() is not None:
                raise SystemExit("the server ended:\n" + "".join(server_log[-30:]))
            time.sleep(1)
        while True:
            try:
                with urllib.request.urlopen(base + "/v1/status", timeout=10) as r:
                    st = json.loads(r.read())
                if st.get("loaded"):
                    serving = (st.get("concurrency") or {}).get("serving")
                    break
            except Exception:
                pass
            if srv.poll() is not None:
                raise SystemExit("the server ended:\n" + "".join(server_log[-30:]))
            time.sleep(2)
        print(f"loaded in {time.monotonic() - t_load:.0f} s; parallel {cfg.get('parallel', 'off')}, the engine serves "
              f"{serving} at once", flush=True)
        model = cfg.get("model_name", "guild")
        results = {}
        warm = post_stream(base + "/v1/chat/completions",
                           {"model": model, "messages": [{"role": "user", "content": "Say hello."}], "max_tokens": 8,
                            "temperature": 0, "stream": True, "chat_template_kwargs": {"enable_thinking": False}})
        print(f"warm-up: {warm['tokens']} tokens", flush=True)
        for c in [int(x) for x in a.concurrency.split(",")]:
            rounds = []
            for r in range(a.rounds):
                qs = [QUESTIONS[(r * c + i) % len(QUESTIONS)] for i in range(c)]
                out = [None] * c
                t0 = time.monotonic()

                def one(i):
                    out[i] = post_stream(base + "/v1/chat/completions",
                                         {"model": model, "messages": [{"role": "user", "content": qs[i]}],
                                          "max_tokens": a.max_tokens, "temperature": 0, "stream": True,
                                          "stream_options": {"include_usage": True},
                                          "chat_template_kwargs": {"enable_thinking": False}})
                ts = [threading.Thread(target=one, args=(i,)) for i in range(c)]
                for t in ts:
                    t.start()
                for t in ts:
                    t.join()
                wall = time.monotonic() - t0
                toks = sum(o["tokens"] for o in out)
                rd = {"wall_s": wall, "tokens": toks, "aggregate_tok_s": toks / wall,
                      "per_request_tok_s": [o["tokens"] / max(o["total"] - o["ttft"], 1e-9) for o in out],
                      "ttft_s": [o["ttft"] for o in out], "total_s": [o["total"] for o in out],
                      "texts": [o["text"][:80] for o in out]}
                rounds.append(rd)
                print(f"C={c} round {r + 1}: {toks} tokens in {wall:.1f} s = {rd['aggregate_tok_s']:.1f} tok/s; "
                      f"per request {statistics.median(rd['per_request_tok_s']):.1f} tok/s; first token median "
                      f"{statistics.median(rd['ttft_s']):.2f} s, last {max(rd['ttft_s']):.2f} s", flush=True)
            best = sorted(rounds, key=lambda x: x["aggregate_tok_s"])[len(rounds) // 2]
            results[c] = {"median_round": best, "rounds": rounds}
        print("SUMMARY " + json.dumps({c: {"aggregate_tok_s": round(v["median_round"]["aggregate_tok_s"], 1),
                                            "per_request_tok_s": round(statistics.median(
                                                v["median_round"]["per_request_tok_s"]), 1),
                                            "ttft_median_s": round(statistics.median(v["median_round"]["ttft_s"]), 2),
                                            "ttft_last_s": round(max(v["median_round"]["ttft_s"]), 2)}
                                       for c, v in results.items()}), flush=True)
        if a.out:
            Path(a.out).write_text(json.dumps({"parallel": cfg.get("parallel", 0), "results": results}, indent=1))
    finally:
        try:
            os.killpg(srv.pid, signal.SIGTERM)
            srv.wait(timeout=120)
        except Exception:
            try:
                os.killpg(srv.pid, signal.SIGKILL)
            except Exception:
                pass
        print(f"(server and engine logs in {log_dir})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
