#!/usr/bin/env python3
"""Validate recorded workloads and emit measured-only context/release SVG graphs.

The manifest supplies explicit hardware, source-weight and configuration identities.
Raw CSVs and recorder metadata remain authoritative; failures are exclusions, never
zero-throughput samples. Uses the same validator/statistics as workload reports.
"""
import argparse
import hashlib
from html import escape
import json
import math
from pathlib import Path
import re

from run_workload_command import sha256
from summarize_workload_benchmarks import summarize_file, option


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def collect(manifest, root):
    records, excluded = [], []
    required = ("release", "date", "engine", "engine_commit", "hardware", "model", "source_weights_sha256",
                "quantization", "workload", "settings", "comparison")
    for measurement in manifest["measurements"]:
        try:
            if any(key not in measurement for key in required):
                raise ValueError("measurement lacks required identity fields")
            if not re.fullmatch(r"[0-9a-f]{64}", measurement["source_weights_sha256"]):
                raise ValueError("source weights require a SHA-256 digest")
            if not re.fullmatch(r"[0-9a-f]{40}", measurement["engine_commit"]):
                raise ValueError("engine_commit must be a full commit SHA")
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", measurement["date"]):
                raise ValueError("date must be YYYY-MM-DD")
            settings = measurement["settings"]
            comparison = measurement["comparison"]
            if not isinstance(comparison, dict) or not all(key in comparison for key in ("id", "equivalent", "note")):
                raise ValueError("comparison requires id, equivalent and note")
            if not isinstance(comparison["equivalent"], bool):
                raise ValueError("comparison.equivalent must be boolean")
            for key in ("kv_format", "kv_placement", "threads", "affinity", "gpu_offload", "speculation",
                        "prefill_batch", "capacity_policy"):
                if key not in settings:
                    raise ValueError("missing setting " + key)
            source = root / measurement["csv"]
            meta = json.loads(source.with_suffix(".command.json").read_text())
            command = meta["command"]
            prompt_file = option(command, "--write-ids") or option(command, "--prompt-ids")
            if prompt_file is None:
                i = next(i for i, word in enumerate(command) if Path(word).name == "llama_prefill_reference")
                prompt_file = command[i + 2]
            prompt_path = Path(prompt_file)
            if not prompt_path.is_absolute():
                prompt_path = Path(meta["cwd"]) / prompt_path
            if not prompt_path.exists():
                prompt_path = source.parent / Path(prompt_file).name
            ids = [int(t) for t in prompt_path.read_text().split()]
            if not ids or min(ids) < 0:
                raise ValueError("invalid frozen prompt IDs")
            prompt_hash = digest(ids)
            groups = summarize_file(source)
            for _, data in groups:
                if (data["metadata_status"] != "complete" or data.get("sample_status") != "complete"
                        or data["role"] not in ("primary", "oracle_reference")):
                    raise ValueError("unusable measurement: " + "; ".join(data["exclusion_reasons"]))
                config, cols = data["configuration"], data["columns"]
                prompt = int(config.get("prompt_tokens", config.get("prompt", 0)))
                generated = int(config.get("generated_tokens", config.get("generated", 0)))
                if prompt != len(ids) or generated < 1:
                    raise ValueError("CSV disagrees with frozen prompt, or has no output")
                metrics = {key: cols[key] for key in ("prefill_tps", "decode_tps", "prefill_ms", "decode_ms",
                                                      "ttft_ms", "load_ms", "host_peak_bytes", "vram_sample_bytes")
                           if key in cols}
                if any(metrics[key]["median"] <= 0 for key in ("prefill_tps", "decode_tps")):
                    raise ValueError("nonpositive throughput")
                records.append({**{key: measurement[key] for key in required}, "source": str(source),
                                "csv_sha256": sha256(source), "prompt_tokens": prompt, "generated_tokens": generated,
                                "prompt_sha256": prompt_hash, "samples": data["samples"], "metrics": metrics,
                                "command_configuration": data["command_configuration"], "build_identity": data["build_identity"]})
        except (OSError, ValueError, KeyError, TypeError, StopIteration) as e:
            excluded.append({"measurement": measurement, "reason": str(e)})
    return {"schema_version": 1, "records": records, "excluded": excluded,
            "comparisons": comparisons(records), "regressions": regressions(records)}


def comparable_key(row):
    return digest({key: row[key] for key in ("hardware", "source_weights_sha256", "quantization",
                                            "prompt_sha256", "prompt_tokens", "generated_tokens")})


def comparisons(records):
    result = []
    for guild in records:
        if not guild["engine"].startswith("guild") or not guild["comparison"]["equivalent"]:
            continue
        for llama in records:
            if (not llama["engine"].startswith("llama") or not llama["comparison"]["equivalent"]
                    or llama["comparison"]["id"] != guild["comparison"]["id"]
                    or comparable_key(guild) != comparable_key(llama)):
                continue
            if any(guild["settings"][key] != llama["settings"][key] for key in ("kv_format", "speculation")):
                continue
            result.append({"guild_release": guild["release"], "date": guild["date"],
                           "guild_engine": guild["engine"], "llama_engine": llama["engine"],
                           "llama_commit": llama["engine_commit"], "comparison_id": guild["comparison"]["id"],
                           "prompt_tokens": guild["prompt_tokens"], "identity": comparable_key(guild),
                           "guild_settings": guild["settings"], "llama_settings": llama["settings"],
                           "speedup": {key: guild["metrics"][key]["median"] / llama["metrics"][key]["median"]
                                       for key in ("prefill_tps", "decode_tps")}})
    return result


def regressions(records):
    groups, result = {}, []
    for row in records:
        key = digest([comparable_key(row), row["settings"], row["engine"]])
        groups.setdefault(key, []).append(row)
    for key, rows in groups.items():
        rows.sort(key=lambda row: (row["date"], row["release"]))
        for previous, current in zip(rows, rows[1:]):
            if previous["release"] == current["release"]:
                continue
            result.append({"identity": key, "previous": previous["release"], "current": current["release"],
                           "percent": {metric: 100 * (current["metrics"][metric]["median"] /
                                                       previous["metrics"][metric]["median"] - 1)
                                       for metric in ("prefill_tps", "decode_tps")}})
    return result


def svg(path, title, series, xlabel, ylabel):
    """Small dependency-free measured-point plot, including dispersion bars."""
    labels = sorted({x for points in series.values() for x, _, _, _ in points})
    if not labels:
        return
    ymax = max(high for points in series.values() for _, _, _, high in points) * 1.12
    if not math.isfinite(ymax) or ymax <= 0:
        raise ValueError("invalid plot range")
    colors = ["#1565c0", "#c62828", "#2e7d32", "#6a1b9a", "#e65100", "#00695c"]
    width, height = 1000, 440 + 24 * len(series)
    def xcoord(x):
        return 90 + labels.index(x) * 820 / max(1, len(labels) - 1)
    def ycoord(y):
        return 360 - y * 290 / ymax
    text = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            '<g font-family="sans-serif" font-size="13" fill="#222">',
            f'<text x="30" y="28" font-size="17">{escape(title)}</text>',
            f'<text x="15" y="54">{escape(ylabel)}</text>']
    for i in range(6):
        y = ymax * i / 5
        text += [f'<path d="M90 {ycoord(y):.2f} H910" stroke="#ddd"/>',
                 f'<text x="30" y="{ycoord(y)+4:.2f}">{y:.2f}</text>']
    for x in labels:
        text.append(f'<text x="{xcoord(x):.2f}" y="382" text-anchor="middle">{escape(str(x))}</text>')
    text.append(f'<text x="500" y="410" text-anchor="middle">{escape(xlabel)}</text>')
    for index, (label, points) in enumerate(sorted(series.items())):
        color = colors[index % len(colors)]
        coords = " ".join(f"{xcoord(x):.2f},{ycoord(y):.2f}" for x, y, _, _ in sorted(points))
        text.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2"/>')
        for x, y, low, high in points:
            text += [f'<path d="M{xcoord(x):.2f} {ycoord(low):.2f} V{ycoord(high):.2f}" stroke="{color}"/>',
                     f'<circle cx="{xcoord(x):.2f}" cy="{ycoord(y):.2f}" r="4" fill="{color}"><title>{y:.4f}</title></circle>']
        text.append(f'<text x="90" y="{440 + 24*index}" fill="{color}">{escape(label)}</text>')
    text += ['</g></svg>']
    path.write_text("\n".join(text) + "\n")


def graphs(report, output):
    contexts, history = {}, {}
    for row in report["records"]:
        # All execution settings stay in the series identity, even when engines are
        # explicitly declared comparable. Different precision/offload never merges.
        base = {key: row[key] for key in ("hardware", "model", "source_weights_sha256", "quantization", "workload",
                                         "generated_tokens")}
        config_id = digest({**base, "settings": row["settings"]})[:12]
        label = f'{row["engine"]} / {row["release"]} / {config_id}'
        context_id = digest(base)[:12]
        for metric in ("prefill_tps", "decode_tps"):
            m = row["metrics"][metric]
            contexts.setdefault((context_id, metric), {}).setdefault(label, []).append(
                (row["prompt_tokens"], m["median"], m["p05"], m["p95"]))
            # Exact prompt, output count and settings are mandatory across releases.
            hist_id = digest({**base, "settings": row["settings"], "prompt": row["prompt_sha256"],
                              "generated": row["generated_tokens"]})[:12]
            history.setdefault((hist_id, metric), {}).setdefault(row["engine"], []).append(
                (row["date"] + " " + row["release"], m["median"], m["p05"], m["p95"]))
    for (identity, metric), series in contexts.items():
        svg(output / f"context-{identity}-{metric}.svg", f"Populated context: {identity} ({metric}; median, p05–p95)",
            series, "Real prompt tokens", "tokens / second")
        if metric == "decode_tps":
            relative = {}
            for label, points in series.items():
                baseline = [y for x, y, _, _ in points if x == 512]
                if len(baseline) == 1:
                    relative[label] = [(x, y / baseline[0], lo / baseline[0], hi / baseline[0]) for x, y, lo, hi in points]
            svg(output / f"context-{identity}-relative.svg", "Decode relative to measured 512-token baseline",
                relative, "Real prompt tokens", "throughput / 512-token throughput")
    for (identity, metric), series in history.items():
        svg(output / f"history-{identity}-{metric}.svg", f"Release history: {identity} ({metric})",
            series, "Release date and identifier", "tokens / second")
    ratios = {}
    for row in report["comparisons"]:
        identity = digest([row["identity"], row["guild_settings"], row["llama_settings"], row["llama_commit"]])[:12]
        for metric, ratio in row["speedup"].items():
            label = row["guild_engine"] + " / " + row["llama_engine"]
            ratios.setdefault((identity, metric), {}).setdefault(label, []).append(
                (row["date"] + " " + row["guild_release"], ratio, ratio, ratio))
    for (identity, metric), series in ratios.items():
        svg(output / f"speedup-{identity}-{metric}.svg", f"Equivalent workloads: {identity} ({metric})",
            series, "Guild release date and identifier", "Guild / llama.cpp throughput")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifest", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    report = collect(json.loads(a.manifest.read_text()), a.manifest.parent)
    (a.output / "measurements.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    graphs(report, a.output)
    print(f'{len(report["records"])} valid measured groups; {len(report["excluded"])} exclusions')
    return int(bool(report["excluded"]))


if __name__ == "__main__":
    raise SystemExit(main())
