#!/usr/bin/env python3
"""Summarize workload CSVs by schema and exact configuration, retaining exclusions.

Profile/counter timings are diagnostic, not primary throughput measurements.
CSV and command files are read only. Use --output for a separate snapshot while
another measurement owner is still writing the default summary.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import re
import statistics as stats

from run_workload_command import public_environment


TIMINGS = {"prefill_ms", "decode_ms", "wall_ms", "ttft_ms", "load_ms", "prefill_tps", "decode_tps"}
MAIN_METRICS = TIMINGS | {
    "vram_sample_bytes", "host_peak_bytes", "host_kv_allocated_bytes", "kv_staging_bytes", "prefill_scratch_bytes",
    "expert_groups", "expert_weight_bytes", "activation_transfer_bytes", "cpu_gu_ms", "cpu_quant_ms", "cpu_down_ms",
    "cpu_group_ms", "host_wait_ms", "gpu_dense_ms", "gpu_gdn_ms", "gpu_attention_ms", "file_reads", "gpu_groups",
}
REJECTED_PROJECTIONS = ("column32-", "column64-", "column128-", "paired-rows-")


def quantile(values, q):
    if not values or not 0 <= q <= 1:
        raise ValueError("quantile requires samples and 0 <= q <= 1")
    s = sorted(values)
    x = (len(s) - 1) * q
    i = int(x)
    return s[i] + (s[min(i + 1, len(s) - 1)] - s[i]) * (x - i)


def distribution(values):
    if not values or not all(math.isfinite(x) for x in values):
        raise ValueError("empty or nonfinite sample")
    med = stats.median(values)
    return {"median": med, "min": min(values), "max": max(values), "p05": quantile(values, .05),
            "p95": quantile(values, .95), "iqr": quantile(values, .75) - quantile(values, .25),
            "mad": stats.median(abs(x - med) for x in values)}


def classify(fields):
    fields = set(fields)
    if {"weight_bytes_per_iteration", "iterations", "cores", "tokens", "wall_ms"} <= fields:
        return "cpu_expert_microbenchmark", {"wall_ms"}
    if {"start_ns", "end_ns"} <= fields:
        return "cpu_activity_profile", set()
    if "layer" in fields and ("rows" in fields or "row" in fields):
        return "per_layer_profile", set()
    if {"batch", "prefill_tps", "decode_tps"} <= fields:
        return "main_benchmark", MAIN_METRICS & fields
    if {"prefill_tokens", "decode_evals", "prefill_ms", "decode_ms"} <= fields:
        return "oracle_benchmark", (TIMINGS | {"host_peak_bytes"}) & fields
    return "unrecognized_csv", set()


def option(command, name):
    for i, word in enumerate(command):
        if word == name and i + 1 < len(command):
            return command[i + 1]
        if word.startswith(name + "="):
            return word[len(name) + 1:]
    return None


def command_configuration(metadata):
    command = metadata.get("command", [])
    config = {}
    for name in ("context", "prompt", "generate", "batch", "runs"):
        value = option(command, "--" + name)
        if value is not None:
            config[name] = value
    # Public per-command env assignments may be inside numactl/env or sudo/perf.
    assignments = dict(word.split("=", 1) for word in command if re.match(r"^[A-Z][A-Z0-9_]*=", word))
    if "environment" in metadata or assignments:
        config["environment"] = {**public_environment(metadata.get("environment", {})), **public_environment(assignments)}
    for i, word in enumerate(command):
        executable = Path(word).name
        if executable == "ornith_prefill_bench":
            config["full_warmup"] = "--full-warmup" in command
            config["sequential"] = "--sequential" in command
        elif executable == "llama_prefill_reference" and len(command) > i + 7:
            config.update(zip(("model", "prompt_ids", "context", "requested_generated", "n_gpu_layers", "threads"),
                              command[i + 1:i + 7]))
            config.update(host_kv="--host-kv" in command, cpu_experts="--cpu-experts" in command,
                          aligned="--aligned" in command)
        elif executable == "cpu_expert_workload" and len(command) > i + 8:
            config["repetitions"] = command[i + 8]
    return config


def provenance(path):
    metadata_path = path.with_suffix(".command.json")
    notes, reasons = [], []
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text())
            if (not isinstance(metadata, dict) or not isinstance(metadata.get("command", []), list)
                    or not all(isinstance(word, str) for word in metadata.get("command", []))
                    or not isinstance(metadata.get("environment", {}), dict)):
                raise ValueError("command metadata must contain an object and command array")
        except (ValueError, OSError) as e:
            return {}, {}, "invalid", [f"unreadable command metadata: {e}"], notes
        if metadata.get("exit_code") is None:
            status = "incomplete"
            reasons.append("command metadata has no final exit_code; CSV rows alone do not establish success")
        elif (metadata["exit_code"] != 0 or metadata.get("wrapper_exit_code", 0) != 0
              or metadata.get("state", "completed") != "completed" or metadata.get("error")
              or metadata.get("cleanup_errors")):
            status = "failed"
            reasons.append("command or measurement wrapper failed")
        else:
            status = "complete"
    else:
        metadata, status = {}, "metadata_missing"
        reasons.append("no .command.json; command, completion and build identity are unverified")
        if path.name == "baseline-b256.csv":
            notes.append("Initial manual baseline; retain the CSV. No metadata has been reconstructed retroactively.")
    return metadata, command_configuration(metadata), status, reasons, notes


def measurement_role(path, schema, metadata, config):
    name = path.stem.lower()
    command = metadata.get("command", [])
    reasons, notes = [], []
    if path.name == "baseline-profile.csv":
        return "invalid_profile_run", ["Nsight import failed and GPU validation ran concurrently; invalid primary timing"], notes
    if schema.endswith("profile"):
        return "instrumentation_profile", ["profile rows are instrumentation, not repeated throughput samples"], notes
    if schema == "unrecognized_csv":
        return "unrecognized", ["CSV schema is not a supported workload benchmark"], notes
    if schema == "cpu_expert_microbenchmark":
        return "cpu_expert_microbenchmark", ["kernel microbenchmark, not end-to-end throughput"], [
            "effective_weight_GBps counts algorithmic weight bytes; it is not measured DRAM bandwidth and may include cache reuse."]
    activity_profile = (name.startswith(("profile-", "activity-"))
                        or any(word.split("=", 1)[0] == "--activity" for word in command)
                        or any("PROFILE" in key or key == "GUILD_DECODE_TIMING" for key in config.get("environment", {})))
    instrumentation = (activity_profile or name.startswith(("counters", "imc-"))
                       or any(Path(word).name in ("nsys", "ncu", "perf") for word in command)
                       or any(word.split("=", 1)[0] == "--dump" for word in command))
    if instrumentation:
        reasons.append("profiling/counters/diagnostic instrumentation; exclude from primary throughput")
        notes.append("Diagnostic/counter timings are kept separate; no slowdown correction has been applied.")
        if activity_profile:
            notes.append("Activity/profile instrumentation was observed to slow these runs by about 20%; do not blend with primary runs.")
    if name.startswith(REJECTED_PROJECTIONS):
        return "rejected_projection_experiment", reasons + ["rejected projection experiment; implementation removed, raw samples retained"], notes
    if name.startswith(("single-row-", "expert32-")):
        return "rejected_cpu_expert_experiment", reasons + ["rejected CPU expert trial; implementation removed, raw samples retained"], notes
    if "projections" in name:
        return "projection_experiment", reasons + ["historical projection experiment, not final-matrix evidence"], notes
    if instrumentation:
        return "instrumented_benchmark", reasons, notes
    if schema == "oracle_benchmark":
        return "oracle_reference", ["reference implementation; keep its context/offload configuration separate"], notes
    return "primary", reasons, notes


def summarize_file(path, label_prefix=None):
    file_label = label_prefix or path.name
    metadata, config, status, reasons, notes = provenance(path)
    base = {"source": str(path), "metadata_status": status, "command_configuration": config,
            "exclusion_reasons": reasons, "notes": notes}
    base["build_identity"] = {k: v for k, v in metadata.items() if k.endswith("_sha256") or k == "git_head"}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        schema, metrics = classify(fields)
        role, role_reasons, role_notes = measurement_role(path, schema, metadata, config)
        base.update(schema=schema, role=role)
        reasons.extend(role_reasons)
        notes.extend(role_notes)
        # Do not expand timestamp/layer profiles into thousands of bogus groups.
        if not metrics:
            base.update(samples=sum(1 for _ in reader), columns={}, primary_eligible=False)
            return [(file_label, base)]
        rows = list(reader)
    if not rows:
        base.update(samples=0, columns={}, primary_eligible=False, metadata_status="invalid")
        reasons.append("CSV has no completed measurement rows")
        return [(file_label, base)]
    if len(set(fields)) != len(fields) or any(None in row or any(v is None for v in row.values()) for row in rows):
        base.update(samples=len(rows), columns={}, primary_eligible=False, metadata_status="invalid")
        reasons.append("malformed/truncated CSV rows or duplicate header fields")
        return [(file_label, base)]
    # Every non-metric field is a dimension. This also separates future context,
    # offload, prompt/output lengths and finish reasons instead of silently mixing.
    dimensions = [col for col in fields if col not in metrics and col != "run"]
    # Completion is checked against requested configurations. Actual output
    # lengths/finish reasons may vary (early EOS) and are summarized separately.
    controlled = [col for col in dimensions if col not in ("generated_tokens", "generated", "decode_evals", "finish")]
    groups, control_groups = {}, {}
    for row in rows:
        key = tuple(row[col] for col in dimensions)
        groups.setdefault(key, []).append(row)
        control_groups.setdefault(tuple(row[col] for col in controlled), []).append(row)
    result = []
    for index, (key, records) in enumerate(groups.items()):
        data = {**base, "exclusion_reasons": list(reasons), "notes": list(notes), "samples": len(records),
                "configuration": dict(zip(dimensions, key)), "columns": {}, "sample_status": "complete"}
        try:
            values = {col: [float(row[col]) for row in records] for col in sorted(metrics)}
            if any(not math.isfinite(x) or x < 0 for samples in values.values() for x in samples):
                raise ValueError("negative/nonfinite metric")
            for row in records:
                for col in ("batch", "prompt_tokens", "prefill_tokens", "generated_tokens", "prompt", "generated", "decode_evals",
                            "layer", "groups", "tokens", "cores", "iterations", "weight_bytes_per_iteration"):
                    if col in row:
                        value = float(row[col])
                        if not math.isfinite(value) or value < 0 or not value.is_integer():
                            raise ValueError(f"invalid count/configuration field {col}")
            if schema == "oracle_benchmark":
                for metric, count, duration in (("prefill_tps", "prefill_tokens", "prefill_ms"),
                                                ("decode_tps", "decode_evals", "decode_ms")):
                    values[metric] = [float(row[count]) * 1000 / float(row[duration]) for row in records]
            elif schema == "cpu_expert_microbenchmark":
                values["wall_ms_per_iteration"] = [float(row["wall_ms"]) / float(row["iterations"]) for row in records]
                values["effective_weight_GBps"] = [float(row["weight_bytes_per_iteration"]) * float(row["iterations"])
                                                   / (float(row["wall_ms"]) * 1e6) for row in records]
            if any(float(row[col]) <= 0 for row in records for col in ("prefill_ms", "decode_ms", "wall_ms", "iterations") if col in row):
                raise ValueError("nonpositive measurement duration/iteration count")
            data["columns"] = {col: distribution(samples) for col, samples in values.items()}
            if "run" in fields:
                run_ids = [int(row["run"]) for row in records]
                data["run_ids"] = run_ids
                peers = control_groups[tuple(records[0][col] for col in controlled)]
                peer_ids = [int(row["run"]) for row in peers]
                if len(set(peer_ids)) != len(peer_ids) or any(run < 0 for run in peer_ids):
                    raise ValueError("duplicate/negative run indices in a requested configuration")
                expected = config.get("runs", config.get("repetitions"))
                if expected is not None and sorted(peer_ids) != list(range(int(expected))):
                    data["sample_status"] = "incomplete"
                    data["exclusion_reasons"].append(f"incomplete run indices in requested configuration; expected 0..{int(expected) - 1}")
        except (ValueError, TypeError, ZeroDivisionError, OverflowError) as e:
            data["sample_status"] = "invalid"
            data["exclusion_reasons"].append(f"invalid samples: {e}")
        data["primary_eligible"] = (role == "primary" and data["metadata_status"] == "complete"
                                    and data["sample_status"] == "complete")
        if not data["primary_eligible"] and not data["exclusion_reasons"]:
            data["exclusion_reasons"].append("measurement is not a completed primary benchmark")
        label = f"{file_label}:b{data['configuration'].get('batch', 'reference')}"
        if len(groups) > 1:
            label += f":config{index + 1}"
        result.append((label, data))
    return result


def summarize_directory(directory, recursive=False):
    measurements = {}
    paths = sorted(directory.rglob("*.csv") if recursive else directory.glob("*.csv"))
    for path in paths:
        if recursive and path.name == "report.csv":
            continue
        file_label = path.relative_to(directory).as_posix() if recursive else path.name
        try:
            measurements.update(summarize_file(path, label_prefix=file_label))
        except (OSError, ValueError, csv.Error) as e:
            measurements[file_label] = {"source": str(path), "schema": "unreadable", "metadata_status": "invalid",
                                        "primary_eligible": False, "exclusion_reasons": [str(e)], "columns": {}}
    return {
        "schema_version": 2,
        "statistical_method": "Per-file, exact-configuration samples; median, min/max, linearly interpolated p05/p95 "
                              "at (n-1)*q, IQR=Q75-Q25, unscaled MAD. No outlier removal or cross-file pooling.",
        "measurements": measurements,
        "primary_measurements": [key for key, data in measurements.items() if data["primary_eligible"]],
        "excluded_measurements": {key: data["exclusion_reasons"] for key, data in measurements.items()
                                  if not data["primary_eligible"]},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, help="default: DIRECTORY/summary.json")
    parser.add_argument("--recursive", action="store_true", help="include nested case CSVs keyed by relative path")
    args = parser.parse_args()
    if not args.directory.is_dir():
        parser.error("directory does not exist")
    output = args.output or args.directory / "summary.json"
    if output.suffix.lower() == ".csv" or output.name.endswith(".command.json"):
        parser.error("output must not replace raw CSV or command metadata")
    result = summarize_directory(args.directory, recursive=args.recursive)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for key, data in result["measurements"].items():
        columns = data["columns"]
        if "prefill_tps" in columns:
            print(key, data["metadata_status"], data["role"], data["samples"], "samples=" + data.get("sample_status", "unavailable"),
                  "pp", round(columns["prefill_tps"]["median"], 3),
                  "tg", round(columns["decode_tps"]["median"], 3) if "decode_tps" in columns else "unavailable")
    print(f"{len(result['primary_measurements'])} primary groups; {len(result['excluded_measurements'])} explicitly excluded groups")


if __name__ == "__main__":
    main()
