#!/usr/bin/env python3
"""Summarize validated CUPTI/CPU intervals using unions, not summed overlap.

Activity/profile timings are instrumentation-only. Memcpy bytes exclude mapped
host-memory traffic performed by kernels and are not a PCIe bandwidth counter.
"""
import argparse
import csv
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path
import re
import subprocess
import zlib


def open_text(path):
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


CPU_TIMES = ("gu_thread_cpu_ms", "quant_thread_cpu_ms", "down_thread_cpu_ms")
COPY_KINDS = {1: "host_to_device", 2: "device_to_host", 3: "host_to_array", 4: "array_to_host",
              5: "array_to_array", 6: "array_to_device", 7: "device_to_array", 8: "device_to_device",
              9: "host_to_host", 10: "peer_to_peer"}


def union_ns(intervals):
    end = None
    total = 0
    for a, b in sorted(intervals):
        if b < a:
            raise ValueError("reversed interval")
        total += b - a if end is None else max(0, b - max(a, end))
        end = b if end is None else max(end, b)
    return total


def category(name):
    name = name.lower()
    if any(x in name for x in ("wait_flag_ge", "wait_", "spin", "poll_")):
        return "device_wait"
    if "mapped" in name and "copy" in name:
        return "mapped_host_transfer"
    if any(x in name for x in ("kvstreammap", "kv_stream", "kv_", "gather")):
        return "kv_staging"
    if "rope" in name:
        return "rope"
    # Check the operation family before generic projection names/signatures.
    if "gdn" in name or "conv" in name:
        return "gdn"
    if "attn" in name or "softmax" in name or "attention" in name:
        return "attention"
    if "router" in name or "top10" in name:
        return "routing"
    if "norm" in name:
        return "normalization"
    if any(x in name for x in ("mmvq", "mmvf", "gemv", "gemm")):
        return "projection"
    if "quant" in name:
        return "activation_quantization"
    if "embed" in name:
        return "embedding"
    return "residual_and_other"


def read_trace(path):
    records, errors = [], []
    with open_text(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        if not {"kind", "start_ns", "end_ns", "bytes", "name"} <= set(reader.fieldnames or []):
            return [], [{"error": "trace lacks required timestamp/kind/bytes/name columns"}]
        for line, row in enumerate(reader, 2):
            try:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError("truncated/malformed trace row")
                kind = row["kind"]
                if kind in ("error", "dropped"):
                    errors.append({"line": line, **row})
                    continue
                if kind not in ("kernel", "copy", "api", "mark"):
                    raise ValueError(f"unsupported activity kind {kind!r}")
                start, end, size = int(row["start_ns"]), int(row["end_ns"]), int(row["bytes"])
                if start <= 0 or end < start or size < 0 or not row["name"]:
                    raise ValueError("missing/invalid timestamp, bytes or name")
                if kind == "mark" and start != end:
                    raise ValueError("phase mark must be a point timestamp")
                if kind != "mark" and end == start:
                    raise ValueError("activity has no duration/timestamp")
                records.append({"kind": kind, "start_ns": start, "end_ns": end, "bytes": size, "name": row["name"]})
            except (ValueError, TypeError) as e:
                errors.append({"line": line, "error": str(e)})
    return records, errors


def read_cpu(path, offset, layers):
    rows = []
    with open_text(path) as f:
        reader = csv.DictReader(f)
        if not {"start_ns", "end_ns", *CPU_TIMES} <= set(reader.fieldnames or []):
            raise ValueError("CPU profile lacks required timestamps/thread CPU times")
        explicit_layers = "layer" in reader.fieldnames
        for line, row in enumerate(reader, 2):
            try:
                start, end = int(row["start_ns"]), int(row["end_ns"])
                times = {key: float(row[key]) for key in CPU_TIMES}
                if (None in row or any(value is None for value in row.values()) or start <= 0 or end <= start
                        or any(not math.isfinite(x) or x < 0 for x in times.values())):
                    raise ValueError("invalid CPU timestamp/thread time or truncated row")
                layer = int(row["layer"]) if explicit_layers else None
                if layer is not None and layer < 0:
                    raise ValueError("negative CPU layer index")
                rows.append({"start_ns": start + offset, "end_ns": end + offset, **times, "layer": layer})
            except (ValueError, TypeError) as e:
                raise ValueError(f"CPU line {line}: {e}") from e
    rows.sort(key=lambda row: (row["start_ns"], row["end_ns"]))
    for ordinal, row in enumerate(rows):
        if not explicit_layers:
            row["layer"] = ordinal % layers
    if not rows:
        raise ValueError("CPU profile has no timestamped calls")
    attribution = {"method": "explicit_layer_column" if explicit_layers else "global_call_ordinal_modulo",
                   "layers": None if explicit_layers else layers, "verified": explicit_layers}
    if not explicit_layers:
        attribution["note"] = ("Assumes one call per layer in a complete sequential CPU log, including warmup. "
                               "Layer IDs are inferred, not recorded; interval overlap does not depend on this assumption.")
    return rows, attribution


def demangle(records):
    original = sorted({row["name"] for row in records if row["kind"] == "kernel"})
    try:
        converted = subprocess.check_output(["c++filt"], input="\n".join(original) + "\n", text=True, timeout=30).splitlines()
        if len(original) != len(converted):
            raise ValueError("c++filt returned the wrong number of names")
        return dict(zip(original, converted)), []
    except (OSError, subprocess.SubprocessError, ValueError) as e:
        return {}, [f"kernel demangling unavailable: {e}; using original names"]


def summarize_phase(records, cpu, start, end, names, clock_offset_deltas=(0,)):
    groups = defaultdict(lambda: {"count": 0, "sum_ms": 0., "bytes": 0, "boundary_crossing_count": 0})
    gpu, useful, kernels, copies, waits, mapped = [], [], [], [], [], []
    category_intervals = defaultdict(list)
    copy_kinds = defaultdict(lambda: {"count": 0, "bytes": 0, "sum_ms": 0.})
    for row in records:
        a, b = row["start_ns"], row["end_ns"]
        kind = row["kind"]
        if kind == "mark" or b <= start or a >= end:
            continue
        clipped = max(a, start), min(b, end)
        name = names.get(row["name"], row["name"])
        group = groups[kind + ":" + name]
        group["count"] += 1
        group["sum_ms"] += (clipped[1] - clipped[0]) / 1e6
        group["boundary_crossing_count"] += int(a < start or b > end)
        # Attribute each copy's full bytes to its start phase, never twice.
        if start <= a < end:
            group["bytes"] += row["bytes"]
        if kind in ("kernel", "copy"):
            gpu.append(clipped)
            cat = category(name) if kind == "kernel" else "memcpy"
            category_intervals[cat].append(clipped)
            if cat != "device_wait":
                useful.append(clipped)
                (kernels if kind == "kernel" else copies).append(clipped)
            else:
                waits.append(clipped)
            if cat == "mapped_host_transfer":
                mapped.append(clipped)
            if kind == "copy":
                match = re.fullmatch(r"copy_(\d+)_(\d+)_(\d+)", name)
                label = name
                if match:
                    copy_kind, src, dst = map(int, match.groups())
                    label = f"{COPY_KINDS.get(copy_kind, 'unknown_' + str(copy_kind))}:src{src}:dst{dst}"
                copy = copy_kinds[label]
                copy["count"] += 1
                copy["sum_ms"] += (clipped[1] - clipped[0]) / 1e6
                if start <= a < end:
                    copy["bytes"] += row["bytes"]
    categories = defaultdict(float)
    for name, group in groups.items():
        if name.startswith("kernel:"):
            categories[category(name[7:])] += group["sum_ms"]
    result = {
        "wall_ms": (end - start) / 1e6,
        "gpu_union_ms": union_ns(gpu) / 1e6,
        "gpu_useful_union_ms": union_ns(useful) / 1e6,
        "gpu_useful_kernel_union_ms": union_ns(kernels) / 1e6,
        "gpu_memcpy_union_ms": union_ns(copies) / 1e6,
        "gpu_device_wait_union_ms": union_ns(waits) / 1e6,
        "gpu_mapped_host_transfer_kernel_union_ms": union_ns(mapped) / 1e6,
        "gpu_category_union_ms": {key: union_ns(value) / 1e6 for key, value in category_intervals.items()},
        "kernel_category_sum_ms": dict(categories),
        "memcpy_by_kind": dict(copy_kinds),
        "activities": dict(sorted(groups.items(), key=lambda group: -group[1]["sum_ms"])),
    }
    if cpu is None:
        result.update(cpu_expert_wall_union_ms=None, cpu_expert_gpu_useful_overlap_ms=None)
        return result
    rows = [row for row in cpu if row["start_ns"] < end and row["end_ns"] > start]
    intervals = [(max(start, row["start_ns"]), min(end, row["end_ns"])) for row in rows]
    cpu_ns, useful_ns = union_ns(intervals), union_ns(useful)
    overlap = cpu_ns + useful_ns - union_ns(intervals + useful)
    sensitivity = {}
    for delta in sorted(set(clock_offset_deltas)):
        shifted = [(max(start, row["start_ns"] + delta), min(end, row["end_ns"] + delta))
                   for row in cpu if row["start_ns"] + delta < end and row["end_ns"] + delta > start]
        sensitivity[delta] = (union_ns(shifted) + useful_ns - union_ns(shifted + useful)) / 1e6
    contained = [row for row in rows if start <= row["start_ns"] and row["end_ns"] <= end]
    times = {key: sum(row[key] for row in contained) for key in CPU_TIMES}
    per_layer = defaultdict(lambda: {"calls": 0, "wall_ms": 0., **{key: 0. for key in CPU_TIMES}})
    for row in rows:
        layer = per_layer[row["layer"]]
        layer["calls"] += 1
        layer["wall_ms"] += (min(end, row["end_ns"]) - max(start, row["start_ns"])) / 1e6
        if start <= row["start_ns"] and row["end_ns"] <= end:
            for key in CPU_TIMES:
                layer[key] += row[key]
    result.update(
        cpu_expert_wall_union_ms=cpu_ns / 1e6,
        cpu_expert_gpu_useful_overlap_ms=overlap / 1e6,
        cpu_expert_gpu_useful_overlap_ms_by_calibration_delta_ns=sensitivity,
        cpu_expert_without_gpu_useful_ms=(cpu_ns - overlap) / 1e6,
        gpu_useful_without_cpu_expert_ms=(useful_ns - overlap) / 1e6,
        neither_cpu_expert_nor_gpu_useful_ms=(end - start - union_ns(intervals + useful)) / 1e6,
        cpu_expert_calls=len(rows), cpu_expert_boundary_crossing_calls=len(rows) - len(contained),
        cpu_expert_thread_times=times, cpu_expert_layers=dict(per_layer),
    )
    return result


def summarize_trace(trace, cpu_path=None, cpu_layers=40, clock_tolerance_ns=1_000_000):
    result = {"trace": str(trace), "cpu": str(cpu_path) if cpu_path else None,
              "measurement_role": "instrumentation_only", "status": "invalid", "errors": [], "warnings": [], "phases": {},
              "notes": ["GPU useful intervals exclude device wait/spin/poll kernels, including wait_flag_ge.",
                        "CPU wall and GPU useful overlap are unions of clipped timestamp intervals. Thread CPU times are sums across workers, not wall time.",
                        "Thread CPU times include only calls wholly inside a phase; boundary-crossing calls cannot be split reliably.",
                        "Memcpy bytes are attributed to the activity start phase; mapped host-memory kernel traffic is not counted in these bytes.",
                        "Category sums/unions may overlap each other; only the overall unions are additive with their complements.",
                        "Overlap at observed calibration offsets reports clock-alignment sensitivity, not a statistical confidence interval or guaranteed error bound.",
                        "Activity/profile instrumentation was observed to slow these runs by about 20%; these timings are not primary throughput."]}
    try:
        records, errors = read_trace(trace)
        result["errors"].extend(errors)
        marks = sorted((row for row in records if row["kind"] == "mark"), key=lambda row: row["start_ns"])
        if len(marks) < 2 or marks[-1]["name"] != "end":
            result["errors"].append({"error": "trace needs phase marks and a terminal end mark"})
        elif any(a["start_ns"] >= b["start_ns"] for a, b in zip(marks, marks[1:])):
            result["errors"].append({"error": "phase marks have duplicate/reversed timestamps"})
        if not any(row["kind"] in ("kernel", "copy") for row in records):
            result["errors"].append({"error": "trace has no timestamped GPU activities"})
        if result["errors"]:
            return result
        cpu = None
        clock_offset_deltas = (0,)
        if cpu_path:
            if cpu_layers < 1 or clock_tolerance_ns < 0:
                raise ValueError("CPU layer count must be positive and clock tolerance nonnegative")
            if any(mark["bytes"] <= 0 for mark in marks):
                raise ValueError("trace lacks host/CUPTI clock calibration in mark.bytes")
            offsets = sorted(mark["start_ns"] - mark["bytes"] for mark in marks)
            # Integer nanoseconds: floating-point arithmetic loses precision at CUPTI epochs.
            low, high = offsets[(len(offsets) - 1) // 2], offsets[len(offsets) // 2]
            offset = low + (high - low) // 2
            spread = offsets[-1] - offsets[0]
            result["clock_alignment"] = {"method": "median_CUPTI_minus_host_mark_ns", "offset_ns": offset,
                                         "calibration_points": len(offsets), "offset_spread_ns": spread,
                                         "offset_min_ns": offsets[0], "offset_max_ns": offsets[-1],
                                         "tolerance_ns": clock_tolerance_ns}
            if spread > clock_tolerance_ns:
                raise ValueError("host/CUPTI calibration offsets disagree beyond tolerance")
            cpu, attribution = read_cpu(cpu_path, offset, cpu_layers)
            clock_offset_deltas = tuple(value - offset for value in offsets) + (0,)
            result["cpu_layer_attribution"] = attribution
            first, last = marks[0]["start_ns"], marks[-1]["start_ns"]
            result["cpu_calls_outside_marked_window"] = sum(row["end_ns"] <= first or row["start_ns"] >= last for row in cpu)
            if not any(row["start_ns"] < last and row["end_ns"] > first for row in cpu):
                raise ValueError("aligned CPU calls do not intersect the GPU phase window; check trace pairing/clocks")
        names, warnings = demangle(records)
        result["warnings"].extend(warnings)
        phase_counts = defaultdict(int)
        for mark, following in zip(marks, marks[1:]):
            phase = mark["name"]
            phase_counts[phase] += 1
            key = phase if phase_counts[phase] == 1 else f"{phase}#{phase_counts[phase]}"
            result["phases"][key] = summarize_phase(records, cpu, mark["start_ns"], following["start_ns"], names, clock_offset_deltas)
        result["status"] = "complete"
    except (OSError, EOFError, zlib.error, UnicodeDecodeError, ValueError, csv.Error) as e:
        result["errors"].append({"error": f"{type(e).__name__}: {e}"})
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("trace", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cpu", type=Path)
    p.add_argument("--cpu-layers", type=int, default=40, help="ordinal diagnostic layer count when CPU CSV lacks layer IDs")
    p.add_argument("--clock-tolerance-ns", type=int, default=1_000_000)
    args = p.parse_args()
    if args.output.resolve() in {path.resolve() for path in (args.trace, args.cpu) if path}:
        p.error("output must not replace an input trace")
    result = summarize_trace(args.trace, args.cpu, args.cpu_layers, args.clock_tolerance_ns)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if result["errors"]:
        raise SystemExit("incomplete/invalid activity trace; see errors in JSON")
    for phase, data in result["phases"].items():
        print(phase, {key: value for key, value in data.items()
                      if key not in ("activities", "cpu_expert_layers", "memcpy_by_kind")})
        for name, value in list(data["activities"].items())[:12]:
            print(f"  {value['sum_ms']:.3f}ms {value['count']} calls {name[:150]}")


if __name__ == "__main__":
    main()
