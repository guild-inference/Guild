#!/usr/bin/env python3
"""Inspect existing CPU objects and first-token dumps; no builds or inference."""
import array
import collections
import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/results/hybrid-workload-optimization"


def command(args):
    return subprocess.check_output(args, cwd=ROOT, text=True)


def identity(path):
    before = path.stat()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"Input changed: {path}")
    return {"path": str(path.relative_to(ROOT)), "bytes": before.st_size,
            "mtime_ns": before.st_mtime_ns, "sha256": digest}


def operands(text):
    return re.split(r",\s*(?![^()]*\))", text.split("#", 1)[0].strip())


def counts(instructions):
    hist, widths, forms = collections.Counter(), collections.Counter(), collections.Counter()
    registers = set()
    stack_read = stack_write = vector_stack_read = vector_stack_write = 0
    for ins in instructions:
        mnemonic, args = ins["mnemonic"], ins["operands"]
        hist[mnemonic] += 1
        found = re.findall(r"%([xyz])mm(\d+)", args)
        registers.update(int(n) for _, n in found)
        if found:
            width = max({"x": 128, "y": 256, "z": 512}[w] for w, _ in found)
            widths[str(width)] += 1
            if mnemonic in ["vpmaddubsw", "vpmaddwd", "vpshufb"] or mnemonic.startswith(("vfmadd", "vfmsub", "vfnmadd")):
                forms[f"{mnemonic}:max_explicit_bits={width}"] += 1
        parts = operands(args)
        def is_stack(x):
            return bool(re.search(r"\([^)]*%(?:rsp|rbp)(?:[,)]|$)", x))
        if mnemonic not in ["lea", "leaq", "leal"]:
            read = any(is_stack(x) for x in parts[:-1])
            write = bool(parts and is_stack(parts[-1]))
            if len(parts) == 1:
                read, write = write, False
            stack_read += read
            stack_write += write
            if mnemonic.startswith("v") and found:
                vector_stack_read += read
                vector_stack_write += write
    shuffle = sum(n for m, n in hist.items() if m.startswith((
        "vpshuf", "vshuf", "vperm", "vinsert", "vextract", "vunpck"))
        or m in ["vmovhlps", "vmovshdup", "vmovsldup"])
    return {
        "static_instruction_count_excluding_nops": len(instructions),
        "mnemonics": dict(sorted(hist.items())),
        "instructions_by_max_explicit_simd_operand_bits": dict(sorted(widths.items())),
        "selected_instruction_forms": dict(sorted(forms.items())),
        "architectural_simd_register_indices_referenced": sorted(registers),
        "distinct_architectural_simd_register_indices": len(registers),
        "extended_simd_register_indices_16_to_31": sorted(n for n in registers if n >= 16),
        "direct_stack_source_operand_instruction_sites": stack_read,
        "direct_stack_destination_operand_instruction_sites": stack_write,
        "vector_direct_stack_read_instruction_sites": vector_stack_read,
        "vector_direct_stack_write_instruction_sites": vector_stack_write,
        "shuffle_permute_insert_extract_instruction_sites": shuffle,
        "vpshufb_sites": hist["vpshufb"], "vpmaddubsw_sites": hist["vpmaddubsw"],
        "vpmaddwd_sites": hist["vpmaddwd"],
        "packed_fp32_fma_sites": sum(n for m, n in hist.items()
                                    if m.startswith(("vfmadd", "vfmsub", "vfnmadd", "vfnmsub")) and m.endswith("ps")),
        "vnni_dot_instruction_sites": sum(n for m, n in hist.items() if m.startswith(("vpdp", "vdp"))),
    }


def parse_disassembly(text):
    result, current = {}, None
    for line in text.splitlines():
        match = re.match(r"^([0-9a-f]+) <(.+)>:$", line)
        if match:
            current = {"address": int(match[1], 16), "name": match[2], "instructions": []}
            result[match[2]] = current
            continue
        match = re.match(r"^\s*([0-9a-f]+):\s+([A-Za-z][\w.]*)\s*(.*?)\s*$", line)
        if match and current:
            mnemonic, args = match[2], match[3]
            if mnemonic.startswith("nop") or (mnemonic in ["cs", "data16"] and "nop" in args):
                continue
            current["instructions"].append({"address": int(match[1], 16), "mnemonic": mnemonic,
                                             "operands": args, "line": line.strip()})
    return result


def assembly():
    build = ROOT / "build-cuda12"
    database = json.loads((build / "compile_commands.json").read_text())
    candidates = [entry for entry in database if Path(entry["file"]).name == "kq_avx2.cpp"
                  and "guild_kernels_cpu.dir" in entry.get("command", "")]
    if len(candidates) != 1:
        raise RuntimeError("the configured bundled CPU target must contain one kq_avx2 translation unit")
    args = shlex.split(candidates[0]["command"])
    cpu_object = Path(candidates[0]["directory"]) / args[args.index("-o") + 1]
    objects = {
        "guild": cpu_object,
        "ggml": ROOT / "build-cuda12/ggml/src/CMakeFiles/ggml-cpu.dir/ggml-cpu/arch/x86/quants.c.o",
    }
    targets = []
    for q in ["q4", "q6"]:
        for nt in [4, 8]:
            targets.append((f"{q}_dual_avx512_nt{nt}", "guild", f"{q}k_dot_2rows_avx512bw_nt<{nt}>", nt))
        targets.append((f"{q}_dual_avx2_runtime_nt", "guild", f"{q}k_dot_2rows_avx2(", "runtime 1..8"))
        targets.append((f"ggml_{q}_single_dot", "ggml", f"ggml_vec_dot_{q}_K_q8_K", "single dot"))
    before = {k: identity(v) for k, v in objects.items()}
    dumps, sizes, cmds = {}, {}, []
    for key, path in objects.items():
        args = ["objdump", "-d", "-C", "--no-show-raw-insn", str(path.relative_to(ROOT))]
        cmds.append(args)
        dumps[key] = parse_disassembly(command(args))
        sizes[key] = {}
        for line in command(["nm", "-S", "--size-sort", "-C", str(path.relative_to(ROOT))]).splitlines():
            match = re.match(r"^[0-9a-f]+ ([0-9a-f]+) \w (.+)$", line)
            if match:
                sizes[key][match[2]] = int(match[1], 16)
    flag_paths = [
        "build-cuda12/CMakeFiles/guild_kernels_cpu.dir/flags.make",
        "build-cuda12/CMakeFiles/guild_kernels.dir/flags.make",
        "build-cuda12/CMakeFiles/guild_runtime.dir/flags.make",
        "build-cuda12/CMakeFiles/cpu_expert_workload.dir/flags.make",
        "build-cuda12/CMakeFiles/ornith_prefill_bench.dir/flags.make",
        "build-cuda12/ggml/src/CMakeFiles/ggml-cpu.dir/flags.make",
    ]
    result = {
        "schema_version": 1, "utc_collected": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scope": "Existing object files only; no build, inference, GPU call, or benchmark executed.",
        "analyzer": identity(Path(__file__).resolve()),
        "objdump_version": command(["objdump", "--version"]).splitlines()[0],
        "commands": cmds, "objects": before,
        "generated_build_flags": [{"identity": identity(ROOT / p), "contents": (ROOT / p).read_text()} for p in flag_paths],
        "method": {
            "counts": "Static instruction sites across a whole function or largest backwards-branch address span; nops excluded. No execution-frequency weighting.",
            "stack": "Direct rsp/rbp memory operands excluding lea. General source/destination fields are AT&T operand positions, not scalar read/write semantics; vector read/write sites additionally need a v-prefixed mnemonic and explicit SIMD operand. Other-register stack aliases are not tracked. Scratch arrays, argument homes and spills are not distinguished.",
            "loop": "Largest direct backwards jump address span; includes alternative/guarded basic blocks and inner loops. Not a dynamic per-iteration instruction count.",
            "simd": "Maximum explicit SIMD operand width, not useful lanes, uops, utilization or memory traffic. Register indices merge xmm/ymm/zmm aliases.",
            "pressure": "Register references and stack sites do not measure peak liveness, physical rename-register occupancy, dynamic spill rate or stalls.",
            "avx2_comparison": "AVX2 dual-row functions take runtime nt; NT4/NT8 calls share one function. Whole-function footprint is not a specialized execution path.",
            "object_provenance": "Hashes identify the inspected objects. Correspondence with every older timed executable is not independently established.",
        }, "functions": {},
    }
    excerpts = ["Existing-object CPU assembly excerpts", "====================================",
                "AT&T syntax; aligned records selected from complete objdump -d -C output.",
                "Representative windows only; no full objdump retained.",
                "Static stack-memory sites are not dynamic spills or vector-utilization counters.", ""]
    for key, obj, needle, nt in targets:
        matches = [v for name, v in dumps[obj].items() if needle in name]
        if len(matches) != 1:
            raise RuntimeError((key, len(matches)))
        fn = matches[0]
        ins = fn["instructions"]
        back = []
        for item in ins:
            match = re.match(r"^([0-9a-f]+)\s+<", item["operands"])
            if item["mnemonic"].startswith("j") and match:
                target = int(match[1], 16)
                if fn["address"] <= target < item["address"]:
                    back.append((item["address"] - target, target, item["address"], item["mnemonic"]))
        loop = max(back) if back else None
        selected_loop = [i for i in ins if loop and loop[1] <= i["address"] <= loop[2]]
        frame_sub = []
        for item in ins:
            match = re.match(r"^\$0x([0-9a-f]+),%rsp$", item["operands"])
            if item["mnemonic"] == "sub" and match:
                frame_sub.append(int(match[1], 16))
        result["functions"][key] = {
            "object": obj, "symbol": fn["name"], "address_hex": hex(fn["address"]),
            "symbol_bytes": sizes[obj][fn["name"]], "token_rows": nt,
            "stack_sub_immediates_bytes": frame_sub, "whole_function": counts(ins),
            "backward_branch_count": len(back),
            "largest_backwards_branch_span": {
                "start_hex": hex(loop[1]), "branch_hex": hex(loop[2]), "branch": loop[3],
                "address_span_bytes": loop[0], "static_sites": counts(selected_loop),
            } if loop else None,
        }
        excerpts += [key, fn["name"], f"object SHA256: {before[obj]['sha256']}",
                     f"symbol bytes: {sizes[obj][fn['name']]}; stack sub immediates: {frame_sub}", ""]
        windows = [(0, min(18, len(ins)), "entry / frame setup")]
        dot = next((i for i, x in enumerate(ins) if x["mnemonic"] in ["vpmaddubsw", "vpmaddwd"]), None)
        if dot is not None:
            windows.append((max(0, dot - 3), min(len(ins), dot + 10), "integer dot/unpack representative"))
        stack_fma = next((i for i, x in enumerate(ins) if x["mnemonic"].startswith("vfm") and "%rsp)" in x["operands"]), None)
        stack_move = next((i for i, x in enumerate(ins) if x["mnemonic"].startswith("vmov") and "%rsp)" in x["operands"]), None)
        spill = stack_fma if stack_fma is not None else stack_move
        if spill is not None:
            windows.append((max(0, spill - 2), min(len(ins), spill + 7), "stack-resident SIMD data representative"))
        if loop:
            idx = next(i for i, x in enumerate(ins) if x["address"] == loop[2])
            windows.append((max(0, idx - 3), min(len(ins), idx + 2), "largest backwards branch"))
        seen = set()
        for a, b, label in windows:
            excerpts.append(f"[{label}]")
            for index in range(a, b):
                if index not in seen:
                    excerpts.append(ins[index]["line"])
                    seen.add(index)
            excerpts.append("...")
        excerpts.append("")
    if before != {k: identity(v) for k, v in objects.items()}:
        raise RuntimeError("Object changed during inspection")
    (OUT / "assembly-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / "assembly-excerpts.txt").write_text("\n".join(excerpts) + "\n")
    for name, fn in result["functions"].items():
        c = fn["whole_function"]
        print("ASSEMBLY", name, "bytes", fn["symbol_bytes"], "instructions", c["static_instruction_count_excluding_nops"],
              "frame_sub", fn["stack_sub_immediates_bytes"], "vector_stack_R/W",
              c["vector_direct_stack_read_instruction_sites"], c["vector_direct_stack_write_instruction_sites"],
              "shuffle", c["shuffle_permute_insert_extract_instruction_sites"],
              "widths", c["instructions_by_max_explicit_simd_operand_bits"],
              "regs", c["distinct_architectural_simd_register_indices"])


def numerical():
    names = ["baseline", "batch-projections", "serial-projections", "forced-avx2",
             "host-pinned", "expert32", "paired-rows", "llama-cpu-only", "llama-ngl16", "llama-dense-gpu"]
    records, arrays, sequences = {}, {}, {}
    for name in names:
        f32, ids = OUT / (name + ".f32"), OUT / (name + ".ids")
        before_f32, before_ids = identity(f32), identity(ids)
        raw = f32.read_bytes()
        if len(raw) % 4:
            raise RuntimeError(f"Not float32 data: {f32}")
        values = array.array("f")
        assert values.itemsize == 4
        values.frombytes(raw)
        if sys.byteorder != "little":
            values.byteswap()
        finite = [i for i, x in enumerate(values) if math.isfinite(x)]
        top1 = max(finite, key=values.__getitem__) if finite else None
        seq = [int(x) for x in ids.read_text().split()]
        if before_f32 != identity(f32) or before_ids != identity(ids):
            raise RuntimeError(f"Dump changed during comparison: {name}")
        arrays[name], sequences[name] = values, seq
        records[name] = {"logits": before_f32, "ids": before_ids, "float32_count": len(values),
                         "token_id_count": len(seq), "nan_count": sum(math.isnan(x) for x in values),
                         "positive_infinity_count": sum(x == math.inf for x in values),
                         "negative_infinity_count": sum(x == -math.inf for x in values),
                         "nonfinite_count": len(values) - len(finite), "top1_token_id": top1}
        assert len(values) == 248320 and len(seq) == 128, (name, len(values), len(seq))
    reference = arrays["baseline"]
    reference_norm = math.sqrt(math.fsum(float(x) * float(x) for x in reference))
    comparisons = []
    for name in names[1:]:
        candidate = arrays[name]
        if records[name]["nonfinite_count"] or records["baseline"]["nonfinite_count"]:
            raise RuntimeError("Nonfinite logits: ordinary error metrics are invalid")
        delta = [float(y) - float(x) for x, y in zip(reference, candidate)]
        squared = math.fsum(x * x for x in delta)
        mismatch = [i for i, (x, y) in enumerate(zip(sequences["baseline"], sequences[name])) if x != y]
        rec = {"reference": "baseline", "candidate": name,
               "logits": {"count": len(reference),
                          "bitwise_equal": records["baseline"]["logits"]["sha256"] == records[name]["logits"]["sha256"],
                          "numeric_equal_count": sum(x == y for x, y in zip(reference, candidate)),
                          "numeric_different_count": sum(x != y for x, y in zip(reference, candidate)),
                          "nonfinite_reference_count": records["baseline"]["nonfinite_count"],
                          "nonfinite_candidate_count": records[name]["nonfinite_count"],
                          "max_abs": max(abs(x) for x in delta),
                          "relative_l2_reference_denominator": math.sqrt(squared) / reference_norm,
                          "reference_l2_norm": reference_norm,
                          "reference_top1_token_id": records["baseline"]["top1_token_id"],
                          "candidate_top1_token_id": records[name]["top1_token_id"],
                          "top1_equal": records["baseline"]["top1_token_id"] == records[name]["top1_token_id"]},
               "generated_ids": {"reference_count": len(sequences["baseline"]), "candidate_count": len(sequences[name]),
                                 "matching_positions": 128 - len(mismatch), "mismatching_positions_count": len(mismatch),
                                 "first_mismatch_position_zero_based": mismatch[0] if mismatch else None,
                                 "identical_prefix_length": mismatch[0] if mismatch else 128,
                                 "mismatch_positions_zero_based": mismatch, "all_equal": not mismatch}}
        comparisons.append(rec)
        print("NUMERICAL", name, "max_abs", rec["logits"]["max_abs"], "relative_l2",
              rec["logits"]["relative_l2_reference_denominator"], "equal_logits", rec["logits"]["numeric_equal_count"],
              "top1", records[name]["top1_token_id"], "matching_ids", 128 - len(mismatch),
              "first_mismatch_position", mismatch[0] if mismatch else None)
    result = {
        "schema_version": 1, "utc_collected": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scope": "Independent stdlib array/math comparison of existing first-callback logits and numeric IDs; no inference, build or token decoding.",
        "analyzer": identity(Path(__file__).resolve()),
        "binary_format": "little-endian IEEE754 binary32; errors accumulated with Python binary64 and math.fsum",
        "relative_l2_definition": "sqrt(sum((candidate-reference)^2)) / sqrt(sum(reference^2)); reference is Guild baseline",
        "top1_definition": "Lowest token index attaining largest finite logit; numeric token ID only",
        "token_position_convention": "Zero-based output positions. Null first mismatch means all 128 IDs match. No decoded text or raw continuation-ID arrays emitted.",
        "provenance_notes": [
            "Guild --dump copies logits at its first callback; later runs overwrite dump files. IDs come from the last dumped request; the bench checks sequence consistency across its runs.",
            "llama_prefill_reference dumps first-token logits on measured run 0; IDs are rewritten each run. Cross-run oracle sequence consistency is not independently established by this file comparison.",
            "Equal GGUF hash does not imply equal weight representation; see weight-audit.stdout.txt for BF16 alpha/beta rounding.",
            "These dumps predate pending corrected rebuild/final matrix and do not validate later binaries.",
        ], "inputs": records, "comparisons": comparisons,
    }
    (OUT / "numerical-comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    if not OUT.is_dir():
        raise RuntimeError("Existing results parent directory required")
    assembly()
    numerical()
