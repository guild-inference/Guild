#!/usr/bin/env python3
import json
import os
import re
import subprocess
import sys
import time

ORNITH_GGUF = "/home/ubuntu/models/ornith-1.5-35b/Ornith-1.5-35B-Q4_K_M.gguf"
BENCH_BIN = "./build-cuda12/ornith_bench"
RESULTS_DIR = "bench/results/ornith-baseline"
CSV_PATH = f"{RESULTS_DIR}/context_scaling.csv"
THREAD_CSV_PATH = f"{RESULTS_DIR}/numa_threads.csv"
LLAMA_CSV_PATH = f"{RESULTS_DIR}/llama_comparison.csv"
JSON_PATH = f"{RESULTS_DIR}/benchmark_results.json"

def run_cmd(cmd, env=None, timeout=600):
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    print(f"Executing: {' '.join(cmd)}")
    start = time.time()
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=full_env, timeout=timeout)
    elapsed = time.time() - start
    return p.returncode, p.stdout, elapsed

def parse_bench_output(output):
    res = {}
    for line in output.splitlines():
        line = line.strip()
        m = re.match(r"Median Prompt Tok/s:\s+([0-9.]+)\s+tok/s", line)
        if m: res["prompt_tok_s"] = float(m.group(1))
        m = re.match(r"Median TTFT:\s+([0-9.]+)\s+ms", line)
        if m: res["ttft_ms"] = float(m.group(1))
        m = re.match(r"Median Decode Tok/s:\s+([0-9.]+)\s+tok/s", line)
        if m: res["decode_tok_s"] = float(m.group(1))
        m = re.match(r"Est\. RAM Bandwidth:\s+([0-9.]+)\s+GiB/s", line)
        if m: res["ram_bandwidth_gib_s"] = float(m.group(1))
        m = re.match(r"Peak VRAM Used:\s+([0-9.]+)\s+MiB", line)
        if m: res["peak_vram_mib"] = float(m.group(1))
        m = re.match(r"Peak Host RAM \(RSS\):\s+([0-9.]+)\s+GiB", line)
        if m: res["peak_ram_gib"] = float(m.group(1))
        m = re.match(r"KV Footprint \(Host\):\s+([0-9.]+)\s+MiB", line)
        if m: res["kv_host_mib"] = float(m.group(1))
        m = re.match(r"KV Staging \(GPU VRAM\):\s+([0-9.]+)\s+MiB", line)
        if m: res["kv_staging_mib"] = float(m.group(1))
        m = re.match(r"RAM Expert Hits:\s+([0-9]+)", line)
        if m: res["ram_expert_hits"] = int(m.group(1))
        m = re.match(r"File Expert Reads:\s+([0-9]+)", line)
        if m: res["file_expert_reads"] = int(m.group(1))
        m = re.match(r"GPU Expert Cache Hits:\s+([0-9]+)", line)
        if m: res["gpu_cache_hits"] = int(m.group(1))
    return res

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_results = {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "date": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "model": "Ornith-1.5-35B-Q4_K_M.gguf",
        "hardware": {
            "gpu": "NVIDIA GeForce GTX 1070 8GB (SM 6.1)",
            "cpu": "Dual Intel Xeon Gold 6130 (32C/64T, 2.10 GHz)",
            "ram": "377 GiB DDR4-2666 (2 NUMA nodes)"
        },
        "context_scaling": [],
        "numa_threads": [],
        "llama_cpp": []
    }

    # =========================================================================
    # Part 1: Context Length Sweep
    # =========================================================================
    print("\n=======================================================")
    print("      PART 1: CONTEXT LENGTH SCALING SWEEP")
    print("=======================================================\n")

    # Context configurations: (target_context, prompt_len, gen_tokens, runs)
    # Using 128 generated tokens across all runs
    context_configs = [
        (512, 384, 128, 2),
        (2048, 1920, 128, 2),
        (2052, 1924, 128, 2),
        (4096, 3968, 128, 2),
        (8192, 4096, 128, 2),
        (16384, 4096, 128, 2),
        (32768, 4096, 128, 2),
    ]

    with open(CSV_PATH, "w") as f:
        f.write("context,prompt_tokens,gen_tokens,prompt_tok_s,ttft_ms,decode_tok_s,ram_bw_gib_s,vram_mib,ram_gib,kv_host_mib,kv_gpu_mib\n")

    for ctx, p_len, g_len, runs in context_configs:
        print(f"\n>>> Running Context = {ctx}, Prompt = {p_len}, Gen = {g_len} <<<")
        cmd = [
            BENCH_BIN,
            "--context", str(ctx),
            "--prompt-len", str(p_len),
            "--gen-tokens", str(g_len),
            "--runs", str(runs),
            "--csv", CSV_PATH
        ]
        code, out, elapsed = run_cmd(cmd, timeout=900)
        print(out)
        if code != 0:
            print(f"FAILED on context {ctx}!")
            sys.exit(code)

        parsed = parse_bench_output(out)
        parsed["context"] = ctx
        parsed["prompt_tokens"] = p_len
        parsed["gen_tokens"] = g_len
        parsed["runs"] = runs
        parsed["wall_elapsed_s"] = elapsed
        all_results["context_scaling"].append(parsed)

    # =========================================================================
    # Part 2: NUMA & CPU Thread Sweep
    # =========================================================================
    print("\n=======================================================")
    print("      PART 2: NUMA & CPU WORKER THREAD SWEEP")
    print("=======================================================\n")

    # Thread configurations: (num_workers, numa_prefix)
    thread_configs = [
        (8, []),
        (16, []),
        (24, []),
        (31, []), # Default (32 physical cores - 1 host thread)
        (32, []),
        (48, []),
        (64, []),
        (16, ["numactl", "--cpunodebind=0", "--membind=0"]),     # Socket 0 local
        (31, ["numactl", "--interleave=0,1"]),                    # Interleaved memory
    ]

    with open(THREAD_CSV_PATH, "w") as f:
        f.write("workers,numa_config,decode_tok_s,prompt_tok_s,ttft_ms,ram_bw_gib_s\n")

    for workers, numa_cmd in thread_configs:
        numa_label = "default"
        if "membind=0" in " ".join(numa_cmd):
            numa_label = "socket0_local"
        elif "interleave" in " ".join(numa_cmd):
            numa_label = "interleaved"

        print(f"\n>>> Testing Workers = {workers}, NUMA = {numa_label} <<<")
        cmd = numa_cmd + [
            BENCH_BIN,
            "--context", "2048",
            "--prompt-len", "32",
            "--gen-tokens", "128",
            "--runs", "2"
        ]
        env = {"GUILD_POOL_WORKERS": str(workers)}
        code, out, elapsed = run_cmd(cmd, env=env, timeout=300)
        if code != 0:
            print(f"Worker sweep failed on {workers} workers!")
            continue

        parsed = parse_bench_output(out)
        parsed["workers"] = workers
        parsed["numa_config"] = numa_label
        all_results["numa_threads"].append(parsed)

        with open(THREAD_CSV_PATH, "a") as f:
            f.write(f"{workers},{numa_label},{parsed.get('decode_tok_s', 0)},{parsed.get('prompt_tok_s', 0)},{parsed.get('ttft_ms', 0)},{parsed.get('ram_bandwidth_gib_s', 0)}\n")
        print(f"Workers {workers} ({numa_label}): Decode = {parsed.get('decode_tok_s')} tok/s, Prompt = {parsed.get('prompt_tok_s')} tok/s")

    # =========================================================================
    # Part 3: llama.cpp Baseline Comparison
    # =========================================================================
    print("\n=======================================================")
    print("      PART 3: LLAMA.CPP BASELINE COMPARISON")
    print("=======================================================\n")

    LLAMA_BENCH = "/home/ubuntu/llama.cpp/build/bin/llama-bench"
    with open(LLAMA_CSV_PATH, "w") as f:
        f.write("backend,threads,ngl,pp512_tok_s,pp2048_tok_s,pp4096_tok_s,tg8_tok_s,tg128_tok_s\n")

    for threads in [16, 32, 64]:
        print(f"\n>>> Running llama-bench: threads={threads}, ngl=0 (CPU) <<<")
        cmd = [
            LLAMA_BENCH,
            "-m", ORNITH_GGUF,
            "-p", "512,2048,4096",
            "-n", "16",
            "-t", str(threads),
            "-ngl", "0"
        ]
        code, out, elapsed = run_cmd(cmd, timeout=600)
        print(out)
        res = {"threads": threads, "ngl": 0, "raw_output": out}
        for line in out.splitlines():
            if "pp512" in line:
                m = re.search(r"pp512\s+\|\s+([0-9.]+)", line)
                if m: res["pp512_tok_s"] = float(m.group(1))
            elif "pp2048" in line:
                m = re.search(r"pp2048\s+\|\s+([0-9.]+)", line)
                if m: res["pp2048_tok_s"] = float(m.group(1))
            elif "pp4096" in line:
                m = re.search(r"pp4096\s+\|\s+([0-9.]+)", line)
                if m: res["pp4096_tok_s"] = float(m.group(1))
            elif "tg" in line:
                m = re.search(r"tg\d+\s+\|\s+([0-9.]+)", line)
                if m: res["tg_tok_s"] = float(m.group(1))
        all_results["llama_cpp"].append(res)
        with open(LLAMA_CSV_PATH, "a") as f:
            f.write(f"CPU,{threads},0,{res.get('pp512_tok_s',0)},{res.get('pp2048_tok_s',0)},{res.get('pp4096_tok_s',0)},{res.get('tg_tok_s',0)},{res.get('tg_tok_s',0)}\n")

    # Evaluate llama.cpp partial GPU offloading
    print("\n>>> Testing llama-bench with partial GPU offload (ngl=4 layers) <<<")
    cmd = [
        LLAMA_BENCH,
        "-m", ORNITH_GGUF,
        "-p", "512,2048",
        "-n", "8",
        "-t", "16",
        "-ngl", "4"
    ]
    code, out, elapsed = run_cmd(cmd, timeout=300)
    print(out)
    all_results["llama_cpp_gpu_offload_result"] = {
        "ngl": 4,
        "return_code": code,
        "output": out
    }

    # Save full JSON
    with open(JSON_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll benchmark results saved to {JSON_PATH} and {CSV_PATH}")

if __name__ == "__main__":
    main()
