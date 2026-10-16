"""Lightweight workload-tool regressions: no inference, builds, GPU, sudo or perf.

    python -m unittest tools.test_workload_tools -v
"""
import csv
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_cpu_experts as cpu_bench
import collect_workload_environment as environment
import run_workload_command as runner
import summarize_workload_activity as activity
import summarize_workload_benchmarks as benchmarks


class TemporaryFiles(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix=".workload-test-", dir=Path(__file__).resolve().parent)
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def csv(self, name, rows, metadata=None, delimiter=","):
        path = self.root / name
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter=delimiter)
            writer.writeheader()
            writer.writerows(rows)
        if metadata is not None:
            path.with_suffix(".command.json").write_text(json.dumps(metadata))
        return path


class BenchmarkSummaries(TemporaryFiles):
    @staticmethod
    def main_row(**changes):
        return {"batch": 256, "run": 0, "prompt_tokens": 512, "prefill_tokens": 511,
                "generated_tokens": 128, "prefill_ms": 2000, "decode_ms": 3000, "wall_ms": 5000,
                "prefill_tps": 255.5, "decode_tps": 42.666, **changes}

    @staticmethod
    def complete(runs=1):
        return {"command": ["ornith_prefill_bench", "--runs", str(runs), "--context", "32768"], "exit_code": 0}

    def test_even_median_tails_iqr_mad_and_single_sample(self):
        summary = benchmarks.distribution([4, 1, 3, 2])
        for key, expected in {"median": 2.5, "p05": 1.15, "p95": 3.85, "iqr": 1.5, "mad": 1}.items():
            self.assertAlmostEqual(summary[key], expected)
        one = benchmarks.distribution([7])
        self.assertEqual((one["median"], one["p05"], one["p95"], one["iqr"], one["mad"]), (7, 7, 7, 0, 0))
        with self.assertRaises(ValueError):
            benchmarks.distribution([float("nan")])

    def test_configuration_dimensions_do_not_pool_context_offload_or_prompt(self):
        path = self.csv("final.csv", [
            self.main_row(context=32768, offload="dense"),
            self.main_row(context=65536, offload="dense"),
            self.main_row(context=32768, offload="cpu"),
            self.main_row(context=32768, offload="dense", prompt_tokens=2048, prefill_tokens=2047),
        ], self.complete())
        data = benchmarks.summarize_file(path)
        self.assertEqual(len(data), 4)
        self.assertTrue(all(group["samples"] == 1 and group["primary_eligible"] for _, group in data))
        self.assertTrue(all("context" not in group["columns"] and "run" not in group["columns"] for _, group in data))

    def test_oracle_derivation_has_full_dispersion_statistics(self):
        path = self.csv("oracle.csv", [
            {"run": i, "prompt": 512, "generated": 128, "prefill_tokens": 511, "decode_evals": 128,
             "prefill_ms": duration, "decode_ms": duration * 2, "wall_ms": duration * 3}
            for i, duration in enumerate([1000, 2000, 3000, 4000])
        ], {"command": ["llama_prefill_reference", "model", "ids", "32768", "128", "16", "32", "out", "--runs", "4"], "exit_code": 0})
        data = benchmarks.summarize_file(path)[0][1]
        self.assertEqual(data["role"], "oracle_reference")
        self.assertEqual(data["command_configuration"]["n_gpu_layers"], "16")
        self.assertEqual(set(data["columns"]["prefill_tps"]), {"median", "min", "max", "p05", "p95", "iqr", "mad"})
        self.assertAlmostEqual(data["columns"]["prefill_tps"]["median"], (255.5 + 511 / 3) / 2)

    def test_microbenchmark_is_not_oracle_or_layer_profile(self):
        path = self.csv("cpu.csv", [{"run": 0, "layer": 0, "groups": 256, "tokens": 8, "cores": 16,
                                     "iterations": 50, "weight_bytes_per_iteration": 1_000_000, "wall_ms": 100, "avx512bw": 1}],
                        {"command": [], "exit_code": 0})
        data = benchmarks.summarize_file(path)[0][1]
        self.assertEqual(data["schema"], "cpu_expert_microbenchmark")
        self.assertEqual(data["columns"]["effective_weight_GBps"]["median"], .5)
        self.assertEqual(data["columns"]["wall_ms_per_iteration"]["median"], 2)
        self.assertFalse(data["primary_eligible"])
        self.assertNotIn("prefill_tps", data["columns"])

    def test_profiles_are_one_explicit_exclusion_without_numeric_expansion(self):
        for name, row in [("layers.csv", {"layer": 3, "rows": 256, "gpu_dense_interval_ms": 123}),
                          ("cpu-profile.csv", {"start_ns": 100, "end_ns": 200, "jobs": 16})]:
            with self.subTest(name=name):
                path = self.csv(name, [row] * 100)
                data = benchmarks.summarize_file(path)
                self.assertEqual(len(data), 1)
                self.assertEqual(data[0][1]["columns"], {})
                self.assertFalse(data[0][1]["primary_eligible"])

    def test_invalid_baseline_incomplete_counters_manual_and_instrumentation_labels(self):
        for name, metadata, status, role in [
            ("baseline-profile.csv", self.complete(), "complete", "invalid_profile_run"),
            ("counters-avx512.csv", {"command": []}, "incomplete", "instrumented_benchmark"),
            ("baseline-b256.csv", None, "metadata_missing", "primary"),
            ("profile-final.csv", self.complete(), "complete", "instrumented_benchmark"),
            ("column32-b256.csv", self.complete(), "complete", "rejected_projection_experiment"),
        ]:
            with self.subTest(name=name):
                path = self.csv(name, [self.main_row()], metadata)
                data = benchmarks.summarize_file(path)[0][1]
                self.assertEqual(data["metadata_status"], status)
                self.assertEqual(data["role"], role)
                self.assertFalse(data["primary_eligible"])
                self.assertTrue(data["exclusion_reasons"])
        self.assertFalse((self.root / "baseline-b256.command.json").exists())
        manual = benchmarks.summarize_file(self.root / "baseline-b256.csv")[0][1]
        self.assertEqual(manual["command_configuration"], {})

    def test_rejected_projection_label_survives_diagnostic_dump(self):
        metadata = self.complete()
        metadata["command"] += ["--dump", "logits"]
        path = self.csv("paired-rows-b256.csv", [self.main_row()], metadata)
        data = benchmarks.summarize_file(path)[0][1]
        self.assertEqual(data["role"], "rejected_projection_experiment")
        self.assertTrue(any("instrumentation" in reason for reason in data["exclusion_reasons"]))

    def test_failed_wrapper_is_invalid_even_if_benchmark_exited_zero(self):
        path = self.csv("broken.csv", [self.main_row()], {**self.complete(), "state": "failed", "wrapper_exit_code": 1})
        data = benchmarks.summarize_file(path)[0][1]
        self.assertEqual(data["metadata_status"], "failed")
        self.assertFalse(data["primary_eligible"])

    def test_malformed_command_metadata_is_an_explicit_exclusion(self):
        for metadata in [{"command": [123]}, {"command": [], "environment": []}]:
            path = self.csv("malformed.csv", [self.main_row()], metadata)
            data = benchmarks.summarize_file(path)[0][1]
            self.assertEqual(data["metadata_status"], "invalid")
            self.assertFalse(data["primary_eligible"])

    def test_nonfinite_truncated_and_short_runs_excluded_without_losing_good_file(self):
        self.csv("nonfinite.csv", [self.main_row(prefill_tps="nan")], self.complete())
        self.csv("short.csv", [self.main_row()], self.complete(10))
        self.csv("good.csv", [self.main_row()], self.complete())
        (self.root / "truncated.csv").write_text("batch,prefill_tps,decode_tps\n256,1\n")
        result = benchmarks.summarize_directory(self.root)
        self.assertEqual(result["primary_measurements"], ["good.csv:b256"])
        self.assertEqual(len(result["excluded_measurements"]), 3)
        json.dumps(result, allow_nan=False)

    def test_live_matrix_prefix_remains_incomplete_not_failed(self):
        metadata = self.complete(10)
        del metadata["exit_code"]
        path = self.csv("baseline-p4096-b256.csv", [self.main_row()], metadata)
        data = benchmarks.summarize_file(path)[0][1]
        self.assertEqual(data["metadata_status"], "incomplete")
        self.assertEqual(data["sample_status"], "incomplete")
        self.assertFalse(data["primary_eligible"])

    def test_early_eos_lengths_remain_separate_without_inventing_missing_runs(self):
        path = self.csv("early-eos.csv", [self.main_row(run=0, finish="length"),
                                         self.main_row(run=1, generated_tokens=17, finish="stop")], self.complete(2))
        groups = benchmarks.summarize_file(path)
        self.assertEqual(len(groups), 2)
        self.assertTrue(all(data["samples"] == 1 and data["sample_status"] == "complete" for _, data in groups))
        self.assertTrue(all(data["primary_eligible"] for _, data in groups))


class ActivitySummaries(TemporaryFiles):
    epoch = 1_700_000_000_000_000_000
    host = 1_000_000_000

    def record(self, kind, a, b, name, size=0):
        return {"kind": kind, "start_ns": self.epoch + int(a * 1e6), "end_ns": self.epoch + int(b * 1e6),
                "bytes": size, "name": name}

    def marks(self):
        return [self.record("mark", 0, 0, "prefill", self.host), self.record("mark", 10, 10, "end", self.host + 10_000_000)]

    def trace(self, rows):
        return self.csv("gpu.tsv", rows, delimiter="\t")

    def test_unions_and_clipped_cpu_gpu_overlap_not_summed_durations(self):
        trace = self.trace(self.marks() + [
            self.record("kernel", 0, 10, "wait_flag_ge"),
            self.record("kernel", 1, 5, "mmvq_a"), self.record("kernel", 3, 7, "mmvq_b"),
            self.record("copy", 8, 9, "copy_1_2_3", 4096),
        ])
        cpu = self.csv("cpu.csv", [
            {"start_ns": self.host + int(a * 1e6), "end_ns": self.host + int(b * 1e6), "layer": layer,
             "gu_thread_cpu_ms": 2, "quant_thread_cpu_ms": 1, "down_thread_cpu_ms": 3}
            for a, b, layer in [(2, 6, 7), (8.5, 10.5, 9)]
        ])
        result = activity.summarize_trace(trace, cpu)
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["clock_alignment"]["offset_ns"], self.epoch - self.host)
        self.assertTrue(result["cpu_layer_attribution"]["verified"])
        phase = result["phases"]["prefill"]
        for key, expected in {"gpu_union_ms": 10, "gpu_useful_union_ms": 7, "gpu_useful_kernel_union_ms": 6,
                              "cpu_expert_wall_union_ms": 5.5, "cpu_expert_gpu_useful_overlap_ms": 4.5,
                              "cpu_expert_without_gpu_useful_ms": 1, "gpu_useful_without_cpu_expert_ms": 2.5,
                              "neither_cpu_expert_nor_gpu_useful_ms": 2}.items():
            self.assertAlmostEqual(phase[key], expected, msg=key)
        self.assertEqual(phase["cpu_expert_boundary_crossing_calls"], 1)
        self.assertEqual(phase["cpu_expert_gpu_useful_overlap_ms_by_calibration_delta_ns"], {0: 4.5})
        self.assertEqual(phase["cpu_expert_thread_times"]["gu_thread_cpu_ms"], 2)
        self.assertEqual(phase["memcpy_by_kind"]["host_to_device:src2:dst3"]["bytes"], 4096)

    def test_kernel_families_and_mapped_traffic_not_counted_as_memcpy_bytes(self):
        for name, expected in [("gdn_ab_multi_kernel", "gdn"), ("native_rope::apply<RopeTab>", "rope"),
                               ("copy_kernel(KvStreamMap, Runs)", "kv_staging"),
                               ("copy_rows_from_mapped_kernel", "mapped_host_transfer"), ("wait_flag_ge", "device_wait")]:
            self.assertEqual(activity.category(name), expected)
        result = activity.summarize_trace(self.trace(self.marks() + [self.record("kernel", 1, 2, "copy_rows_from_mapped_kernel")]))
        phase = result["phases"]["prefill"]
        self.assertEqual(phase["gpu_mapped_host_transfer_kernel_union_ms"], 1)
        self.assertEqual(phase["memcpy_by_kind"], {})
        self.assertIsNone(phase["cpu_expert_gpu_useful_overlap_ms"])

    def test_missing_timestamps_drops_and_terminal_mark_fail_closed(self):
        cases = [
            self.marks() + [{"kind": "kernel", "start_ns": 0, "end_ns": 0, "bytes": 0, "name": "mmvq"}],
            self.marks() + [self.record("kernel", 1, 2, "mmvq"), {"kind": "dropped", "start_ns": 0, "end_ns": 0, "bytes": 3, "name": "status_0"}],
            [self.marks()[0], self.record("kernel", 1, 2, "mmvq")],
        ]
        for rows in cases:
            with self.subTest(rows=rows):
                result = activity.summarize_trace(self.trace(rows))
                self.assertTrue(result["errors"])
                self.assertEqual(result["status"], "invalid")
                self.assertFalse(result["phases"])

    def test_calibration_missing_inconsistent_or_wrong_cpu_pairing_fails(self):
        cpu = self.csv("cpu.csv", [{"start_ns": self.host + 1_000_000, "end_ns": self.host + 2_000_000,
                                    "gu_thread_cpu_ms": 1, "quant_thread_cpu_ms": 1, "down_thread_cpu_ms": 1}])
        for delta in [None, 2_000_000, 1_000_000_000]:
            marks = self.marks()
            if delta is None:
                marks[0]["bytes"] = 0
            elif delta == 2_000_000:
                marks[1]["bytes"] += delta
            else:
                for mark in marks:
                    mark["bytes"] += delta
            result = activity.summarize_trace(self.trace(marks + [self.record("kernel", 1, 2, "mmvq")]), cpu)
            self.assertTrue(result["errors"])
            self.assertEqual(result["status"], "invalid")

    def test_union_handles_negative_and_overlapping_intervals(self):
        self.assertEqual(activity.union_ns([(-5, -1), (0, 2), (1, 3)]), 7)
        with self.assertRaises(ValueError):
            activity.union_ns([(2, 1)])

    def test_ordinal_layer_attribution_is_explicitly_unverified(self):
        cpu = self.csv("cpu.csv", [{"start_ns": 100 + i * 10, "end_ns": 105 + i * 10,
                                    "gu_thread_cpu_ms": 1, "quant_thread_cpu_ms": 1, "down_thread_cpu_ms": 1} for i in range(3)])
        rows, attribution = activity.read_cpu(cpu, 1000, 2)
        self.assertEqual([row["layer"] for row in rows], [0, 1, 0])
        self.assertFalse(attribution["verified"])


class CommandCleanup(TemporaryFiles):
    def test_only_public_workload_environment_no_credentials(self):
        env = {"GUILD_FORCE_AVX2": "1", "CUDA_VISIBLE_DEVICES": "0", "OMP_NUM_THREADS": "8", "GGML_TEST": "1",
               "HOME": "/tmp", "AWS_SECRET_ACCESS_KEY": "private", "GUILD_API_KEY": "private", "GGML_AUTH_TOKEN": "private"}
        self.assertEqual(set(runner.public_environment(env)), {"GUILD_FORCE_AVX2", "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "GGML_TEST"})
        self.assertEqual(runner.public_command(["env", "GUILD_API_KEY=private", "GUILD_FORCE_AVX2=1", "bench", "--api-key", "private"]),
                         ["env", "GUILD_API_KEY=<redacted>", "GUILD_FORCE_AVX2=1", "bench", "--api-key", "<redacted>"])

    def test_monitor_is_reaped_and_metadata_finalized_on_benchmark_popen_error(self):
        real_popen = subprocess.Popen
        monitors = []

        def launch(command, **kwargs):
            if command[0] == "sudo":
                process = real_popen([sys.executable, "-c", "import time; time.sleep(60)"], **kwargs)
                monitors.append(process)
                return process
            raise FileNotFoundError("fixture benchmark missing")

        metadata = {}
        with mock.patch.object(runner.subprocess, "Popen", side_effect=launch), mock.patch.object(runner, "capture", return_value={}):
            code = runner.execute(["missing-benchmark"], os.environ.copy(), self.root / "failed", metadata, turbostat=True)
        self.assertEqual(code, 127)
        self.assertIsNotNone(monitors[0].poll())
        self.assertEqual(metadata["state"], "failed")
        self.assertIsNone(metadata["exit_code"])
        saved = json.loads((self.root / "failed.command.json").read_text())
        self.assertIn("turbostat_exit_code", saved)
        self.assertEqual(saved["wrapper_exit_code"], 127)

    def test_privileged_cleanup_targets_only_the_created_group_noninteractively(self):
        with mock.patch.object(runner.os, "killpg", side_effect=PermissionError), \
                mock.patch.object(runner, "capture", return_value={"status": 0}) as capture:
            runner.signal_process_group(987654, signal.SIGINT)
        capture.assert_called_once_with(["sudo", "-n", "kill", "-s", "2", "--", "-987654"], timeout=5)
        with self.assertRaises(ValueError):
            runner.signal_process_group(os.getpgrp(), signal.SIGINT)

    def test_root_proc_permission_failure_does_not_invalidate_successful_child(self):
        real_read = Path.read_text

        def protected(path, *args, **kwargs):
            if str(path).startswith("/proc/"):
                raise PermissionError("fixture root perf process")
            return real_read(path, *args, **kwargs)

        metadata = {}
        with mock.patch.object(runner, "capture", return_value={}), mock.patch.object(Path, "read_text", protected):
            code = runner.execute([sys.executable, "-c", "import time; time.sleep(.05)"], os.environ.copy(), self.root / "protected", metadata)
        self.assertEqual(code, 0)
        self.assertEqual(metadata["state"], "completed")
        sample = json.loads((self.root / "protected.telemetry.jsonl").read_text().splitlines()[0])
        self.assertEqual(sample["numa_maps_error"], "PermissionError")

    def test_owned_process_group_cleanup_stops_grandchild_and_spares_other_session(self):
        sentinel = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
        child_script = ("import subprocess,sys,time; "
                        "p=subprocess.Popen([sys.executable,'-c','import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)']); "
                        "print(p.pid,flush=True); time.sleep(60)")
        child = subprocess.Popen([sys.executable, "-c", child_script], start_new_session=True, stdout=subprocess.PIPE, text=True)
        try:
            grandchild = int(child.stdout.readline())
            runner.stop_process_group(child, timeout=.2)
            self.assertIsNotNone(child.poll())
            self.assertIsNone(sentinel.poll())
            # An orphan may briefly be a zombie awaiting init; it must not run.
            for _ in range(50):
                try:
                    status = Path(f"/proc/{grandchild}/status").read_text()
                except FileNotFoundError:
                    break
                if "State:\tZ" in status:
                    break
                import time
                time.sleep(.01)
            else:
                self.fail("owned grandchild survived cleanup")
        finally:
            runner.stop_process_group(child, timeout=.2)
            child.stdout.close()
            runner.stop_process_group(sentinel, timeout=.2)

    def test_cpu_matrix_refuses_incomplete_measurement_instead_of_skipping(self):
        (self.root / "cpu-avx512-g1-c1-t1.command.json").write_text(json.dumps({"command": []}))
        with mock.patch.object(sys, "argv", ["bench_cpu_experts.py", "--output", str(self.root)]), \
                mock.patch.object(cpu_bench.subprocess, "run") as run:
            with self.assertRaisesRegex(RuntimeError, "incomplete measurement"):
                cpu_bench.main()
            run.assert_not_called()

    def test_cpu_matrix_refuses_truncated_csv_even_with_zero_exit(self):
        (self.root / "cpu-avx512-g1-c1-t1.command.json").write_text(json.dumps({"command": [], "exit_code": 0}))
        (self.root / "cpu-avx512-g1-c1-t1.csv").write_text("run,wall_ms\n0,123\n")
        with mock.patch.object(sys, "argv", ["bench_cpu_experts.py", "--output", str(self.root)]), \
                mock.patch.object(cpu_bench.subprocess, "run") as run:
            with self.assertRaisesRegex(RuntimeError, "incomplete CSV"):
                cpu_bench.main()
            run.assert_not_called()

    def test_sigterm_finalizes_metadata_and_reaps_child(self):
        calls = []

        def interrupt(_command, **_kwargs):
            calls.append(1)
            if len(calls) == 1:
                os.kill(os.getpid(), signal.SIGTERM)
            return {}

        metadata = {}
        original_handler = signal.getsignal(signal.SIGTERM)
        with mock.patch.object(runner, "capture", side_effect=interrupt):
            code = runner.execute([sys.executable, "-c", "import time; time.sleep(60)"],
                                  os.environ.copy(), self.root / "interrupted", metadata)
        self.assertEqual(code, 143)
        self.assertEqual(metadata["interrupted_signal"], signal.SIGTERM)
        self.assertEqual(metadata["state"], "failed")
        self.assertIsNotNone(metadata["exit_code"])
        self.assertEqual(signal.getsignal(signal.SIGTERM), original_handler)


class CompilerMetadata(TemporaryFiles):
    def test_flags_make_fills_missing_root_commands_and_retains_source_isa_and_response_files(self):
        (self.root / "compile_commands.json").write_text(json.dumps([
            {"file": "/src/arch/x86/quants.c", "command": "cc -O3 -c quants.c", "directory": str(self.root)}]))
        for target, source, flags in [
            ("guild_kernels_cpu", "kq_avx2.cpp", "CXX_DEFINES = \n\nCXX_FLAGS = -O3\n# Custom options: kq_avx2.cpp.o_OPTIONS = -mavx2;-mfma\n"),
            ("guild_kernels", "native_mmvq.cu", "CUDA_FLAGS = -O3 --generate-code=arch=compute_61,code=sm_61\nCUDA_INCLUDES = --options-file includes.rsp\n"),
        ]:
            folder = self.root / "CMakeFiles" / f"{target}.dir"
            folder.mkdir(parents=True)
            (folder / "build.make").write_text(f"object.o: /src/{source}\n")
            (folder / "flags.make").write_text(flags)
        (self.root / "includes.rsp").write_text("-I/include -I/ggml")
        data = environment.compiler_settings(self.root)
        self.assertEqual(len(data["compile_commands"]), 1)
        self.assertEqual(len(data["target_compile_flags"]), 2)
        self.assertTrue(any("-mavx2;-mfma" in row["raw"] for row in data["target_compile_flags"]))
        self.assertTrue(any(row["response_files"] for row in data["target_compile_flags"]))
        self.assertTrue(any(row["variables"].get("CXX_DEFINES") == "" for row in data["target_compile_flags"]))


if __name__ == "__main__":
    unittest.main()
