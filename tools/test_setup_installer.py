#!/usr/bin/env python3
"""Tests for install.sh hardware detection, piped execution, integrity verification, update, and rollback."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
INSTALL_SH = ROOT / "install.sh"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def make_mock_release(staging_root: Path, tag: str, variant_id: str, archive_name: str, version_text: str):
    pkg_dir = staging_root / f"guild-{variant_id}"
    pkg_dir.mkdir(parents=True, exist_ok=True)
    bin_file = pkg_dir / "guild.bin"
    bin_file.write_text(
        f'#!/bin/sh\nif [ "$1" = "--version" ]; then\n  echo "Guild {version_text}"\n  exit 0\nfi\n'
        f'if [ "$1" = "--help" ]; then\n  echo "Usage: guild [command]"\n  exit 0\nfi\n'
        f'echo "Guild mock {version_text}"\n'
    )
    bin_file.chmod(0o755)
    launcher = pkg_dir / "guild"
    launcher.write_text(
        '#!/bin/sh\nset -eu\nHERE=$(dirname "$(readlink -f "$0")")\nexec "$HERE/guild.bin" "$@"\n'
    )
    launcher.chmod(0o755)

    exec_sha = sha256_file(bin_file)
    archive_path = staging_root / archive_name
    with tarfile.open(archive_path, "w:gz") as tf:
        tf.add(pkg_dir, arcname=pkg_dir.name)
    arch_sha = sha256_file(archive_path)
    (staging_root / f"{archive_name}.sha256").write_text(f"{arch_sha}  {archive_name}\n")

    manifest = {
        "version": "0.2.0",
        "software_version": version_text.lstrip("v"),
        "display_version": version_text,
        "release_tag": tag,
        "channel": "preview",
        "executable_sha256": exec_sha,
        "variants": {
            variant_id: {
                "variant_id": variant_id,
                "archive": archive_name,
                "sha256": arch_sha,
                "executable_sha256": exec_sha,
            }
        },
    }
    (staging_root / "guild-release.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return arch_sha, exec_sha


class InstallerTest(unittest.TestCase):
    def test_hardware_detection_matrix(self):
        base_env = dict(
            os.environ,
            GUILD_MOCK_OS="Linux",
            GUILD_MOCK_ARCH="x86_64",
            GUILD_MOCK_CPU_FLAGS="fpu sse4_2 avx avx2 fma f16c bmi2 avx512f",
        )

        # 1. Pascal GTX 1070 (sm_61, driver 580.178.04) -> linux-x86_64-cuda12-sm61
        env_pascal = dict(
            base_env,
            GUILD_MOCK_GPU_NAME="NVIDIA GeForce GTX 1070",
            GUILD_MOCK_GPU_CC="6.1",
            GUILD_MOCK_GPU_DRIVER="580.178.04",
        )
        out = subprocess.check_output(["bash", str(INSTALL_SH), "--dry-run"], env=env_pascal, text=True)
        info = json.loads(out.splitlines()[-1])
        self.assertEqual(info["variant"], "linux-x86_64-cuda12-sm61")
        self.assertEqual(info["archive"], "guild-linux-x86_64-cuda12-sm61.tar.gz")

        # 2. Blackwell RTX 5070 (sm_120, driver 575.10) -> linux-x86_64-cuda13
        env_blackwell = dict(
            base_env,
            GUILD_MOCK_GPU_NAME="NVIDIA GeForce RTX 5070",
            GUILD_MOCK_GPU_CC="12.0",
            GUILD_MOCK_GPU_DRIVER="575.10",
        )
        out = subprocess.check_output(["bash", str(INSTALL_SH), "--dry-run"], env=env_blackwell, text=True)
        info = json.loads(out.splitlines()[-1])
        self.assertEqual(info["variant"], "linux-x86_64-cuda13")

        # 3. CPU-only host -> linux-x86_64-cpu
        env_cpu = dict(base_env, GUILD_MOCK_GPU_NAME="", GUILD_MOCK_GPU_CC="", GUILD_MOCK_GPU_DRIVER="")
        # Ensure nvidia-smi is not called when GUILD_MOCK_GPU_CC is empty by overriding PATH to standard dirs without nvidia-smi if needed
        out = subprocess.check_output(["bash", str(INSTALL_SH), "--variant", "cpu", "--dry-run"], env=env_cpu, text=True)
        info = json.loads(out.splitlines()[-1])
        self.assertEqual(info["variant"], "linux-x86_64-cpu")

        # 4. Pre-AVX2 CPU -> fails closed with source build guidance
        env_no_avx2 = dict(base_env, GUILD_MOCK_CPU_FLAGS="fpu sse4_2 avx")
        res = subprocess.run(["bash", str(INSTALL_SH), "--dry-run"], env=env_no_avx2, capture_output=True, text=True)
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("GUILD_ISA_FLOOR", res.stderr)

    def test_piped_install_update_rollback_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            rel_v1 = tmp_path / "rel-v1"
            rel_v2 = tmp_path / "rel-v2"
            install_root = tmp_path / "install-root"
            bin_dir = tmp_path / "bin"
            user_data = tmp_path / "user-data"
            user_data.mkdir()
            sentinel_model = user_data / "ornith.gpack"
            sentinel_model.write_text("precious-model-weights")

            make_mock_release(
                rel_v1,
                "v0.2.0-preview.1",
                "linux-x86_64-cuda12-sm61",
                "guild-linux-x86_64-cuda12-sm61.tar.gz",
                "v0.2.0-preview.1",
            )
            make_mock_release(
                rel_v2,
                "v0.2.0-preview.2",
                "linux-x86_64-cuda12-sm61",
                "guild-linux-x86_64-cuda12-sm61.tar.gz",
                "v0.2.0-preview.2",
            )

            env = dict(
                os.environ,
                HOME=str(tmp_path),
                GUILD_MOCK_OS="Linux",
                GUILD_MOCK_ARCH="x86_64",
                GUILD_MOCK_CPU_FLAGS="avx2 fma f16c bmi2 avx512f",
                GUILD_MOCK_GPU_NAME="NVIDIA GeForce GTX 1070",
                GUILD_MOCK_GPU_CC="6.1",
                GUILD_MOCK_GPU_DRIVER="580.178.04",
                GUILD_INSTALL_ROOT=str(install_root),
                GUILD_BIN_DIR=str(bin_dir),
                GUILD_RELEASE_BASE_URL=f"file://{rel_v1}",
            )

            # 1. Execute via stdin pipe (`cat install.sh | bash`)
            res1 = subprocess.run(
                ["bash"],
                input=INSTALL_SH.read_text(),
                env=env,
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertIn("v0.2.0-preview.1", res1.stdout)
            ver_out = subprocess.check_output([str(bin_dir / "guild"), "--version"], text=True).strip()
            self.assertEqual(ver_out, "Guild v0.2.0-preview.1")

            # 2. Upgrade to v0.2.0-preview.2
            env_v2 = dict(env, GUILD_RELEASE_BASE_URL=f"file://{rel_v2}")
            subprocess.run(["bash", str(INSTALL_SH)], env=env_v2, check=True, capture_output=True, text=True)
            ver_out2 = subprocess.check_output([str(bin_dir / "guild"), "--version"], text=True).strip()
            self.assertEqual(ver_out2, "Guild v0.2.0-preview.2")

            # 3. Attempt tampered installation -> must fail and keep v0.2.0-preview.2 intact
            rel_bad = tmp_path / "rel-bad"
            make_mock_release(
                rel_bad,
                "v0.2.0-preview.3",
                "linux-x86_64-cuda12-sm61",
                "guild-linux-x86_64-cuda12-sm61.tar.gz",
                "v0.2.0-preview.3",
            )
            (rel_bad / "guild-linux-x86_64-cuda12-sm61.tar.gz.sha256").write_text(
                "0" * 64 + "  guild-linux-x86_64-cuda12-sm61.tar.gz\n"
            )
            env_bad = dict(env, GUILD_RELEASE_BASE_URL=f"file://{rel_bad}")
            bad_res = subprocess.run(["bash", str(INSTALL_SH)], env=env_bad, capture_output=True, text=True)
            self.assertNotEqual(bad_res.returncode, 0)
            self.assertEqual(
                subprocess.check_output([str(bin_dir / "guild"), "--version"], text=True).strip(),
                "Guild v0.2.0-preview.2",
            )

            # 4. Rollback to v0.2.0-preview.1
            subprocess.run(["bash", str(INSTALL_SH), "--rollback"], env=env, check=True, capture_output=True, text=True)
            self.assertEqual(
                subprocess.check_output([str(bin_dir / "guild"), "--version"], text=True).strip(),
                "Guild v0.2.0-preview.1",
            )

            # 5. Uninstall preserves user models/data
            subprocess.run(["bash", str(INSTALL_SH), "--uninstall"], env=env, check=True, capture_output=True, text=True)
            self.assertFalse((bin_dir / "guild").exists())
            self.assertTrue(sentinel_model.exists())


if __name__ == "__main__":
    unittest.main()
