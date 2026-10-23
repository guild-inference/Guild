#!/usr/bin/env bash
# Guild One-Command Native Installer for Linux (x86_64)
#
# Usage (latest preview):
#   curl -fsSL https://raw.githubusercontent.com/guild-inference/Guild/main/install.sh | bash
#
# Version-pinned installation:
#   curl -fsSL https://raw.githubusercontent.com/guild-inference/Guild/main/install.sh | GUILD_VERSION=v0.2.0-preview.1 bash
#
# Options (via flags or environment variables):
#   --version <tag>          Pin release tag (env: GUILD_VERSION)
#   --channel <preview|stable> Release channel (env: GUILD_CHANNEL, default: preview)
#   --variant <auto|cuda13|cuda12-sm61|cpu> Force variant (env: GUILD_VARIANT, default: auto)
#   --install-root <dir>     Installation root (env: GUILD_INSTALL_ROOT, default: ~/.local/lib/guild)
#   --bin-dir <dir>          User binary directory (env: GUILD_BIN_DIR, default: ~/.local/bin)
#   --no-path-update         Do not modify shell profile PATH
#   --rollback               Restore the preserved previous Guild installation
#   --uninstall              Remove installed Guild binaries (preserves models, GPACK/GEXEC, and data)
#   --dry-run                Detect hardware and print selected artifact without installing

set -euo pipefail

REPO="${GUILD_GITHUB_REPO:-guild-inference/Guild}"
VERSION="${GUILD_VERSION:-}"
CHANNEL="${GUILD_CHANNEL:-preview}"
VARIANT="${GUILD_VARIANT:-auto}"
INSTALL_ROOT="${GUILD_INSTALL_ROOT:-${HOME}/.local/lib/guild}"
BIN_DIR="${GUILD_BIN_DIR:-${HOME}/.local/bin}"
RELEASE_BASE_URL="${GUILD_RELEASE_BASE_URL:-}"
UPDATE_PATH=1
ACTION="install"

while [ $# -gt 0 ]; do
    case "$1" in
        --version)
            VERSION="$2"
            shift 2
            ;;
        --channel)
            CHANNEL="$2"
            shift 2
            ;;
        --variant)
            VARIANT="$2"
            shift 2
            ;;
        --install-root)
            INSTALL_ROOT="$2"
            shift 2
            ;;
        --bin-dir)
            BIN_DIR="$2"
            shift 2
            ;;
        --no-path-update)
            UPDATE_PATH=0
            shift
            ;;
        --rollback)
            ACTION="rollback"
            shift
            ;;
        --uninstall)
            ACTION="uninstall"
            shift
            ;;
        --dry-run)
            ACTION="dry-run"
            shift
            ;;
        -h|--help)
            sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "guild-install: unknown argument '$1'" >&2
            exit 2
            ;;
    esac
done

log_info() {
    printf '\033[1;32m==>\033[0m %s\n' "$*"
}

log_warn() {
    printf '\033[1;33m[warn]\033[0m %s\n' "$*" >&2
}

log_err() {
    printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2
}

# Handle uninstall (never deletes user models, GPACKs, GEXEC caches, or ~/.local/share/guild)
if [ "$ACTION" = "uninstall" ]; then
    log_info "Uninstalling Guild binaries from ${INSTALL_ROOT} and ${BIN_DIR}/guild"
    rm -f "${BIN_DIR}/guild"
    rm -rf "${INSTALL_ROOT}"
    log_info "Uninstalled Guild executable. User models and GPACK/GEXEC caches were preserved."
    exit 0
fi

# Handle rollback to preserved previous installation
if [ "$ACTION" = "rollback" ]; then
    PREV_LINK="${INSTALL_ROOT}/previous"
    CUR_LINK="${INSTALL_ROOT}/current"
    if [ ! -L "$PREV_LINK" ] && [ ! -d "$PREV_LINK" ]; then
        log_err "No previous Guild installation found at ${PREV_LINK} to roll back to."
        exit 1
    fi
    PREV_TARGET="$(readlink -f "$PREV_LINK")"
    if [ ! -x "${PREV_TARGET}/guild" ]; then
        log_err "Previous installation at ${PREV_TARGET} is missing executable 'guild'."
        exit 1
    fi
    log_info "Rolling back Guild to ${PREV_TARGET}..."
    ln -sfn "$PREV_TARGET" "${CUR_LINK}.tmp.$$"
    mv -Tf "${CUR_LINK}.tmp.$$" "$CUR_LINK"
    mkdir -p "$BIN_DIR"
    ln -sfn "${CUR_LINK}/guild" "${BIN_DIR}/guild.tmp.$$"
    mv -Tf "${BIN_DIR}/guild.tmp.$$" "${BIN_DIR}/guild"
    "${BIN_DIR}/guild" --version
    log_info "Rollback complete: $( "${BIN_DIR}/guild" --version | head -n 1 )"
    exit 0
fi

# 1. Detect OS, Architecture, CPU ISA, and GPU / Driver
OS_NAME="${GUILD_MOCK_OS:-$(uname -s)}"
ARCH_NAME="${GUILD_MOCK_ARCH:-$(uname -m)}"

if [ "$OS_NAME" != "Linux" ]; then
    log_err "install.sh supports Linux only (detected OS: ${OS_NAME}). On Windows PowerShell, use install.ps1."
    exit 1
fi

if [ "$ARCH_NAME" != "x86_64" ] && [ "$ARCH_NAME" != "amd64" ]; then
    log_err "Precompiled Linux binaries require x86_64 (detected: ${ARCH_NAME})."
    log_err "Build from source instead: git clone https://github.com/${REPO}.git && cmake -S Guild -B build && cmake --build build -j"
    exit 1
fi

CPU_FLAGS="${GUILD_MOCK_CPU_FLAGS:-}"
if [ -z "$CPU_FLAGS" ] && [ -r /proc/cpuinfo ]; then
    CPU_FLAGS="$(grep -m1 -E '^flags' /proc/cpuinfo | cut -d: -f2 || true)"
fi

for req_flag in avx2 fma f16c bmi2; do
    if ! printf ' %s ' "$CPU_FLAGS" | grep -q " ${req_flag} "; then
        log_err "Host CPU lacks required '${req_flag}' instruction support for portable AVX2 precompiled binaries."
        log_err "Build from source with an older-CPU ISA floor instead:"
        log_err "  git clone https://github.com/${REPO}.git && cd Guild"
        log_err "  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGUILD_ISA_FLOOR=avx"
        log_err "  cmake --build build -j"
        exit 1
    fi
done

HAS_AVX512=0
if printf ' %s ' "$CPU_FLAGS" | grep -q " avx512f "; then
    HAS_AVX512=1
fi

GPU_NAME="${GUILD_MOCK_GPU_NAME:-}"
GPU_CC="${GUILD_MOCK_GPU_CC:-}"
GPU_DRIVER="${GUILD_MOCK_GPU_DRIVER:-}"

if [ -z "$GPU_CC" ] && command -v nvidia-smi >/dev/null 2>&1; then
    SMI_LINE="$(nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader,nounits 2>/dev/null | head -n 1 || true)"
    if [ -n "$SMI_LINE" ]; then
        GPU_NAME="$(printf '%s' "$SMI_LINE" | cut -d',' -f1 | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
        GPU_CC="$(printf '%s' "$SMI_LINE" | cut -d',' -f2 | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
        GPU_DRIVER="$(printf '%s' "$SMI_LINE" | cut -d',' -f3 | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
    fi
fi

RESOLVED_VARIANT=""
if [ "$VARIANT" != "auto" ]; then
    case "$VARIANT" in
        cuda13|linux-x86_64-cuda13) RESOLVED_VARIANT="linux-x86_64-cuda13" ;;
        cuda12-sm61|cuda12|linux-x86_64-cuda12-sm61) RESOLVED_VARIANT="linux-x86_64-cuda12-sm61" ;;
        cpu|linux-x86_64-cpu) RESOLVED_VARIANT="linux-x86_64-cpu" ;;
        *)
            log_err "Unsupported --variant '${VARIANT}' (expected auto, cuda13, cuda12-sm61, or cpu)."
            exit 2
            ;;
    esac
else
    if [ -n "$GPU_CC" ] && [ -n "$GPU_DRIVER" ]; then
        CC_MAJOR="$(printf '%s' "$GPU_CC" | cut -d'.' -f1)"
        CC_MINOR="$(printf '%s' "$GPU_CC" | cut -d'.' -f2)"
        CC_INT=$(( CC_MAJOR * 10 + CC_MINOR ))
        DRV_MAJOR="$(printf '%s' "$GPU_DRIVER" | cut -d'.' -f1)"

        if [ "$CC_INT" -ge 75 ] && [ "$DRV_MAJOR" -ge 570 ]; then
            RESOLVED_VARIANT="linux-x86_64-cuda13"
        elif [ "$CC_INT" -ge 60 ] && [ "$CC_INT" -le 89 ] && [ "$DRV_MAJOR" -ge 525 ]; then
            RESOLVED_VARIANT="linux-x86_64-cuda12-sm61"
        elif [ "$CC_INT" -lt 60 ]; then
            log_warn "GPU '${GPU_NAME}' (sm_${CC_INT}) is below Guild's minimum compute capability 6.0; selecting CPU management binary."
            RESOLVED_VARIANT="linux-x86_64-cpu"
        else
            log_warn "NVIDIA driver ${GPU_DRIVER} is older than CUDA 12/13 minimum (525+); selecting CPU management binary."
            RESOLVED_VARIANT="linux-x86_64-cpu"
        fi
    else
        RESOLVED_VARIANT="linux-x86_64-cpu"
    fi
fi

case "$RESOLVED_VARIANT" in
    linux-x86_64-cuda13) ARCHIVE_NAME="guild-linux-x86_64-cuda13.tar.gz" ;;
    linux-x86_64-cuda12-sm61) ARCHIVE_NAME="guild-linux-x86_64-cuda12-sm61.tar.gz" ;;
    linux-x86_64-cpu) ARCHIVE_NAME="guild-linux-x86_64-cpu.tar.gz" ;;
esac

log_info "Detected host: Linux ${ARCH_NAME} (AVX2=yes, AVX-512=${HAS_AVX512}, GPU='${GPU_NAME:-none}', CC='${GPU_CC:-n/a}', Driver='${GPU_DRIVER:-n/a}')"
log_info "Selected variant: ${RESOLVED_VARIANT} (${ARCHIVE_NAME})"

if [ "$ACTION" = "dry-run" ]; then
    printf '{"os":"Linux","arch":"%s","gpu_name":"%s","gpu_cc":"%s","gpu_driver":"%s","variant":"%s","archive":"%s","install_root":"%s","bin_dir":"%s"}\n' \
        "$ARCH_NAME" "${GPU_NAME:-none}" "${GPU_CC:-}" "${GPU_DRIVER:-}" "$RESOLVED_VARIANT" "$ARCHIVE_NAME" "$INSTALL_ROOT" "$BIN_DIR"
    exit 0
fi

# 2. Resolve release tag & download manifest
fetch_file() {
    local url="$1"
    local dst="$2"
    if [[ "$url" == file://* ]]; then
        cp "${url#file://}" "$dst"
    elif command -v curl >/dev/null 2>&1; then
        curl -fsSL --retry 3 --output "$dst" "$url"
    elif command -v wget >/dev/null 2>&1; then
        wget -q -O "$dst" "$url"
    else
        log_err "Neither curl nor wget is installed."
        exit 1
    fi
}

mkdir -p "$INSTALL_ROOT"
WORK_DIR="$(mktemp -d "${INSTALL_ROOT}/.staging-XXXXXX")"
cleanup() {
    rm -rf "$WORK_DIR"
}
trap cleanup EXIT

if [ -n "$RELEASE_BASE_URL" ]; then
    MANIFEST_URL="${RELEASE_BASE_URL%/}/guild-release.json"
    fetch_file "$MANIFEST_URL" "${WORK_DIR}/guild-release.json"
    if [ -z "$VERSION" ]; then
        VERSION="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["release_tag"])' "${WORK_DIR}/guild-release.json")"
    fi
    ASSET_BASE_URL="${RELEASE_BASE_URL%/}"
else
    if [ -z "$VERSION" ]; then
        log_info "Resolving latest ${CHANNEL} release from ${REPO}..."
        fetch_file "https://api.github.com/repos/${REPO}/releases?per_page=30" "${WORK_DIR}/releases.json"
        VERSION="$(python3 - "${WORK_DIR}/releases.json" "$CHANNEL" <<'PY'
import json, sys
releases = json.load(open(sys.argv[1]))
channel = sys.argv[2]
for rel in releases:
    if rel.get("draft"):
        continue
    if channel == "stable" and rel.get("prerelease"):
        continue
    assets = {a.get("name") for a in rel.get("assets", [])}
    if "guild-release.json" in assets:
        print(rel["tag_name"])
        sys.exit(0)
sys.exit(1)
PY
)" || {
            log_err "No compatible published release with guild-release.json found on ${REPO} (channel: ${CHANNEL})."
            exit 1
        }
    fi
    ASSET_BASE_URL="https://github.com/${REPO}/releases/download/${VERSION}"
    fetch_file "${ASSET_BASE_URL}/guild-release.json" "${WORK_DIR}/guild-release.json"
fi

log_info "Downloading Guild ${VERSION} (${ARCHIVE_NAME})..."
fetch_file "${ASSET_BASE_URL}/${ARCHIVE_NAME}" "${WORK_DIR}/${ARCHIVE_NAME}"
fetch_file "${ASSET_BASE_URL}/${ARCHIVE_NAME}.sha256" "${WORK_DIR}/${ARCHIVE_NAME}.sha256"

# 3. Verify archive SHA-256 against .sha256 and guild-release.json manifest
log_info "Verifying SHA-256 integrity and release manifest..."
(cd "$WORK_DIR" && sha256sum --check "${ARCHIVE_NAME}.sha256" >/dev/null)

ACTUAL_ARCHIVE_SHA="$(sha256sum "${WORK_DIR}/${ARCHIVE_NAME}" | awk '{print $1}')"
EXPECTED_EXEC_SHA="$(python3 - "${WORK_DIR}/guild-release.json" "$RESOLVED_VARIANT" "$VERSION" "$ACTUAL_ARCHIVE_SHA" <<'PY'
import json, sys
manifest_path, variant_id, expected_tag, actual_archive_sha = sys.argv[1:5]
m = json.load(open(manifest_path))
if m.get("release_tag") != expected_tag:
    raise SystemExit(f"Release manifest tag mismatch: expected {expected_tag}, got {m.get('release_tag')}")
variants = m.get("variants", {})
if variant_id in variants:
    v = variants[variant_id]
    if v.get("sha256") != actual_archive_sha:
        raise SystemExit(f"Manifest archive SHA-256 mismatch for {variant_id}")
    print(v["executable_sha256"])
else:
    print(m["executable_sha256"])
PY
)"

# 4. Extract and validate staged binary before touching active installation
tar -xzf "${WORK_DIR}/${ARCHIVE_NAME}" -C "$WORK_DIR"
EXTRACTED_DIR="$(find "$WORK_DIR" -mindepth 1 -maxdepth 1 -type d | head -n 1)"
if [ -z "$EXTRACTED_DIR" ] || [ ! -x "${EXTRACTED_DIR}/guild" ] || [ ! -x "${EXTRACTED_DIR}/guild.bin" ]; then
    log_err "Extracted archive is missing executable guild / guild.bin."
    exit 1
fi

ACTUAL_EXEC_SHA="$(sha256sum "${EXTRACTED_DIR}/guild.bin" | awk '{print $1}')"
if [ "$ACTUAL_EXEC_SHA" != "$EXPECTED_EXEC_SHA" ]; then
    log_err "Executable SHA-256 mismatch: expected ${EXPECTED_EXEC_SHA}, got ${ACTUAL_EXEC_SHA}"
    exit 1
fi

# Verify runtime shared libraries resolve cleanly
if command -v ldd >/dev/null 2>&1; then
    LDD_OUT="$(LD_LIBRARY_PATH="${EXTRACTED_DIR}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ldd "${EXTRACTED_DIR}/guild.bin" 2>&1 || true)"
    if printf '%s' "$LDD_OUT" | grep -q "not found"; then
        log_err "Staged binary has unresolved shared library dependencies:"
        printf '%s\n' "$LDD_OUT" >&2
        exit 1
    fi
fi

# Smoke-test staged binary in isolated data dir
mkdir -p "${WORK_DIR}/smoke-data"
GUILD_DATA_DIR="${WORK_DIR}/smoke-data" "${EXTRACTED_DIR}/guild" --version >/dev/null
GUILD_DATA_DIR="${WORK_DIR}/smoke-data" "${EXTRACTED_DIR}/guild" --help >/dev/null

# 5. Install into versioned directory and preserve previous installation for rollback
VERSION_DIR="${INSTALL_ROOT}/versions/${VERSION}-${RESOLVED_VARIANT}"
mkdir -p "${INSTALL_ROOT}/versions" "$BIN_DIR"

# Stage into a unique version directory so a running binary is never overwritten in place
STAGE_DEST="${VERSION_DIR}.new.$$"
rm -rf "$STAGE_DEST"
mv "$EXTRACTED_DIR" "$STAGE_DEST"
cp "${WORK_DIR}/guild-release.json" "${STAGE_DEST}/guild-release.json"

if [ -d "$VERSION_DIR" ]; then
    mv "$VERSION_DIR" "${VERSION_DIR}.old.$$"
fi
mv "$STAGE_DEST" "$VERSION_DIR"
rm -rf "${VERSION_DIR}.old.$$" 2>/dev/null || true

CUR_LINK="${INSTALL_ROOT}/current"
PREV_LINK="${INSTALL_ROOT}/previous"
if [ -L "$CUR_LINK" ]; then
    OLD_TARGET="$(readlink -f "$CUR_LINK" || true)"
    if [ -n "$OLD_TARGET" ] && [ -d "$OLD_TARGET" ] && [ "$OLD_TARGET" != "$(readlink -f "$VERSION_DIR")" ]; then
        ln -sfn "$OLD_TARGET" "${PREV_LINK}.tmp.$$"
        mv -Tf "${PREV_LINK}.tmp.$$" "$PREV_LINK"
    fi
fi

ln -sfn "$VERSION_DIR" "${CUR_LINK}.tmp.$$"
mv -Tf "${CUR_LINK}.tmp.$$" "$CUR_LINK"

ln -sfn "${CUR_LINK}/guild" "${BIN_DIR}/guild.tmp.$$"
mv -Tf "${BIN_DIR}/guild.tmp.$$" "${BIN_DIR}/guild"

# 6. Ensure user-local PATH contains BIN_DIR
PATH_UPDATED=0
case ":${PATH}:" in
    *":${BIN_DIR}:"*) ;;
    *)
        if [ "$UPDATE_PATH" -eq 1 ]; then
            EXPORT_LINE="export PATH=\"${BIN_DIR}:\$PATH\""
            for rc_file in "${HOME}/.bashrc" "${HOME}/.profile" "${HOME}/.zshrc"; do
                if [ -f "$rc_file" ] || [ "$rc_file" = "${HOME}/.profile" ]; then
                    if ! grep -Fq "$BIN_DIR" "$rc_file" 2>/dev/null; then
                        printf '\n# Added by Guild installer\n%s\n' "$EXPORT_LINE" >> "$rc_file"
                        PATH_UPDATED=1
                    fi
                fi
            done
        fi
        ;;
esac

# 7. Final post-install verification
INSTALLED_VER="$("${BIN_DIR}/guild" --version | head -n 1)"
"${BIN_DIR}/guild" --help >/dev/null

log_info "Installed ${INSTALLED_VER} (${RESOLVED_VARIANT})"
printf '  Executable : %s/guild -> %s/guild\n' "$BIN_DIR" "$VERSION_DIR"
printf '  Launch TUI : guild tui\n'
printf '  Pull model : guild pull qwen3.8-flash-next\n'
printf '  Rollback   : bash install.sh --rollback\n'
if [ "$PATH_UPDATED" -eq 1 ]; then
    printf '  Note       : Added %s to your shell profile. Run: export PATH="%s:$PATH"\n' "$BIN_DIR" "$BIN_DIR"
fi
