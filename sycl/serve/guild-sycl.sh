#!/usr/bin/env bash
# Native oneAPI launcher for the experimental Intel Arc engine.
set -euo pipefail
here=$(cd "$(dirname "$0")/../.." && pwd)
if [ -f /opt/intel/oneapi/setvars.sh ]; then
    source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1
fi
exec "${GUILD_SYCL_BIN:-$here/build-sycl/guild-generate}" "$@"
