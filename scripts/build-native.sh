#!/usr/bin/env bash
# Build the native lmtorch executable.
#   scripts/build-native.sh            release: static libtorch, libstdc++ and libgcc; only glibc is a shared library
#                                      (LMTORCH_FULL_STATIC=ON links glibc statically too)
#   scripts/build-native.sh --dev      development: against a PyTorch wheel's libtorch (shared libraries, fast to set up)
# Linux (GCC), macOS (AppleClang) and Windows (MSVC from Git Bash: run inside a Visual Studio developer environment).
# Environment: BUILD_DIR (default build, or build-dev for --dev), LIBTORCH_PREFIX (default ~/.cache/lmtorch/libtorch-static),
# TORCH_PYTHON (--dev: a python whose torch provides the CMake package), JOBS (default 3).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE=release
[ "${1:-}" = "--dev" ] && MODE=dev
BUILD_DIR="${BUILD_DIR:-$HERE/$([ "$MODE" = dev ] && echo build-dev || echo build)}"
JOBS="${JOBS:-3}"
# The reviewed lmcode-cpp commit and per-file hashes are committed in sdk.lock.
"$HERE/scripts/sync-sdk.sh" --verify >/dev/null
# CMake and cl.exe on Windows want drive-letter paths, not Git Bash's /c/... form.
native() { case "$(uname -s)" in MINGW*|MSYS*|CYGWIN*) cygpath -m "$1";; *) printf '%s\n' "$1";; esac; }
ARGS=(-S "$(native "$HERE")" -B "$(native "$BUILD_DIR")" -G Ninja -DCMAKE_BUILD_TYPE=Release)
case "$(uname -s)" in MINGW*|MSYS*|CYGWIN*) ARGS+=(-DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl);; esac
if [ "$MODE" = dev ]; then
  PY="${TORCH_PYTHON:-$HOME/.cache/lmtorch/oracle-venv/bin/python}"
  ARGS+=(-DCMAKE_PREFIX_PATH="$("$PY" -c 'import torch; print(torch.utils.cmake_prefix_path)')" -DLMTORCH_LIBTORCH_PREFIX=)
else
  PREFIX="${LIBTORCH_PREFIX:-${XDG_CACHE_HOME:-$HOME/.cache}/lmtorch/libtorch-static}"
  ARGS+=(-DLMTORCH_LIBTORCH_PREFIX="$(native "$PREFIX")" -DLMTORCH_FULL_STATIC="${LMTORCH_FULL_STATIC:-OFF}")
fi
cmake "${ARGS[@]}"
cmake --build "$(native "$BUILD_DIR")" -j"$JOBS"
echo "built $BUILD_DIR/bin/lmtorch$(case "$(uname -s)" in MINGW*|MSYS*|CYGWIN*) echo .exe;; esac)"
