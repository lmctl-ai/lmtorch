#!/usr/bin/env bash
# Build the native lmtorch executable.
#   scripts/build-native.sh            release: static libtorch, libstdc++ and libgcc; only glibc is a shared library
#                                      (LMTORCH_FULL_STATIC=ON links glibc statically too)
#   scripts/build-native.sh --dev      development: against a PyTorch wheel's libtorch (shared libraries, fast to set up)
# Linux (GCC), macOS (AppleClang) and Windows (MSVC from Git Bash: run inside a Visual Studio developer environment).
# Environment: BUILD_DIR (default build, or build-dev for --dev), LIBTORCH_PREFIX (default ~/.cache/lmtorch/libtorch-static),
# TORCH_PYTHON (--dev: a python whose torch provides the CMake package), LMCPP_DIR (the lmcpp checkout), JOBS (default 3).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE=release
[ "${1:-}" = "--dev" ] && MODE=dev
BUILD_DIR="${BUILD_DIR:-$HERE/$([ "$MODE" = dev ] && echo build-dev || echo build)}"
JOBS="${JOBS:-3}"
LMCPP_EXPECTED=93a1847c081894ab1e9ed02ad5ea6350b5d5bd06
if [ -z "${LMCPP_DIR:-}" ]; then
  for candidate in "$HERE/../../lmctl-ai/lmcpp" "$HERE/../lmcpp"; do
    [ -d "$candidate/.git" ] && { LMCPP_DIR="$candidate"; break; }
  done
fi
[ -n "${LMCPP_DIR:-}" ] || { echo "lmcpp checkout not found (set LMCPP_DIR)" >&2; exit 1; }
git -C "$LMCPP_DIR" cat-file -e "$LMCPP_EXPECTED^{commit}" 2>/dev/null || { echo "lmcpp checkout at $LMCPP_DIR lacks pinned revision $LMCPP_EXPECTED" >&2; exit 1; }
# The pinned revision is extracted from git, so the checkout's working tree and branch do not matter.
LMCPP_SRC="$BUILD_DIR/lmcpp-src"
if [ "$(cat "$LMCPP_SRC/.revision" 2>/dev/null || true)" != "$LMCPP_EXPECTED" ]; then
  rm -rf "$LMCPP_SRC"
  mkdir -p "$LMCPP_SRC"
  git -C "$LMCPP_DIR" archive "$LMCPP_EXPECTED" CMakeLists.txt LICENSE cmake include src examples/toy | tar -x -C "$LMCPP_SRC"
  echo "$LMCPP_EXPECTED" > "$LMCPP_SRC/.revision"
fi
"$HERE/scripts/sync-sdk.sh" --verify >/dev/null
# CMake and cl.exe on Windows want drive-letter paths, not Git Bash's /c/... form.
native() { case "$(uname -s)" in MINGW*|MSYS*|CYGWIN*) cygpath -m "$1";; *) printf '%s\n' "$1";; esac; }
ARGS=(-S "$(native "$HERE")" -B "$(native "$BUILD_DIR")" -G Ninja -DCMAKE_BUILD_TYPE=Release -DLMCPP_SOURCE_DIR="$(native "$LMCPP_SRC")")
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
