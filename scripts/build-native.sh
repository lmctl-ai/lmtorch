#!/usr/bin/env bash
# Build the native torchcli executable without docker.
#   scripts/build-native.sh            release: fully static, against the libtorch from scripts/build-libtorch.sh
#   scripts/build-native.sh --dev      development: against a PyTorch wheel's libtorch (shared libraries, fast to set up)
# Environment: BUILD_DIR (default build, or build-dev for --dev), LIBTORCH_PREFIX (default ~/.cache/torchcli/libtorch-static),
# TORCH_PYTHON (--dev: a python whose torch provides the CMake package), LMCPP_DIR (the lmcpp checkout), JOBS (default 3).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE=release
[ "${1:-}" = "--dev" ] && MODE=dev
BUILD_DIR="${BUILD_DIR:-$HERE/$([ "$MODE" = dev ] && echo build-dev || echo build)}"
JOBS="${JOBS:-3}"
LMCPP_EXPECTED=f6c6f51797912a369d6b27cf9e96f148a9a0e4d1
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
  git -C "$LMCPP_DIR" archive "$LMCPP_EXPECTED" CMakeLists.txt cmake include src examples/toy | tar -x -C "$LMCPP_SRC"
  echo "$LMCPP_EXPECTED" > "$LMCPP_SRC/.revision"
fi
"$HERE/scripts/sync-sdk.sh" --verify >/dev/null
ARGS=(-S "$HERE" -B "$BUILD_DIR" -G Ninja -DCMAKE_BUILD_TYPE=Release -DLMCPP_SOURCE_DIR="$LMCPP_SRC")
if [ "$MODE" = dev ]; then
  PY="${TORCH_PYTHON:-$HOME/.cache/torchcli/oracle-venv/bin/python}"
  ARGS+=(-DCMAKE_PREFIX_PATH="$("$PY" -c 'import torch; print(torch.utils.cmake_prefix_path)')" -DTORCHCLI_LIBTORCH_PREFIX=)
else
  PREFIX="${LIBTORCH_PREFIX:-${XDG_CACHE_HOME:-$HOME/.cache}/torchcli/libtorch-static}"
  ARGS+=(-DTORCHCLI_LIBTORCH_PREFIX="$PREFIX" -DTORCHCLI_FULL_STATIC="${TORCHCLI_FULL_STATIC:-ON}")
fi
cmake "${ARGS[@]}"
cmake --build "$BUILD_DIR" -j"$JOBS"
echo "built $BUILD_DIR/bin/torchcli"
