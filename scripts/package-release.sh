#!/usr/bin/env bash
# Package one release binary: stripped executable plus LICENSE, THIRD_PARTY_NOTICES.md and every bundled component's license.
# Usage: scripts/package-release.sh BINARY PLATFORM OUT_DIR   (e.g. build/bin/lmtorch linux-arm64 dist)
# Writes OUT_DIR/lmtorch-PLATFORM (the executable) and OUT_DIR/lmtorch-PLATFORM.tar.gz (executable + licenses).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$1"; PLATFORM="$2"; OUT="$3"
LICENSES="${LIBTORCH_PREFIX:-${XDG_CACHE_HOME:-$HOME/.cache}/lmtorch/libtorch-static}/licenses"
[ -f "$LICENSES/pytorch-LICENSE" ] || { echo "no license texts in $LICENSES (rebuild with scripts/build-libtorch.sh)" >&2; exit 1; }
mkdir -p "$OUT"
stage="$(mktemp -d)"
trap 'rm -rf "$stage"' EXIT
dir="$stage/lmtorch-$PLATFORM"
mkdir -p "$dir/licenses"
strip -o "$dir/lmtorch" "$BIN"
cp "$HERE/LICENSE" "$HERE/THIRD_PARTY_NOTICES.md" "$dir/"
cp "$LICENSES"/* "$HERE/third_party/dslvm/LICENSE" "$dir/licenses/" 2>/dev/null
mv "$dir/licenses/LICENSE" "$dir/licenses/cpp-dsl-sdk-LICENSE"
cp "$(dirname "$(dirname "$BIN")")/lmcpp-src/LICENSE" "$dir/licenses/lmcpp-LICENSE"
tar -C "$stage" -czf "$OUT/lmtorch-$PLATFORM.tar.gz" "lmtorch-$PLATFORM"
cp "$dir/lmtorch" "$OUT/lmtorch-$PLATFORM"
(cd "$OUT" && sha256sum "lmtorch-$PLATFORM" "lmtorch-$PLATFORM.tar.gz" > "lmtorch-$PLATFORM.sha256")
echo "packaged $OUT/lmtorch-$PLATFORM{,.tar.gz,.sha256}"
