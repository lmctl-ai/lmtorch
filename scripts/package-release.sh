#!/usr/bin/env bash
# Package one release binary: stripped executable plus LICENSE, THIRD_PARTY_NOTICES.md and every bundled component's license.
# Usage: scripts/package-release.sh BINARY PLATFORM OUT_DIR   (e.g. build/bin/lmtorch linux-arm64 dist)
# Writes OUT_DIR/lmtorch-PLATFORM (the executable; lmtorch-PLATFORM.exe on Windows), an archive with the executable and
# the license texts (lmtorch-PLATFORM.tar.gz; lmtorch-PLATFORM.zip on Windows) and lmtorch-PLATFORM.sha256 for both.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$1"; PLATFORM="$2"; OUT="$3"
case "$(uname -s)" in Darwin) OS=macos;; MINGW*|MSYS*|CYGWIN*) OS=windows;; *) OS=linux;; esac
EXE=""; [ "$OS" = windows ] && EXE=.exe
[ -f "$BIN" ] || BIN="$BIN$EXE"
LICENSES="${LIBTORCH_PREFIX:-${XDG_CACHE_HOME:-$HOME/.cache}/lmtorch/libtorch-static}/licenses"
[ -f "$LICENSES/pytorch-LICENSE" ] || { echo "no license texts in $LICENSES (rebuild with scripts/build-libtorch.sh)" >&2; exit 1; }
mkdir -p "$OUT"
stage="$(mktemp -d)"
trap 'rm -rf "$stage"' EXIT
name="lmtorch-$PLATFORM"
dir="$stage/$name"
mkdir -p "$dir/licenses"
case "$OS" in
  linux|macos) strip -o "$dir/lmtorch" "$BIN";;   # macOS: Apple strip; libc++/libSystem stay as the only dylibs
  windows) cp "$BIN" "$dir/lmtorch.exe";;          # MSVC Release keeps no symbols in the .exe (no PDB is shipped)
esac
cp "$HERE/LICENSE" "$HERE/THIRD_PARTY_NOTICES.md" "$dir/"
cp "$LICENSES"/* "$HERE/third_party/dslvm/LICENSE" "$dir/licenses/" 2>/dev/null
mv "$dir/licenses/LICENSE" "$dir/licenses/cpp-dsl-sdk-LICENSE"
cp "$(dirname "$(dirname "$BIN")")/lmcpp-src/LICENSE" "$dir/licenses/lmcpp-LICENSE"
if [ "$OS" = windows ]; then
  archive="$name.zip"
  py="$(command -v python3 || command -v python)"
  "$py" -c 'import shutil, sys; shutil.make_archive(sys.argv[1], "zip", root_dir=sys.argv[2], base_dir=sys.argv[3])' \
    "$(cygpath -m "$OUT")/$name" "$(cygpath -m "$stage")" "$name"
else
  archive="$name.tar.gz"
  tar -C "$stage" -czf "$OUT/$archive" "$name"
fi
cp "$dir/lmtorch$EXE" "$OUT/$name$EXE"
sha() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$@"; else shasum -a 256 "$@"; fi; }
(cd "$OUT" && sha "$name$EXE" "$archive" > "$name.sha256")
echo "packaged $OUT/$name$EXE, $OUT/$archive, $OUT/$name.sha256"
