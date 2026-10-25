#!/usr/bin/env bash
# Vendor the lmcode-cpp VM, its built-in std catalog headers, the canonical std schema and the merge helper from one
# pinned commit, with a per-file content hash (same contract as lmcad/lmhtml).
set -eu
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SDK="${LMCODE_CPP:-$HERE/../lmcode-cpp}"
LOCK="$HERE/sdk.lock"
VENDOR=third_party/dslvm
hash() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1"; else shasum -a 256 "$1"; fi | cut -c1-64; }
if [ "${1:-}" = "--verify" ]; then
  [ -f "$LOCK" ] || { echo "sdk.lock missing" >&2; exit 1; }
  count=0
  while IFS='=' read -r key want; do
    case "$key" in
      sha256.*)
        file="${key#sha256.}"
        [ -f "$HERE/$file" ] && [ "$(hash "$HERE/$file")" = "$want" ] || { echo "$file differs from sdk.lock: run scripts/sync-sdk.sh" >&2; exit 1; }
        count=$((count+1)) ;;
    esac
  done < "$LOCK"
  [ "$count" -gt 0 ] || { echo "sdk.lock has no per-file hashes" >&2; exit 1; }
  # Every vendored file must be pinned (no stray header left behind by an older sync).
  for file in "$HERE/$VENDOR/LICENSE" "$HERE/$VENDOR/dslvm.hpp" "$HERE/$VENDOR"/stdlib/*.hpp "$HERE/$VENDOR/std/lmcode-std.schema.json" "$HERE/scripts/merge-std.py"; do
    [ -f "$file" ] || { echo "missing SDK file: $file" >&2; exit 1; }
    relative="${file#"$HERE/"}"
    grep -qx "sha256.$relative=$(hash "$file")" "$LOCK" || { echo "$relative is not pinned in sdk.lock" >&2; exit 1; }
  done
  echo "$count SDK files match sdk.lock ($(sed -n 's/^commit=//p' "$LOCK"))"
  exit 0
fi
[ -f "$SDK/include/dslvm.hpp" ] || { echo "lmcode-cpp checkout not found at $SDK (set LMCODE_CPP)" >&2; exit 1; }
ref="${LMCODE_CPP_REF:-origin/main}"
commit="$(git -C "$SDK" rev-parse "$ref")"
rm -rf "$HERE/$VENDOR"
mkdir -p "$HERE/$VENDOR/stdlib" "$HERE/$VENDOR/std"
printf 'repo=https://github.com/lmctl-ai/lmcode-cpp\ncommit=%s\n' "$commit" > "$LOCK"
while IFS= read -r source; do
  case "$source" in
    include/dslvm.hpp) target="$VENDOR/dslvm.hpp" ;;
    LICENSE) target="$VENDOR/LICENSE" ;;
    stdlib/*.hpp|std/lmcode-std.schema.json) target="$VENDOR/$source" ;;
    scripts/merge-std.py) target="$source" ;;
    *) continue ;;
  esac
  git -C "$SDK" show "$commit:$source" > "$HERE/$target"
  printf 'sha256.%s=%s\n' "$target" "$(hash "$HERE/$target")" >> "$LOCK"
done < <(git -C "$SDK" ls-tree -r --name-only "$commit" -- LICENSE include/dslvm.hpp stdlib/ std/lmcode-std.schema.json scripts/merge-std.py)
echo "synced SDK files from lmcode-cpp $commit"
