#!/bin/sh
# Run every examples/*.lms through the host shim in --check, --dry-run and real modes from a scratch directory below HOME (the shim's
# requirement; artifacts land under it), with this repository's bin on PATH. Recipes run in place, so their imports resolve against the
# checked-in cli.schema.dev.json and the sibling lmcreator checkout. A recipe whose sibling schema (train-report.lms: lmcreator) or whose
# lmcreator native helpers are not present is reported as skipped, never as passed. Usage: scripts/run-recipes.sh [examples/x.lms ...];
# exit 1 when an invocation fails.
set -u
repo=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
work=$(mktemp -d "${HOME:?}/torchcli-recipes.XXXXXX")
trap 'rm -rf "$work"' EXIT HUP INT TERM
export PATH="$repo/bin:$PATH"
creator="$repo/../../lmctlhq/lmcreator"
helpers="${HOME}/.cache/lmcreator-native-$(uname -m)"
if [ -x "$creator/bin/lmcreator" ] && [ -x "$helpers/resvg-embed" ]; then
  export PATH="$creator/bin:$PATH" RESVG_BIN="$helpers/resvg-embed" PAGE_PROBE_BIN="$helpers/page-probe" PDF_WRITE_BIN="$helpers/pdf-write" PDF_READ_BIN="$helpers/pdf-read" FIELD_RENDER_BIN="$helpers/field-render"
  creator_ok=1
else
  creator_ok=0
fi
cd "$work" || exit 1
export TMPDIR="$work/.tmp"; mkdir -p "$TMPDIR"
fail=0; n=0; skipped=0
if [ $# -gt 0 ]; then set -- "$@"; else set -- "$repo"/examples/*.lms; fi
for f in "$@"; do
  name=$(basename "$f" .lms); f="$repo/examples/$name.lms"
  missing=
  for imp in $(sed -n 's/^import "\([^"]*\)".*/\1/p' "$f"); do
    case "$imp" in
      ../cli.schema.dev.json) ;;
      *lmcreator*) [ "$creator_ok" -eq 1 ] && [ -f "$repo/examples/$imp" ] || missing="$imp (lmcreator checkout and native helpers)";;
      *) [ -f "$repo/examples/$imp" ] || missing="$imp";;
    esac
  done
  if [ -n "$missing" ]; then skipped=$((skipped+1)); printf '%-24s skipped: %s not present\n' "$name" "$missing"; continue; fi
  for mode in --check --dry-run real; do
    n=$((n+1))
    case "$mode" in
      --check) lmscript --check "$f" > "$work/last.log" 2>&1; rc=$?;;
      --dry-run) lmscript --dry-run "$f" "artifacts/$name-dry" > "$work/last.log" 2>&1; rc=$?;;
      real) lmscript "$f" "artifacts/$name" > "$work/last.log" 2>&1; rc=$?;;
    esac
    printf '%-24s %-10s rc=%d\n' "$name" "$mode" "$rc"
    if [ "$rc" -ne 0 ]; then fail=$((fail+1)); sed 's/^/    /' "$work/last.log" | tail -n 20; fi
  done
done
echo "invocations=$n failures=$fail skipped_recipes=$skipped"
[ "$fail" -eq 0 ]
