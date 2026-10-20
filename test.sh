#!/usr/bin/env bash
# The native test suite (no docker): SDK pin, schema and signature parity, static-binary check, C++ unit tests, the DSL
# API and refusal suites, every _dsl_ example, the lmapi recipes, and (when a Python with torch is available) the PyTorch
# oracles for both front ends.
#   TORCHCLI_BIN       binary under test (default build/bin/torchcli, else build-dev/bin/torchcli)
#   LMSCRIPT_BIN       compiler (default lmscript on PATH; must emit DSL bytecode v5)
#   ORACLE_PYTHON      python with torch + safetensors (default ~/.cache/torchcli/oracle-venv/bin/python when present)
#   TORCHCLI_REQUIRE_STATIC=1  fail unless the binary is a static executable (release gate)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
if [ -z "${TORCHCLI_BIN:-}" ]; then
  if [ -x build/bin/torchcli ]; then TORCHCLI_BIN="$HERE/build/bin/torchcli"; else TORCHCLI_BIN="$HERE/build-dev/bin/torchcli"; fi
fi
BIN="$(cd "$(dirname "$TORCHCLI_BIN")" && pwd)/$(basename "$TORCHCLI_BIN")"
UNIT="$(dirname "$(dirname "$BIN")")/torchcli_unit"
LMSCRIPT="${LMSCRIPT_BIN:-$(command -v lmscript || true)}"
ORACLE_PYTHON="${ORACLE_PYTHON:-}"
[ -n "$ORACLE_PYTHON" ] || { [ -x "$HOME/.cache/torchcli/oracle-venv/bin/python" ] && ORACLE_PYTHON="$HOME/.cache/torchcli/oracle-venv/bin/python"; } || true
WORK="$(mktemp -d "${TMPDIR:-/tmp}/torchcli-test-XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
pass=0
ok() { pass=$((pass + 1)); echo "ok $pass - $1"; }
fail() { echo "FAIL: $*" >&2; exit 1; }
[ -x "$BIN" ] || fail "no binary at $BIN (scripts/build-native.sh [--dev])"
[ -n "$LMSCRIPT" ] || fail "lmscript not found: npm install -g @lmctl-ai/lmctl (>= 0.2.346), or set LMSCRIPT_BIN"
# Schema-module imports and the example helper launch torchcli by name.
export PATH="$(dirname "$BIN"):$PATH"
export TORCHCLI_BIN="$BIN"

scripts/sync-sdk.sh --verify >/dev/null || fail "third_party/dslvm/dslvm.hpp differs from sdk.lock"
ok "vendored dslvm.hpp matches sdk.lock"
printf 'p = _dsl_ schema="%s";\nwhile (false) {}\nend _dsl_\nsavedsl(p, "%s");\n' "$HERE/schema/torchcli.dsl.json" "$WORK/probe.dslb" > "$WORK/probe.lms"
"$LMSCRIPT" "$WORK/probe.lms" >/dev/null || fail "$LMSCRIPT cannot compile a v5 while loop against the schema"
[ "$(od -An -t u1 -j 5 -N 1 "$WORK/probe.dslb" | tr -d ' ')" = 5 ] || fail "$LMSCRIPT does not emit bytecode v5"
ok "compiler $("$LMSCRIPT" --version) emits DSL bytecode v5"

"$BIN" --dsl-schema --json | cmp - schema/torchcli.dsl.json || fail "embedded DSL schema differs from schema/torchcli.dsl.json"
ok "--dsl-schema --json prints the schema bytes exactly"
diff -u <("$LMSCRIPT" --dsl-signatures schema/torchcli.dsl.json) <("$BIN" --dsl-signatures) || fail "registry and schema signatures differ"
ok "registry signatures equal the compiler's ($("$BIN" --dsl-signatures | wc -l | tr -d ' ') signatures)"
"$BIN" --dsl-schema > "$WORK/summary.txt"
missing=0
while read -r sig; do
  case "$sig" in method\ *) member="${sig#method *.}"; grep -qF "  $member" "$WORK/summary.txt" || missing=$((missing + 1));; esac
done < <("$BIN" --dsl-signatures)
[ "$missing" = 0 ] || fail "--dsl-schema summary is missing $missing members"
ok "--dsl-schema summary lists every member"
"$BIN" --schema | cmp - cli.schema.dev.json || fail "lmapi --schema differs from cli.schema.dev.json"
ok "lmapi --schema is byte-identical to cli.schema.dev.json"

if file -L "$BIN" | grep -q 'statically linked'; then
  { ldd "$BIN" 2>&1 || true; } | grep -q "not a dynamic executable" || fail "static binary still reports dynamic dependencies"
  ok "static executable: $(stat -c %s "$BIN") bytes, no shared-library dependencies"
elif [ "${TORCHCLI_REQUIRE_STATIC:-0}" = 1 ]; then
  fail "$BIN is not statically linked"
else
  echo "# note: $BIN is a dynamic development build (TORCHCLI_REQUIRE_STATIC=1 enforces the release form)"
fi

if [ -x "$UNIT" ]; then (cd "$WORK" && "$UNIT") || fail "C++ safetensors unit tests"; ok "C++ safetensors unit tests"; fi

mkdir -p "$WORK/api/dsl-api-work"
"$LMSCRIPT" tests/dsl/api.lms "$WORK/api/api.dslb" dsl-api-work >/dev/null
(cd "$WORK/api" && "$BIN" --dsl api.dslb > result.json) || fail "DSL API suite: $(cat "$WORK/api/result.json")"
grep -q '"series":\[1,2.5\]' "$WORK/api/result.json" || fail "DSL API suite result: $(cat "$WORK/api/result.json")"
ok "DSL API suite ($(grep -c 'out.check' tests/dsl/api.lms) checks)"
mkdir -p "$WORK/negative"
(cd "$WORK/negative" && "$LMSCRIPT" "$HERE/tests/dsl/negative.lms" "$BIN") > "$WORK/negative.log" || fail "refusal suite: $(cat "$WORK/negative.log")"
ok "$(tail -1 "$WORK/negative.log")"
# The two front ends share the safetensors reader: a BOOL artifact written by the DSL is a coded lmapi refusal.
printf 'p = _dsl_ schema="%s";\ntorch.tensor("[true, false]").save("bools.safetensors");\nend _dsl_\nsavedsl(p, "%s");\n' "$HERE/schema/torchcli.dsl.json" "$WORK/bools.dslb" > "$WORK/bools.lms"
"$LMSCRIPT" "$WORK/bools.lms" >/dev/null
(cd "$WORK" && "$BIN" --dsl bools.dslb >/dev/null) || fail "writing a BOOL artifact"
set +e; (cd "$WORK" && "$BIN" tensor_inspect --input bools.safetensors#tensor) > "$WORK/bools.json"; code=$?; set -e
[ "$code" = 2 ] && grep -q '"tool_code": "INVALID_DTYPE"' "$WORK/bools.json" || fail "lmapi on a BOOL artifact: exit $code $(cat "$WORK/bools.json")"
ok "a DSL-written BOOL artifact is a coded lmapi refusal (INVALID_DTYPE)"
printf 'p = _dsl_ schema="%s";\nlet x = torch.ones(2) + 1;\nend _dsl_\n' "$HERE/schema/torchcli.dsl.json" > "$WORK/illtyped.lms"
set +e; "$LMSCRIPT" --check "$WORK/illtyped.lms" > "$WORK/illtyped.log" 2>&1; code=$?; set -e
[ "$code" = 2 ] && grep -q 'illtyped.lms:2:' "$WORK/illtyped.log" || fail "ill-typed chunk: exit $code $(cat "$WORK/illtyped.log")"
ok "an ill-typed chunk is a located compile error (exit 2) before anything runs"

# Every _dsl_ example, from a scratch workspace (outputs must stay inside the working directory).
run_example() {
  local name="$1" expect="$2"; shift 2
  (cd "$WORK" && "$LMSCRIPT" "$HERE/examples/$name.lms" "$@") > "$WORK/$name.log" 2>&1 || fail "$name: $(cat "$WORK/$name.log")"
  grep -qF -- "$expect" "$WORK/$name.log" || fail "$name output lacks '$expect': $(cat "$WORK/$name.log")"
  ok "example $name"
}
run_example tensor-chain-dsl "[0.5,0.5,0.5,0.5]"
run_example grad-check-dsl "dx: [2,-4,6] loss: 14 finite-difference error below 1e-6: true"
run_example mlp-train-dsl "loss first: 0.886624 last: 0.057287"
run_example train-report-dsl "loss first: 8.315850  last: 0.078491  min: 0.078491"
grep -q 'Training R-squared: 0.989220' "$WORK/artifacts/train-report-dsl/train-report.md" || fail "train-report-dsl report"
run_example classify-moons-dsl "4 predict: decision regions"
run_example lr-sweep-dsl "2 select: lr 0.2"
run_example checkpoint-resume-dsl "checkpoints bit-identical: true"
run_example cnn-bars-dsl "2 classify picture 4: vertical"
run_example regression-csv-dsl "coefficients within 0.25 of the generating formula"
# The lmapi recipes run against the same native binary (no docker shim).
run_example tensor-chain "[0.5,0.5,0.5,0.5]" "artifacts/recipe-chain"
run_example grad-check "dx: [2,-4,6] loss: 14" "artifacts/recipe-grad"
# The lmapi recipe prints full float64 text; BLAS-free static GEMM and the wheel's kernels differ in the last float32 bits.
run_example mlp-train "loss first: 0.88662" "artifacts/recipe-mlp"
grep -q 'last: 0.057286' "$WORK/mlp-train.log" || fail "mlp-train final loss: $(cat "$WORK/mlp-train.log")"
for f in examples/*.lms; do
  case "$f" in examples/train-report.lms) continue;; esac
  "$LMSCRIPT" --check "$f" >/dev/null || fail "lmscript --check $f"
done
ok "every example passes lmscript --check"

if [ -n "$ORACLE_PYTHON" ] && "$ORACLE_PYTHON" -c 'import torch, safetensors, numpy' 2>/dev/null; then
  "$ORACLE_PYTHON" tests/dsl_oracle.py "$BIN" "$LMSCRIPT" > "$WORK/oracle.log" 2>&1 || fail "DSL oracle: $(tail -20 "$WORK/oracle.log")"
  ok "$(tail -1 "$WORK/oracle.log")"
  [ -x "$UNIT" ] && { "$ORACLE_PYTHON" tests/integration.py "$BIN" "$UNIT" tests/torch_oracle.py > "$WORK/integration.log" 2>&1 || fail "lmapi oracle: $(tail -20 "$WORK/integration.log")"; ok "lmapi 12-operation Torch oracle and 21 regression cases"; }
else
  echo "# skipped: PyTorch oracles (set ORACLE_PYTHON to a python with torch, safetensors and numpy)"
fi
echo "torchcli native tests: $pass passed"
