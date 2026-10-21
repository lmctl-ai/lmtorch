#!/usr/bin/env bash
# The native test suite (no docker): SDK pin, schema and signature parity, static-binary check, C++ unit tests, the DSL
# API and refusal suites, every _dsl_ example, the lmapi recipes, and (when a Python with torch is available) the PyTorch
# oracles for both front ends.
#   LMTORCH_BIN       binary under test (default build/bin/lmtorch, else build-dev/bin/lmtorch)
#   LMSCRIPT_BIN       compiler (default lmscript on PATH; must emit DSL bytecode v5)
#   ORACLE_PYTHON      python with torch + safetensors (default ~/.cache/lmtorch/oracle-venv/bin/python when present)
#   LMTORCH_REQUIRE_RELEASE=1  fail unless only system libraries are dynamically linked (release gate): glibc's own on
#                              Linux, libc++ and libSystem on macOS, Windows system DLLs (no VC++ runtime) on Windows
# Runs under bash on Linux, macOS and Windows (Git Bash; the binaries are lmtorch.exe and lmtorch_unit.exe).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
case "$(uname -s)" in Darwin) OS=macos;; MINGW*|MSYS*|CYGWIN*) OS=windows;; *) OS=linux;; esac
EXE=""; [ "$OS" = windows ] && EXE=.exe
# Native Windows programs (lmtorch.exe, node, python) cannot open Git Bash's /c/... or /tmp paths written into files or
# environment variables; npath gives the drive-letter form (C:/...). Elsewhere it is the identity.
npath() { if [ "$OS" = windows ]; then cygpath -m -l "$1"; else printf '%s\n' "$1"; fi; }
if [ -z "${LMTORCH_BIN:-}" ]; then
  if [ -x "build/bin/lmtorch$EXE" ]; then LMTORCH_BIN="$HERE/build/bin/lmtorch$EXE"; else LMTORCH_BIN="$HERE/build-dev/bin/lmtorch$EXE"; fi
fi
BIN="$(cd "$(dirname "$LMTORCH_BIN")" && pwd)/$(basename "$LMTORCH_BIN")"
BIN_DIR="$(dirname "$BIN")"
UNIT="$(dirname "$BIN_DIR")/lmtorch_unit$EXE"
LMSCRIPT="${LMSCRIPT_BIN:-$(command -v lmscript || true)}"
ORACLE_PYTHON="${ORACLE_PYTHON:-}"
for candidate in "$HOME/.cache/lmtorch/oracle-venv/bin/python" "$HOME/.cache/lmtorch/oracle-venv/Scripts/python.exe"; do
  [ -n "$ORACLE_PYTHON" ] || { [ -x "$candidate" ] && ORACLE_PYTHON="$candidate"; } || true
done
WORK="$(mktemp -d "${TMPDIR:-/tmp}/lmtorch-test-XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
WORK="$(npath "$WORK")"
HERE_N="$(npath "$HERE")"
pass=0
ok() { pass=$((pass + 1)); echo "ok $pass - $1"; }
fail() { echo "FAIL: $*" >&2; exit 1; }
[ -x "$BIN" ] || fail "no binary at $BIN (scripts/build-native.sh [--dev])"
[ -n "$LMSCRIPT" ] || fail "lmscript not found: npm install -g @lmctl-ai/lmctl (>= 0.2.346), or set LMSCRIPT_BIN"
# Schema-module imports and the example helper launch lmtorch by name.
export PATH="$BIN_DIR:$PATH"
BIN="$(npath "$BIN")"
UNIT="$(npath "$UNIT")"
export LMTORCH_BIN="$BIN"
# The Python oracles start the compiler themselves: on Windows that is npm's lmscript.cmd, not the Git Bash shim.
LMSCRIPT_PY="$LMSCRIPT"
[ "$OS" = windows ] && [ -f "$LMSCRIPT.cmd" ] && LMSCRIPT_PY="$(npath "$LMSCRIPT.cmd")"

scripts/sync-sdk.sh --verify >/dev/null || fail "third_party/dslvm/dslvm.hpp differs from sdk.lock"
ok "vendored dslvm.hpp matches sdk.lock"
printf 'p = _dsl_ schema="%s";\nwhile (false) {}\nend _dsl_\nsavedsl(p, "%s");\n' "$HERE_N/schema/lmtorch.dsl.json" "$WORK/probe.dslb" > "$WORK/probe.lms"
"$LMSCRIPT" "$WORK/probe.lms" >/dev/null || fail "$LMSCRIPT cannot compile a v5 while loop against the schema"
[ "$(od -An -t u1 -j 5 -N 1 "$WORK/probe.dslb" | tr -d ' ')" = 5 ] || fail "$LMSCRIPT does not emit bytecode v5"
ok "compiler $("$LMSCRIPT" --version) emits DSL bytecode v5"

"$BIN" --dsl-schema --json | cmp - schema/lmtorch.dsl.json || fail "embedded DSL schema differs from schema/lmtorch.dsl.json"
ok "--dsl-schema --json prints the schema bytes exactly"
diff -u <("$LMSCRIPT" --dsl-signatures schema/lmtorch.dsl.json) <("$BIN" --dsl-signatures) || fail "registry and schema signatures differ"
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
for flags in "--cli-schema" "--cli-schema --json"; do "$BIN" $flags | cmp - cli.schema.dev.json || fail "$flags differs from cli.schema.dev.json"; done
ok "host flag standard: --cli-schema [--json] prints the lmapi document, --dsl-schema --json the DSL schema"
# _dsl_ cli="PROGRAM" (lmscript R42) compiles against the binary's own schema; older compilers reject the option.
printf 'p = _dsl_ cli="%s";\nout.set("v", torch.ones(3).sum().item());\nend _dsl_\nsavedsl(p, "%s");\n' "$BIN" "$WORK/cli.dslb" > "$WORK/cli.lms"
if "$LMSCRIPT" --check "$WORK/cli.lms" >/dev/null 2>&1; then
  "$LMSCRIPT" "$WORK/cli.lms" >/dev/null || fail "compiling with cli=$BIN"
  (cd "$WORK" && "$BIN" --dsl cli.dslb) | grep -q '"v":3.0' || fail "running a cli= compiled program"
  ok "_dsl_ cli=\"lmtorch\" compiles against the binary's own schema and runs"
else
  echo "# skipped: _dsl_ cli= (this lmscript predates lmscript R42)"
fi

# Release form: libtorch and the C++ runtime static; only the operating system's own libraries are dynamic.
size="$(wc -c < "$BIN" | tr -d ' ')"
case "$OS" in
  linux)
    # libstdc++ and libgcc static; at most glibc's own libraries are shared (or nothing: fully static).
    needed="$(readelf -d "$BIN" 2>/dev/null | sed -n 's/.*Shared library: \[\(.*\)\]/\1/p' | sort | tr '\n' ' ')"
    foreign="$(printf '%s\n' $needed | grep -Ev '^(libc\.so\.6|libm\.so\.6|libpthread\.so\.0|libdl\.so\.2|librt\.so\.1|ld-linux.*\.so\.[0-9]+)$' || true)"
    if [ -z "$needed" ]; then form="fully static"; else form="glibc-only dynamic ($needed), needs glibc >= $(objdump -T "$BIN" | grep -o 'GLIBC_[0-9.]*' | sed 's/GLIBC_//' | sort -V | tail -1)"; fi;;
  macos)
    # libc++ and libSystem ship with macOS (no Homebrew, no @rpath); the deployment target is the macOS floor.
    needed="$(otool -L "$BIN" | tail -n +2 | awk '{print $1}' | sort | tr '\n' ' ')"
    foreign="$(printf '%s\n' $needed | grep -Ev '^/usr/lib/(libc\+\+\.1|libSystem\.B)\.dylib$' || true)"
    minos="$(otool -l "$BIN" | awk '/LC_BUILD_VERSION/ {b=1} b && $1 == "minos" {print $2; exit}')"
    form="system-library-only dynamic ($needed), needs macOS >= ${minos:-unknown}";;
  windows)
    # Static MSVC runtime (/MT): no vcruntime*/msvcp*/ucrtbase/api-ms-win-crt-* DLLs, only DLLs every Windows has.
    if command -v dumpbin >/dev/null 2>&1; then
      needed="$(dumpbin -nologo -dependents "$BIN" | tr -d '\r' | sed -n 's/^ *\([A-Za-z0-9_.-]*\.[dD][lL][lL]\)$/\1/p' | tr 'A-Z' 'a-z' | sort -u | tr '\n' ' ')"
    else
      needed="$(objdump -p "$BIN" | tr -d '\r' | sed -n 's/.*DLL Name: *//p' | tr 'A-Z' 'a-z' | sort -u | tr '\n' ' ')"
    fi
    [ -n "$needed" ] || fail "cannot list the DLLs $BIN imports (dumpbin or objdump needed)"
    foreign="$(printf '%s\n' $needed | grep -Ev '^(kernel32|kernelbase|ntdll|advapi32|user32|gdi32|shell32|shlwapi|ole32|oleaut32|uuid|comdlg32|ws2_32|bcrypt|crypt32|secur32|dbghelp|psapi|userenv|iphlpapi|rpcrt4|version|winmm|powrprof|setupapi|cfgmgr32|synchronization|api-ms-win-core-[a-z0-9-]+)\.dll$' || true)"
    form="Windows system DLLs only ($needed), static MSVC runtime";;
esac
if [ -z "$foreign" ]; then
  ok "release form: $form, $size bytes"
elif [ "${LMTORCH_REQUIRE_RELEASE:-0}" = 1 ]; then
  fail "$BIN links non-system shared libraries: $foreign"
else
  echo "# note: $BIN is a development build linking $(echo $foreign | wc -w | tr -d ' ') non-system libraries (LMTORCH_REQUIRE_RELEASE=1 enforces the release form)"
fi

if [ -x "$UNIT" ]; then (cd "$WORK" && "$UNIT") || fail "C++ safetensors unit tests"; ok "C++ safetensors unit tests"; fi

mkdir -p "$WORK/api/dsl-api-work"
"$LMSCRIPT" tests/dsl/api.lms "$WORK/api/api.dslb" dsl-api-work >/dev/null
(cd "$WORK/api" && "$BIN" --dsl api.dslb > result.json) || fail "DSL API suite: $(cat "$WORK/api/result.json")"
grep -q '"series":\[1,2.5\]' "$WORK/api/result.json" || fail "DSL API suite result: $(cat "$WORK/api/result.json")"
ok "DSL API suite ($(grep -c 'out.check' tests/dsl/api.lms) checks)"
mkdir -p "$WORK/negative"
(cd "$WORK/negative" && "$LMSCRIPT" "$HERE_N/tests/dsl/negative.lms" "$BIN") > "$WORK/negative.log" || fail "refusal suite: $(cat "$WORK/negative.log")"
ok "$(tail -1 "$WORK/negative.log")"
# The two front ends share the safetensors reader: a BOOL artifact written by the DSL is a coded lmapi refusal.
printf 'p = _dsl_ schema="%s";\ntorch.tensor("[true, false]").save("bools.safetensors");\nend _dsl_\nsavedsl(p, "%s");\n' "$HERE_N/schema/lmtorch.dsl.json" "$WORK/bools.dslb" > "$WORK/bools.lms"
"$LMSCRIPT" "$WORK/bools.lms" >/dev/null
(cd "$WORK" && "$BIN" --dsl bools.dslb >/dev/null) || fail "writing a BOOL artifact"
set +e; (cd "$WORK" && "$BIN" tensor_inspect --input bools.safetensors#tensor) > "$WORK/bools.json"; code=$?; set -e
[ "$code" = 2 ] && grep -q '"tool_code": "INVALID_DTYPE"' "$WORK/bools.json" || fail "lmapi on a BOOL artifact: exit $code $(cat "$WORK/bools.json")"
ok "a DSL-written BOOL artifact is a coded lmapi refusal (INVALID_DTYPE)"
printf 'p = _dsl_ schema="%s";\nlet x = torch.ones(2) + 1;\nend _dsl_\n' "$HERE_N/schema/lmtorch.dsl.json" > "$WORK/illtyped.lms"
set +e; "$LMSCRIPT" --check "$WORK/illtyped.lms" > "$WORK/illtyped.log" 2>&1; code=$?; set -e
[ "$code" = 2 ] && grep -q 'illtyped.lms:2:' "$WORK/illtyped.log" || fail "ill-typed chunk: exit $code $(cat "$WORK/illtyped.log")"
ok "an ill-typed chunk is a located compile error (exit 2) before anything runs"

# Every _dsl_ example, from a scratch workspace (outputs must stay inside the working directory).
run_example() {
  local name="$1" expect="$2"; shift 2
  (cd "$WORK" && "$LMSCRIPT" "$HERE_N/examples/$name.lms" "$@") > "$WORK/$name.log" 2>&1 || fail "$name: $(cat "$WORK/$name.log")"
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
  "$ORACLE_PYTHON" tests/dsl_oracle.py "$BIN" "$LMSCRIPT_PY" > "$WORK/oracle.log" 2>&1 || fail "DSL oracle: $(tail -20 "$WORK/oracle.log")"
  ok "$(tail -1 "$WORK/oracle.log")"
  [ -x "$UNIT" ] && { "$ORACLE_PYTHON" tests/integration.py "$BIN" "$UNIT" tests/torch_oracle.py > "$WORK/integration.log" 2>&1 || fail "lmapi oracle: $(tail -20 "$WORK/integration.log")"; ok "lmapi 12-operation Torch oracle and 21 regression cases"; }
else
  echo "# skipped: PyTorch oracles (set ORACLE_PYTHON to a python with torch, safetensors and numpy; PyTorch publishes no torch 2.9.1 wheel for macOS x86_64)"
fi
echo "lmtorch native tests: $pass passed"
