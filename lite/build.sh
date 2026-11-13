#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/build-lite"
mkdir -p "$OUT/include/nlohmann" "$OUT/native" "$OUT/wasm"
# Public MIT JSON dependency, pinned source and digest; the private compiler is never a build input.
JSON="$OUT/include/nlohmann/json.hpp"
if [ ! -f "$JSON" ]; then curl -fsSL https://raw.githubusercontent.com/nlohmann/json/v3.11.3/single_include/nlohmann/json.hpp -o "$JSON"; fi
printf '%s  %s\n' 9bea4c8066ef4a1c206b2be5a36302f8926f7fdc6087af5d20b417d0cf103ea6 "$JSON" | sha256sum -c -
python3 - "$ROOT" <<'PY'
import pathlib,sys
r=pathlib.Path(sys.argv[1]); raw=(r/'schema/lmtorch.dsl.json').read_bytes();(r/'build-lite/include/schema_embedded.hpp').write_text('#pragma once\ninline constexpr unsigned char lite_schema[] = {'+','.join(str(x) for x in raw)+'};\n')
PY
FLAGS=(-std=c++17 -O2 -ffp-contract=off -I"$ROOT/lite" -I"$ROOT/third_party/dslvm" -I"$ROOT/include" -I"$OUT/include")
if [ "${1:-all}" != wasm ]; then
  "${CXX:-g++}" "${FLAGS[@]}" "$ROOT/lite/host.cpp" "$ROOT/src/safetensors.cpp" -o "$OUT/native/lmtorch"
fi
if [ "${1:-all}" != native ]; then
  EMSDK="${EMSDK:-$HOME/repos/lmctlhq/lmlive/.deps/emsdk}"
  "$EMSDK/upstream/emscripten/em++" "${FLAGS[@]}" -fwasm-exceptions "$ROOT/lite/host.cpp" "$ROOT/src/safetensors.cpp" \
    -sMODULARIZE=1 -sEXPORT_NAME=createLmtorchLite -sENVIRONMENT=web,worker,node \
    -lnodefs.js -sEXPORTED_RUNTIME_METHODS=callMain,FS,ENV,NODEFS -sINVOKE_RUN=0 -sFORCE_FILESYSTEM=1 -sALLOW_MEMORY_GROWTH=1 \
    -sMAXIMUM_MEMORY=536870912 -o "$OUT/wasm/lmtorch-lite.cjs"
fi
