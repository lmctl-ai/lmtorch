#!/usr/bin/env python3
"""Static negative controls for removing the legacy lmcpp/lmapi surface."""
from pathlib import Path
import re
import sys

ROOT = Path(sys.argv[1]).resolve()
failures = []

def text(relative):
    path = ROOT / relative
    return path.read_text(encoding="utf-8") if path.exists() else None

for relative in (
    "CMakeLists.txt",
    "include/lmtorch/ops.hpp",
    "include/lmtorch/safetensors.hpp",
    "src/main.cpp",
    "src/ops.cpp",
    "src/safetensors.cpp",
    "src/dsl_host.cpp",
    "scripts/build-native.sh",
    "scripts/package-release.sh",
    ".github/workflows/ci.yml",
):
    body = text(relative)
    if body is None:
        continue
    if "lmcpp" in body.lower():
        failures.append(f"{relative} still references lmcpp")

cmake = text("CMakeLists.txt") or ""
if "find_package(lmcpp" in cmake or "LMCPP_SOURCE_DIR" in cmake:
    failures.append("CMake still discovers or embeds lmcpp")

main = text("src/main.cpp") or ""
if any(token in main for token in ("make_lmtorch_spec", "lmcpp::run_cli", "--cli-schema")):
    failures.append("main.cpp still exposes the legacy lmapi CLI")

workflow = text(".github/workflows/ci.yml") or ""
if "repository: lmctl-ai/lmcpp" in workflow or "LMCPP_REF" in workflow:
    failures.append("CI still checks out or pins lmcpp")

lock = text("sdk.lock") or ""
commit = re.search(r"(?m)^commit=([0-9a-f]{40})$", lock)
if "repo=https://github.com/lmctl-ai/lmcode-cpp\n" not in lock or not commit:
    failures.append("sdk.lock must pin lmcode-cpp by a full 40-character commit SHA")

test_script = text("test.sh") or ""
if '"$BIN" --schema' in test_script or '"$BIN" tensor_' in test_script or "--cli-schema" in test_script:
    failures.append("test.sh still invokes the retired lmapi CLI")

if failures:
    print("FAIL legacy SDK cutover controls:")
    for failure in failures:
        print(f"  - {failure}")
    raise SystemExit(1)
print("PASS no lmcpp/lmapi implementation or test dependencies; full lmcode-cpp pin present")
