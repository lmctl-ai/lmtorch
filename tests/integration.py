"""CPU-only lmcode host integration: public API, refusals, numerics and invocation frames."""
from pathlib import Path
import json
import subprocess
import sys
import tempfile

binary, lmscript = sys.argv[1:3]
root = Path(__file__).resolve().parents[1]

# Exercise the complete compiled lmcode API from one real host invocation.
with tempfile.TemporaryDirectory(prefix="lmtorch-api-") as scratch:
    work = Path(scratch)
    api = work / "api"
    api.mkdir()
    bytecode = api / "api.dslb"
    subprocess.run([lmscript, str(root / "tests/dsl/api.lms"), str(bytecode), "dsl-api-work"],
                   cwd=root, check=True, capture_output=True)
    result = subprocess.run([binary, "--lmcode", str(bytecode)], cwd=api, check=True, text=True, capture_output=True)
    lines = result.stdout.splitlines()
    starts = [i for i, line in enumerate(lines) if line == "LMCODE_RETURN_BEGIN"]
    ends = [i for i, line in enumerate(lines) if line == "LMCODE_RETURN_END"]
    assert len(starts) == len(ends) == 1 and ends[0] == starts[0] + 2, result.stdout
    envelope = json.loads(lines[starts[0] + 1])
    assert envelope["ok"] and envelope["result"]["series"] == [1, 2.5], envelope

# Preserve the native refusal/location suite using the manifest path through run_lmcode.
with tempfile.TemporaryDirectory(prefix="lmtorch-negative-") as scratch:
    result = subprocess.run([lmscript, str(root / "tests/dsl/negative.lms"), binary],
                            cwd=scratch, check=True, text=True, capture_output=True)
    assert "cases passed" in result.stdout, result.stdout

# CPU numerical oracle and run_lmcode's v1 manifest/frame contract.
subprocess.run([sys.executable, str(root / "tests/dsl_oracle.py"), binary, lmscript], check=True)
subprocess.run([sys.executable, str(root / "tests/check-invocation-manifest.py"), binary, lmscript], check=True)
print("lmcode integration: API/refusal suites, CPU PyTorch oracle and manifest-v1 controls passed")
