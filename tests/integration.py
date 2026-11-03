"""CPU-only lmcode host integration: public API, refusals, numerics and invocation frames."""
from pathlib import Path
import json
import subprocess
import sys
import tempfile

binary, lmscript = sys.argv[1:3]
root = Path(__file__).resolve().parents[1]

# Exercise the complete compiled lmcode API through run_lmcode and its terminal frame.
with tempfile.TemporaryDirectory(prefix="lmtorch-api-") as scratch:
    work = Path(scratch)
    api = work / "api"
    api.mkdir()
    subprocess.run([lmscript, str(root / "tests/dsl/api.lms"), binary, str(api / "dsl-api-work")],
                   cwd=api, check=True, capture_output=True, text=True)

# Preserve the native refusal/location suite using the manifest path through run_lmcode.
with tempfile.TemporaryDirectory(prefix="lmtorch-negative-") as scratch:
    result = subprocess.run([lmscript, str(root / "tests/dsl/negative.lms"), binary],
                            cwd=scratch, check=True, text=True, capture_output=True)
    assert "cases passed" in result.stdout, result.stdout

# CPU numerical oracle and run_lmcode's v1 manifest/frame contract.
subprocess.run([sys.executable, str(root / "tests/dsl_oracle.py"), binary, lmscript], check=True)
subprocess.run([sys.executable, str(root / "tests/check-invocation-manifest.py"), binary, lmscript], check=True)
print("lmcode integration: API/refusal suites, CPU PyTorch oracle and manifest-v1 controls passed")
