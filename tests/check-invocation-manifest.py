#!/usr/bin/env python3
"""Offline controls for run_lmcode, invocation manifest v1 and the terminal return frame."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

binary = str(Path(sys.argv[1]).resolve())
lmscript = sys.argv[2]
root = Path(__file__).resolve().parents[1]
schema = str(root / "schema" / "lmtorch.dsl.json")

wrapper_source = r'''import json
import os
from pathlib import Path
import subprocess
import sys

host, mode, *args = sys.argv[1:]
manifest_path = Path(os.environ["LMCODE_INVOCATION_FILE"])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
assert manifest["version"] == 1 and manifest["mode"] == "single"
manifest["dry"] = mode == "dry"
manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
child = subprocess.run([host, *args], capture_output=True)
lines = child.stdout.splitlines()
starts = [i for i, line in enumerate(lines) if line == b"LMCODE_RETURN_BEGIN"]
ends = [i for i, line in enumerate(lines) if line == b"LMCODE_RETURN_END"]
assert len(starts) == len(ends) == 1 and ends[0] == starts[0] + 2, (child.returncode, child.stdout, child.stderr)
envelope = json.loads(lines[starts[0] + 1])
Path(mode + "-frame.json").write_text(json.dumps(envelope), encoding="utf-8")
sys.stdout.buffer.write(child.stdout)
sys.stderr.buffer.write(child.stderr)
sys.exit(child.returncode)
'''

with tempfile.TemporaryDirectory(prefix="lmtorch-invocation-") as scratch:
    work = Path(scratch)
    wrapper = work / "host-wrapper.py"
    driver = work / "run-lmcode.lms"
    wrapper.write_text(wrapper_source, encoding="utf-8")
    driver.write_text(
        'BIN = ARGV[0]; WRAPPER = ARGV[1]; PYTHON = ARGV[2];\n'
        f'p = _lmcode_ schema={json.dumps(schema)};\n'
        'out.write_text("manifest-sentinel.txt", "real effect");\n'
        'torch.tensor("[1, 2]").save("manifest-tensor.safetensors");\n'
        'out.set("marker", "real");\n'
        'end _lmcode_\n'
        'dry = run_lmcode(p, {command: [PYTHON, WRAPPER, BIN, "dry", "--lmcode", "$DSLFILE"]});\n'
        'write_file("dry-result.json", json_stringify(dry));\n'
        'real = run_lmcode(p, {command: [PYTHON, WRAPPER, BIN, "real", "--lmcode", "$DSLFILE"]});\n'
        'write_file("real-result.json", json_stringify(real));\n',
        encoding="utf-8",
    )
    env = dict(os.environ)
    env["HOME"] = str(work / "home")
    env["LMCTL_HOME"] = str(work / "home" / ".lmctl")
    Path(env["HOME"]).mkdir()
    subprocess.run([lmscript, str(driver), binary, str(wrapper), sys.executable], cwd=work, env=env,
                   check=True, capture_output=True, text=True)

    dry_result = json.loads((work / "dry-result.json").read_text(encoding="utf-8"))
    dry_frame = json.loads((work / "dry-frame.json").read_text(encoding="utf-8"))
    assert dry_result == {"ok": True, "result": {}, "dry": True}, dry_result
    assert dry_frame == {"ok": True, "result": {}, "dry": True}, dry_frame
    assert not (work / "manifest-sentinel.txt").exists()
    assert not (work / "manifest-tensor.safetensors").exists()

    real_result = json.loads((work / "real-result.json").read_text(encoding="utf-8"))
    real_frame = json.loads((work / "real-frame.json").read_text(encoding="utf-8"))
    assert real_result == {"ok": True, "result": {"marker": "real"}}, real_result
    assert real_frame == {"ok": True, "result": {"marker": "real"}}, real_frame
    assert (work / "manifest-sentinel.txt").read_text(encoding="utf-8") == "real effect"
    assert (work / "manifest-tensor.safetensors").is_file()

print("PASS run_lmcode manifest v1, dry/real effects and terminal return frames")
