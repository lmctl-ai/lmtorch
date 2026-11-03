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
if mode == "legacy":
    manifest.pop("dry", None)
else:
    manifest["dry"] = mode == "dry"
controlled_manifest = Path.cwd() / (mode + "-invocation.json")
controlled_manifest.write_text(json.dumps(manifest), encoding="utf-8")
env = os.environ.copy()
env["LMCODE_INVOCATION_FILE"] = str(controlled_manifest.resolve())
run_dir = Path.cwd() / (mode + "-work")
run_dir.mkdir(exist_ok=True)
child = subprocess.run([host, *args], capture_output=True, env=env, cwd=run_dir)
lines = child.stdout.splitlines()
starts = [i for i, line in enumerate(lines) if line == b"LMCODE_RETURN_BEGIN"]
ends = [i for i, line in enumerate(lines) if line == b"LMCODE_RETURN_END"]
assert len(starts) == len(ends) == 1 and ends[0] == starts[0] + 2, (child.returncode, child.stdout, child.stderr)
envelope = json.loads(lines[starts[0] + 1])
Path(mode + "-frame.json").write_text(json.dumps(envelope), encoding="utf-8")
# run_lmcode itself submitted dry:false and rejects a host dry marker; strip only for its decoder while
# preserving the exact host frame above for the protocol assertion.
reported = dict(envelope)
reported.pop("dry", None)
relayed = b"LMCODE_RETURN_BEGIN\n" + json.dumps(reported, separators=(",", ":")).encode() + b"\nLMCODE_RETURN_END\n"
sys.stdout.buffer.write(relayed)
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
        'write_file("real-result.json", json_stringify(real));\n'
        'legacy = run_lmcode(p, {command: [PYTHON, WRAPPER, BIN, "legacy", "--lmcode", "$DSLFILE"]});\n'
        'write_file("legacy-result.json", json_stringify(legacy));\n',
        encoding="utf-8",
    )
    env = dict(os.environ)
    env["HOME"] = str(work / "home")
    env["LMCTL_HOME"] = str(work / "home" / ".lmctl")
    Path(env["HOME"]).mkdir()
    compiler = subprocess.run([lmscript, str(driver), binary, str(wrapper), sys.executable], cwd=work, env=env,
                              capture_output=True, text=True)
    if compiler.returncode:
        raise AssertionError(f"run_lmcode dry/real driver failed ({compiler.returncode}):\n"
                             f"stdout:\n{compiler.stdout}\nstderr:\n{compiler.stderr}")

    dry_result = json.loads((work / "dry-result.json").read_text(encoding="utf-8"))
    dry_frame = json.loads((work / "dry-frame.json").read_text(encoding="utf-8"))
    assert dry_result == {"ok": True, "result": {}}, dry_result
    assert dry_frame == {"ok": True, "result": {}, "dry": True}, dry_frame
    assert not (work / "dry-work" / "manifest-sentinel.txt").exists()
    assert not (work / "dry-work" / "manifest-tensor.safetensors").exists()

    real_result = json.loads((work / "real-result.json").read_text(encoding="utf-8"))
    real_frame = json.loads((work / "real-frame.json").read_text(encoding="utf-8"))
    assert real_result == {"ok": True, "result": {"marker": "real"}}, real_result
    assert real_frame == {"ok": True, "result": {"marker": "real"}}, real_frame
    assert (work / "real-work" / "manifest-sentinel.txt").read_text(encoding="utf-8") == "real effect"
    assert (work / "real-work" / "manifest-tensor.safetensors").is_file()

    legacy_result = json.loads((work / "legacy-result.json").read_text(encoding="utf-8"))
    legacy_frame = json.loads((work / "legacy-frame.json").read_text(encoding="utf-8"))
    assert legacy_result == {"ok": True, "result": {"marker": "real"}}, legacy_result
    assert legacy_frame == {"ok": True, "result": {"marker": "real"}}, legacy_frame
    assert (work / "legacy-work" / "manifest-sentinel.txt").read_text(encoding="utf-8") == "real effect"

print("PASS run_lmcode manifest v1, optional false dry compatibility, dry/real effects and terminal frames")
