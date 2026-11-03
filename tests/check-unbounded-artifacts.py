#!/usr/bin/env python3
"""Real CPU host round trips above the former implicit byte/rank limits."""
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

binary = str(Path(sys.argv[1]).resolve())
lmscript = sys.argv[2]
schema = str(Path(__file__).resolve().parents[1] / "schema/lmtorch.dsl.json")
with tempfile.TemporaryDirectory(prefix="lmtorch-unbounded-") as scratch:
    work = Path(scratch)
    env = dict(os.environ, HOME=str(work / "home"), LMCTL_HOME=str(work / "home/.lmctl"))
    Path(env["HOME"]).mkdir()

    def artifact(name, header, size, value=b""):
        encoded = json.dumps(header, separators=(",", ":")).encode()
        encoded += b" " * (-len(encoded) % 8)
        with (work / name).open("wb") as stream:
            stream.write(struct.pack("<Q", len(encoded)))
            stream.write(encoded)
            stream.write(value)
            stream.truncate(8 + len(encoded) + size)

    artifact("header.safetensors", {"tensor": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
             "__metadata__": {"padding": "x" * (1024 * 1024 + 1)}}, 4, struct.pack("<f", 7))
    count = 67108865  # F32 bytes exceed 256 MiB; only scalar facts are returned.
    artifact("large-in.safetensors", {"tensor": {"dtype": "F32", "shape": [count], "data_offsets": [0, count * 4]}}, count * 4)
    artifact("rank.safetensors", {"tensor": {"dtype": "F32", "shape": [1] * 33, "data_offsets": [0, 4]}}, 4, struct.pack("<f", 9))
    artifact("message-a.safetensors", {"tensor": {"dtype": "F32", "shape": [1] * 64, "data_offsets": [0, 4]}}, 4, struct.pack("<f", 1))
    artifact("message-b.safetensors", {"tensor": {"dtype": "F32", "shape": [1] * 63 + [2], "data_offsets": [0, 8]}}, 8, struct.pack("<ff", 1, 2))
    with (work / "large.csv").open("wb") as stream:
        # A long valid header, followed by one small row: no huge JSON result or tensor.
        stream.write(b"column")
        for _ in range(256):
            stream.write(b" " * (1024 * 1024))
        stream.write(b"\n7\n")

    cases = [
        ("message", 'torch.stack(torch.list().add(torch.open("message-a.safetensors").get("tensor")).add(torch.open("message-b.safetensors").get("tensor")), 0);', None),
        ("model dimensions", 'out.set("parameters", torch.sequential().linear(1000001, 1).num_parameters());', {"parameters": 1000002}),
        ("padding", 'out.set("layers", torch.sequential().conv2d(1, 1, 1, 1, 1001).layers());', {"layers": 1}),
        ("values", 'out.set("values", torch.ones(1000001));', None),
        ("dimensions", 'out.set("numel", torch.zeros(0, 2147483649).numel());', {"numel": 0}),
        ("header", 'out.set("value", torch.open("header.safetensors").get("tensor").item());', {"value": 7}),
        ("read", 'out.set("numel", torch.open("large-in.safetensors").get("tensor").numel());', {"numel": count}),
        ("write", f'let mark = torch.mark(); torch.zeros({count}).save("large-out.safetensors"); torch.release(mark); '
                  'out.set("numel", torch.open("large-out.safetensors").get("tensor").numel());', {"numel": count}),
        ("rank", 'out.set("rank", torch.open("rank.safetensors").get("tensor").dim());', {"rank": 33}),
        ("text", 'out.set("value", torch.read_csv("large.csv", true).item());', {"value": 7}),
    ]
    failures = []
    for name, body, expected in cases:
        driver = work / (name + ".lms")
        driver.write_text('p = _lmcode_ schema=' + json.dumps(schema) + ';\n' + body + '\nend _lmcode_\n'
                          'r = run_lmcode(p, {command: [' + json.dumps(binary) + ', "--lmcode", "$DSLFILE"]});\n'
                          'println(json_stringify(r));\n', encoding="utf-8")
        result = subprocess.run([lmscript, str(driver)], cwd=work, env=env, capture_output=True, text=True)
        try:
            assert result.returncode == 0, (name, result.stderr)
            actual = json.loads(result.stdout)
            if name == "message":
                assert actual.get("ok") is False and actual.get("phase") == "run", (name, actual)
                assert "at entry 1" in actual["message"], "host error message must retain its tail beyond 300 characters"
            elif name == "values":
                assert actual.get("ok") is True, (name, actual)
                values = actual["result"]["values"]
                assert len(values) == 1000001 and all(value == 1 for value in values), "tensor result must not be capped or truncated"
            else:
                assert actual == {"ok": True, "result": expected}, (name, result.stdout)
            if name == "write":
                assert (work / "large-out.safetensors").stat().st_size > 256 * 1024 * 1024
            print("PASS no implicit " + name + " bound", flush=True)
        except (AssertionError, ValueError) as error:
            failures.append(str(error))
            print("FAIL " + str(error), flush=True)
    assert not failures, failures
