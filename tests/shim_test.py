import json
import os
import pathlib
import subprocess
import sys
import tempfile

shim = pathlib.Path(sys.argv[1]).resolve()
home = pathlib.Path.home().resolve()
with tempfile.TemporaryDirectory(prefix="torchcli-shim-", dir=home) as td:
    root = pathlib.Path(td)
    fake = root / "fakebin"
    fake.mkdir()
    tracked = root / "created.txt"
    args_file = root / "args.txt"
    mktemp = fake / "mktemp"
    mktemp.write_text("#!/bin/sh\np=$(python3 -c 'import tempfile; print(tempfile.mktemp())')\n: >\"$p\"\necho \"$p\" >>\"$TRACKED\"\necho \"$p\"\n")
    mktemp.chmod(0o755)
    docker = fake / "docker"
    docker.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\" >\"$ARGS_FILE\"\ncase \"${FAKE_DOCKER_CASE:-crash}\" in\n  ok) echo '{\"ok\":true}'; exit 0;;\n  daemon) echo 'docker: Cannot connect to the Docker daemon' >&2; exit 125;;\n  partial) echo '{'; exit 143;;\n  crash) exit \"${FAKE_DOCKER_RC:-139}\";;\nesac\n")
    docker.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = str(fake) + os.pathsep + env["PATH"]
    env["TRACKED"] = str(tracked)
    env["ARGS_FILE"] = str(args_file)
    tracked.write_text("")

    colon_dir = root / "project:with-colon"
    colon_dir.mkdir()
    env["FAKE_DOCKER_CASE"] = "ok"
    p = subprocess.run([str(shim), "--schema"], cwd=colon_dir, env=env, text=True, capture_output=True)
    assert p.returncode == 0 and json.loads(p.stdout) == {"ok": True}, (p.returncode, p.stdout, p.stderr)
    argv = args_file.read_text().splitlines()
    mount_index = argv.index("--mount")
    assert argv[mount_index + 1] == f"type=bind,src={colon_dir},dst={colon_dir}", argv
    assert "-v" not in argv

    env["FAKE_DOCKER_CASE"] = "crash"
    env["FAKE_DOCKER_RC"] = "139"
    p = subprocess.run([str(shim), "--schema"], cwd=colon_dir, env=env, text=True, capture_output=True)
    body = json.loads(p.stdout)
    assert p.returncode == 1 and body["error"]["code"] == "INTERNAL_ERROR", (p.returncode, p.stdout, p.stderr)
    assert body["error"]["data"] == {"tool_code": "TOOL_CRASH", "exit": 139}, body
    env["FAKE_DOCKER_CASE"] = "partial"
    p = subprocess.run([str(shim), "--schema"], cwd=colon_dir, env=env, text=True, capture_output=True)
    body = json.loads(p.stdout)
    assert p.returncode == 1 and body["error"]["data"] == {"tool_code": "TOOL_CRASH", "exit": 143}, (p.returncode, p.stdout)
    env["FAKE_DOCKER_CASE"] = "crash"
    env["FAKE_DOCKER_RC"] = "125"
    p = subprocess.run([str(shim), "--schema"], cwd=colon_dir, env=env, text=True, capture_output=True)
    body = json.loads(p.stdout)
    assert p.returncode == 1 and body["error"]["code"] == "INTERNAL_ERROR" and body["error"]["data"]["exit"] == 125, (p.returncode, p.stdout)
    env["FAKE_DOCKER_CASE"] = "daemon"
    p = subprocess.run([str(shim), "--schema"], cwd=colon_dir, env=env, text=True, capture_output=True)
    assert p.returncode == 8 and json.loads(p.stdout)["error"]["code"] == "SERVICE_UNAVAILABLE", (p.returncode, p.stdout)

    for unsafe in (pathlib.Path("/"), home, pathlib.Path(tempfile.gettempdir())):
        p = subprocess.run([str(shim), "--schema"], cwd=unsafe, env=env, text=True, capture_output=True)
        assert p.returncode == 2, (unsafe, p.returncode, p.stdout, p.stderr)
        envelope = json.loads(p.stdout)
        assert envelope["success"] is False and envelope["error"]["code"] == "INVALID_REQUEST", envelope
    created = tracked.read_text().splitlines()
    assert len(created) == 10 and all(not pathlib.Path(path).exists() for path in created), created
print("torchcli shim enforces cwd boundary, supports colon mounts, maps crashes, and cleans temporary files")
