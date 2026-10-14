import json
import os
import pathlib
import subprocess
import signal
import time
import sys
import tempfile

shim = pathlib.Path(sys.argv[1]).resolve()
home = pathlib.Path.home().resolve()
with tempfile.TemporaryDirectory(prefix="torchcli-shim-", dir=home) as td:
    root = pathlib.Path(td)
    fake = root / "fakebin"
    fake.mkdir()
    args_file = root / "args.txt"
    block_file = root / "block-started"
    control_file = root / "docker-controls"
    docker = fake / "docker"
    docker.write_text("""#!/bin/sh
case "$1" in
  kill) echo kill >>"$CONTROL_FILE"; exit 0;;
  rm) echo rm >>"$CONTROL_FILE"; exit 0;;
  run) ;;
  *) exit 0;;
esac
printf '%s\n' "$@" >"$ARGS_FILE"
case "${FAKE_DOCKER_CASE:-crash}" in
  ok) echo '{"ok":true}'; exit 0;;
  block) : >"$BLOCK_FILE"; while :; do sleep 1; done;;
  missing) echo 'docker: No such image: missing' >&2; exit 125;;
  daemon) echo 'docker: Cannot connect to the Docker daemon' >&2; exit 125;;
  partial) echo '{'; exit 143;;
  crash) exit "${FAKE_DOCKER_RC:-139}";;
esac
""")
    docker.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = str(fake) + os.pathsep + env["PATH"]
    env["ARGS_FILE"] = str(args_file)
    env["BLOCK_FILE"] = str(block_file)
    env["CONTROL_FILE"] = str(control_file)

    colon_dir = root / "project:with-colon"
    colon_dir.mkdir()
    env["FAKE_DOCKER_CASE"] = "ok"
    p = subprocess.run([str(shim), "--schema"], cwd=colon_dir, env=env, text=True, capture_output=True)
    assert p.returncode == 0 and json.loads(p.stdout) == {"ok": True}, (p.returncode, p.stdout, p.stderr)
    argv = args_file.read_text().splitlines()
    mount_index = argv.index("--mount")
    assert argv[mount_index + 1] == f"type=bind,src={colon_dir},dst={colon_dir}", argv
    assert "--pull=never" in argv
    assert "TORCHCLI_READONLY" in argv and "TORCHCLI_DRYRUN" in argv
    assert "-v" not in argv

    env["FAKE_DOCKER_CASE"] = "block"
    blocked = subprocess.Popen([str(shim), "--schema"], cwd=colon_dir, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    deadline = time.monotonic() + 5
    while not block_file.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert block_file.exists(), "mock docker did not start"
    os.kill(blocked.pid, signal.SIGINT)
    blocked.communicate(timeout=3)
    assert blocked.returncode == 130, blocked.returncode
    assert "kill" in control_file.read_text() and "rm" in control_file.read_text(), control_file.read_text()
    assert not list(colon_dir.glob(".torchcli-tmp.*")), "signal trap leaked workspace temp directory"

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

    peers = []
    for index in range(4):
        peer_env = dict(env, FAKE_DOCKER_CASE="ok", ARGS_FILE=str(root / f"args-{index}"))
        peers.append(subprocess.Popen([str(shim), "--schema"], cwd=colon_dir, env=peer_env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE))
    for peer in peers:
        stdout, stderr = peer.communicate(timeout=5)
        assert peer.returncode == 0, (peer.returncode, stdout, stderr)
    names = []
    for index in range(4):
        peer_argv = (root / f"args-{index}").read_text().splitlines()
        names.append(peer_argv[peer_argv.index("--name") + 1])
    assert len(set(names)) == 4, names
    assert not list(colon_dir.glob(".torchcli-tmp.*"))
    env["FAKE_DOCKER_CASE"] = "missing"
    p = subprocess.run([str(shim), "--schema"], cwd=colon_dir, env=env, text=True, capture_output=True)
    assert p.returncode == 8 and "unavailable" in p.stdout.lower(), (p.returncode, p.stdout, p.stderr)

    # Mount grammar metacharacters must be rejected before invoking Docker.
    attack = root / ("x/,src=/tmp/pwn,dst=" + str(root / "x"))
    attack.parent.mkdir(parents=True, exist_ok=True)
    attack.mkdir()
    args_file.unlink(missing_ok=True)
    p = subprocess.run([str(shim), "--schema"], cwd=attack, env=env, text=True, capture_output=True)
    assert p.returncode == 2 and "comma" in p.stdout.lower(), (p.returncode, p.stdout, p.stderr)
    assert not args_file.exists() and not pathlib.Path("/tmp/pwn").exists()
    for bad_name in ("comma,name", 'quote"name', "equals=name", "line\nbreak", "control\x01byte"):
        bad_dir = root / ("unsafe-" + bad_name)
        bad_dir.mkdir()
        p = subprocess.run([str(shim), "--schema"], cwd=bad_dir, env=env, text=True, capture_output=True)
        assert p.returncode == 2 and "control" in p.stdout.lower() or p.returncode == 2 and "comma" in p.stdout.lower(), (bad_name, p.returncode, p.stdout)
        assert not args_file.exists()

    for unsafe in (pathlib.Path("/"), home, pathlib.Path(tempfile.gettempdir())):
        p = subprocess.run([str(shim), "--schema"], cwd=unsafe, env=env, text=True, capture_output=True)
        assert p.returncode == 2, (unsafe, p.returncode, p.stdout, p.stderr)
        envelope = json.loads(p.stdout)
        assert envelope["success"] is False and envelope["error"]["code"] == "INVALID_REQUEST", envelope
print("torchcli shim boundary, concurrency, cancellation, docker arguments, and failure mapping passed")
