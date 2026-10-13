import os
import pathlib
import subprocess
import sys
import tempfile

shim = pathlib.Path(sys.argv[1]).resolve()
with tempfile.TemporaryDirectory(prefix="torchcli-shim-") as td:
    root = pathlib.Path(td)
    fake = root / "fakebin"
    fake.mkdir()
    tracked = root / "created.txt"
    mktemp = fake / "mktemp"
    mktemp.write_text("#!/bin/sh\np=$(python3 -c 'import tempfile; print(tempfile.mktemp())')\n: >\"$p\"\necho \"$p\" >>\"$TRACKED\"\necho \"$p\"\n")
    mktemp.chmod(0o755)
    docker = fake / "docker"
    docker.write_text("#!/bin/sh\nif [ \"${FAKE_DOCKER_CASE:-}\" = daemon ]; then echo 'docker: Cannot connect to the Docker daemon' >&2; exit 125; fi\nexit \"${FAKE_DOCKER_RC:-139}\"\n")
    docker.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = str(fake) + os.pathsep + env["PATH"]
    env["TRACKED"] = str(tracked)
    tracked.write_text("")
    p = subprocess.run([str(shim), "--schema"], cwd=td, env=env, text=True, capture_output=True)
    assert p.returncode == 139, (p.returncode, p.stdout, p.stderr)
    assert '"code":"TOOL_CRASH"' in p.stdout, p.stdout
    env["FAKE_DOCKER_RC"]="125"
    p = subprocess.run([str(shim), "--schema"], cwd=td, env=env, text=True, capture_output=True)
    assert p.returncode == 125 and '"code":"TOOL_CRASH"' in p.stdout, (p.returncode,p.stdout,p.stderr)
    env["FAKE_DOCKER_CASE"]="daemon"
    p = subprocess.run([str(shim), "--schema"], cwd=td, env=env, text=True, capture_output=True)
    assert p.returncode == 8 and '"code":"SERVICE_UNAVAILABLE"' in p.stdout, (p.returncode,p.stdout,p.stderr)
    created = tracked.read_text().splitlines()
    assert len(created) == 6 and all(not pathlib.Path(path).exists() for path in created), created
print("torchcli shim separates daemon failures from crashes and cleans temporary files")
