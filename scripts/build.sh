#!/bin/sh
# Build the torchcli images from a staged context: the pinned lmcpp checkout (LMCPP_DIR, default ../../lmctl-ai/lmcpp next to the
# repositories root; must be at the expected revision and clean) and this repository, laid out as docker/Dockerfile expects
# (lmctl-ai/lmcpp and cppwrapper/pytorch). Usage: scripts/build.sh [test|runtime]; runtime is tagged torchcli:dev (the shim's image).
set -eu
root=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
target=${1:-runtime}
case "$target" in test) tag=torchcli:test;; runtime) tag=torchcli:dev;; *) echo "target is test or runtime" >&2; exit 2;; esac
lmcpp=${LMCPP_DIR:-$root/../../lmctl-ai/lmcpp}
expected=f6c6f51797912a369d6b27cf9e96f148a9a0e4d1
[ "$(git -C "$lmcpp" rev-parse HEAD)" = "$expected" ] || { echo "lmcpp revision mismatch: expected $expected" >&2; exit 1; }
[ -z "$(git -C "$lmcpp" status --porcelain --untracked-files=no)" ] || { echo 'lmcpp checkout is dirty' >&2; exit 1; }
context=$(mktemp -d)
trap 'rm -rf "$context"' EXIT HUP INT TERM
mkdir -p "$context/lmctl-ai/lmcpp" "$context/cppwrapper/pytorch"
git -C "$lmcpp" archive "$expected" CMakeLists.txt cmake include src examples/toy | tar -x -C "$context/lmctl-ai/lmcpp"
for path in CMakeLists.txt requirements-torch.lock requirements-test.lock include src tests scripts bin examples docker cli.schema.dev.json; do
  cp -R "$root/$path" "$context/cppwrapper/pytorch/"
done
docker build -f "$context/cppwrapper/pytorch/docker/Dockerfile" --target "$target" -t "$tag" "$context"
