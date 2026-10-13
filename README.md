# torchcli

`torchcli` is a small C++17 CPU tensor tool set using libtorch from the ARM64
PyTorch wheel. Its artifact format is safetensors; output paths are exclusive,
so each producing call needs a fresh path.

## Build and test

From this repository, with the sibling `lmctl-ai/lmcpp` checkout under the
same repositories root (lmcpp API baseline d3e9f9f), build the native ARM64
test and runtime images:

```sh
docker build --platform linux/arm64 -f docker/Dockerfile --target test -t torchcli:test ~/repos
docker run --rm --network=none torchcli:test ctest --test-dir /build --output-on-failure
docker build --platform linux/arm64 -f docker/Dockerfile --target runtime -t torchcli:dev ~/repos
```

The build uses the binary `torch==2.9.1+cpu` CPython 3.11 wheel and at most two
C++ compile jobs. `docker/Dockerfile.dockerignore` restricts the repositories-root
build context to lmcpp and this project.

## Use the host shim

Put the repository's absolute `bin` directory on `PATH`, then invoke `torchcli`
from the artifact workspace. The shim starts a fresh offline container and
mounts only the current directory read-write at the same absolute path using
Docker's `--mount` form. It requires cwd to be a strict descendant of `$HOME`
and refuses `/`, `$HOME`, and paths outside `$HOME`. C++ artifact reads and
writes reject paths whose canonical target escapes cwd:

```sh
export PATH="$PWD/bin:$PATH"
export TMPDIR="$PWD/.tmp"
mkdir -p "$TMPDIR" artifacts/run01
torchcli --schema > cli.schema.dev.json
torchcli tensor_create --shape 2 --shape 2 --fill 1 --out artifacts/run01/a.safetensors
torchcli tensor_matmul --a artifacts/run01/a.safetensors#tensor --b artifacts/run01/a.safetensors#tensor --out artifacts/run01/b.safetensors
torchcli tensor_inspect --input artifacts/run01/b.safetensors#tensor
```

The shim maps recognized Docker daemon/image startup failures to
`SERVICE_UNAVAILABLE`. A container crash or signal exit returns protocol
`INTERNAL_ERROR` with exit 1 and `data.tool_code: "TOOL_CRASH"` plus the
container exit status; raw signal statuses are not used as protocol exits.

The schema publishes the twelve operations implemented in this stage. Recipe
examples accept the run directory as their first argument:

```sh
lmscript --check examples/tensor-chain.lms
lmscript examples/tensor-chain.lms artifacts/recipe-chain
lmscript --check examples/grad-check.lms
lmscript examples/grad-check.lms artifacts/recipe-grad
lmscript --check examples/mlp-train.lms
lmscript examples/mlp-train.lms artifacts/recipe-mlp
```

Recipe outputs and all tensor references must stay inside the mounted workspace.
See [docs/parity.md](docs/parity.md) for implemented behavior and boundaries.
