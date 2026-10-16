# torchcli

`torchcli` is a small C++17 CPU tensor tool set using libtorch from the CPU
PyTorch wheel (aarch64 on the development host; the amd64 wheel is locked too). Its artifact format is safetensors; output paths are exclusive,
so each producing call needs a fresh path.

## Build and test

From this repository, with the sibling `lmctl-ai/lmcpp` checkout under the
same repositories root at revision `f6c6f51797912a369d6b27cf9e96f148a9a0e4d1`
(`LMCPP_DIR` overrides the path; `scripts/build.sh` refuses another revision or
a dirty checkout), build the test and runtime images for the host's
architecture:

```sh
scripts/build.sh test                                   # torchcli:test
docker run --rm --pull=never --network=none torchcli:test   # ctest --test-dir /build --output-on-failure
scripts/build.sh runtime                                # torchcli:dev, the image the host shim runs
```

`scripts/build.sh` stages a minimal context (lmcpp sources plus this project)
and builds `docker/Dockerfile`, whose base is the multi-architecture index
digest of `python:3.11-slim-bookworm` (its linux/arm64 manifest is the image
every earlier build used); BuildKit picks the native platform. The wheel locks
(`requirements-torch.lock`, `requirements-test.lock`) carry one hash per
architecture (manylinux aarch64 and x86_64) for the binary `torch==2.9.1+cpu`
CPython 3.11 wheel and the other platform wheels, so `pip --require-hashes`
installs the native wheel on either architecture. At most two C++ compile jobs.

Continuous integration (`.github/workflows/ci.yml`) repeats the two builds, the
full test image, a secrets scan, the schema gate (`torchcli --schema` must equal
`cli.schema.dev.json`), an lmts parse of the schema and every recipe through
the host shim (`scripts/run-recipes.sh`; the lmcreator recipe is skipped
without that checkout) on `ubuntu-24.04-arm` and `ubuntu-24.04`, and publishes
nothing. A weekly workflow re-downloads every pinned wheel with its locked hash
on both architectures and reports drift. torchcli was developed and measured on
aarch64; AMD64_STATUS

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

## Train and report

[`examples/train-report.lms`](examples/train-report.lms) combines torchcli training
with lmcreator charts and a DOCX report. It generates a 16-point polynomial
regression set in lmscript, initializes a seeded 2-8-1 MLP, trains for the
requested steps, scores the training and held-out predictions, and writes loss
and prediction charts plus a labeled pipeline diagram. The DOCX embeds both
charts and the diagram.

This cross-tool example expects the lmcreator checkout beside `cppwrapper`
under the same repositories directory, and an aarch64 lmcreator-native helper
cache. From this repository root:

```sh
REPO_ROOT="$(pwd -P)"
CREATOR_BIN="$(cd ../../lmctlhq/lmcreator/bin && pwd -P)"
export PATH="$REPO_ROOT/bin:$CREATOR_BIN:$PATH"
export TMPDIR="$REPO_ROOT/artifacts/train-report-tmp"
mkdir -p "$TMPDIR"
export RESVG_BIN="$HOME/.cache/lmcreator-native-aarch64/resvg-embed"
export PAGE_PROBE_BIN="$HOME/.cache/lmcreator-native-aarch64/page-probe"
export PDF_WRITE_BIN="$HOME/.cache/lmcreator-native-aarch64/pdf-write"
export PDF_READ_BIN="$HOME/.cache/lmcreator-native-aarch64/pdf-read"
export FIELD_RENDER_BIN="$HOME/.cache/lmcreator-native-aarch64/field-render"
lmscript --check examples/train-report.lms
lmscript examples/train-report.lms artifacts/train-report 60 0.05 11
```

Arguments are `[run_dir, steps, lr, seed]`, defaulting to
`artifacts/train-report`, `60`, `0.05`, and `11`. A representative 60-step run
reported first/last loss `8.315850` / `0.078491`, training MSE `0.071583`, and
R² `0.989220`. It writes `charts/{loss-vs-step,predicted-vs-target}.{png,svg}`
and chart metadata, `diagram/training-pipeline.{png,svg}`, and
`train-report.docx` with three embedded images (plus its Markdown source).
