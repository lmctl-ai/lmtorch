# lmtorch

`lmtorch` is a small C++17 CPU tensor tool built on libtorch. One executable has two front ends:

- **lmapi operations** (`lmtorch tensor_create ...`, `lmtorch --schema`): twelve stateless operations over safetensors
  artifacts, called from lmscript through the published schema `cli.schema.dev.json`. Output paths are exclusive, so each
  producing call needs a fresh path.
- **`_dsl_` programs** (`lmtorch --dsl prog.dslb`): lmscript compiles a `_dsl_` block against
  `schema/lmtorch.dsl.json` into bytecode; lmtorch loads it with the embedded cpp-dsl-sdk VM and runs it against libtorch
  in one process: tensors, operators, autograd, `nn.Sequential` models, SGD/Adam, safetensors/CSV files.
  API: [docs/dsl-api.md](docs/dsl-api.md); examples: [examples/README.md](examples/README.md).

## Install

Download the executable for your machine from the [latest release](https://github.com/lmctl-ai/lmtorch/releases/latest)
(Linux x64 or arm64, glibc 2.35 or newer; no Python, no other dependency):

```sh
curl -LO https://github.com/lmctl-ai/lmtorch/releases/latest/download/lmtorch-linux-arm64   # or lmtorch-linux-x64
chmod +x lmtorch-linux-arm64 && mv lmtorch-linux-arm64 ~/.local/bin/lmtorch
```

The `.tar.gz` asset holds the same executable with `LICENSE`, `THIRD_PARTY_NOTICES.md` and every bundled license.
Compiling `_dsl_` programs needs `lmscript` >= 0.2.359 (`npm install -g @lmctl-ai/lmctl`, Node.js >= 24.15).

## Build from source

```sh
scripts/build-libtorch.sh          # once: static CPU-only libtorch 2.9.1 from source into ~/.cache/lmtorch (~1.5 h, 4 cores)
scripts/build-native.sh            # build/bin/lmtorch: everything static except glibc
./test.sh                          # the native test suite
```

Requirements: GCC or Clang with C++17, CMake >= 3.27 (PyTorch's minimum), Ninja, git, `nlohmann-json3-dev`, the lmcpp checkout containing
revision `f6c6f51797912a369d6b27cf9e96f148a9a0e4d1` (`LMCPP_DIR`, default `../../lmctl-ai/lmcpp` or `../lmcpp`), and for the
libtorch build only a Python 3 with `pyyaml` and `typing_extensions` (torchgen code generation; nothing Python is linked).
Compiling `_dsl_` programs needs `lmscript` >= 0.2.359 (`npm install -g @lmctl-ai/lmctl`, Node.js >= 24.15).

For development, `scripts/build-native.sh --dev` builds `build-dev/bin/lmtorch` against a PyTorch wheel's shared libtorch
(`TORCH_PYTHON`, default `~/.cache/lmtorch/oracle-venv/bin/python`). `test.sh` tests whichever binary exists
(`LMTORCH_BIN` overrides; `LMTORCH_REQUIRE_RELEASE=1` refuses anything but glibc as a shared library) and, when `ORACLE_PYTHON` (or that venv) has
torch, safetensors and numpy, compares both front ends with Python PyTorch. Details and decisions:
[durable-memory/native-build.md](durable-memory/native-build.md), [durable-memory/dsl-host.md](durable-memory/dsl-host.md).

```sh
export PATH="$PWD/build/bin:$PATH"
lmtorch --dsl-schema                       # the DSL API, grouped by type
lmscript examples/classify-moons-dsl.lms    # a four-step train/evaluate/predict pipeline
lmtorch tensor_create --shape 2 --shape 2 --fill 1 --out artifacts/a.safetensors   # an lmapi operation
```

All artifact paths, in both front ends, must stay inside the working directory.

CI (`.github/workflows/ci.yml`) builds and tests the release and development forms on Linux x64 and arm64; a `v*` tag
publishes the release assets (`scripts/package-release.sh`). License: MIT (`LICENSE`); bundled components:
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
