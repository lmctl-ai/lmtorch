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

Download the executable for your machine from the [latest release](https://github.com/lmctl-ai/lmtorch/releases/latest).
Each one is a single self-contained file: CPU-only libtorch is linked in, no Python and no other runtime is needed.

| Asset | Runs on |
|---|---|
| `lmtorch-linux-x64`, `lmtorch-linux-arm64` | Linux with glibc 2.35 or newer (Ubuntu 22.04, Debian 12, RHEL 10, ...) |
| `lmtorch-macos-arm64`, `lmtorch-macos-x64` | macOS 13 Ventura or newer (Apple silicon, Intel) |
| `lmtorch-windows-x64.exe` | Windows 10 / Server 2016 or newer, x64; no Visual C++ redistributable needed |

```sh
# Linux (or lmtorch-linux-x64)
curl -LO https://github.com/lmctl-ai/lmtorch/releases/latest/download/lmtorch-linux-arm64
chmod +x lmtorch-linux-arm64 && mv lmtorch-linux-arm64 ~/.local/bin/lmtorch

# macOS (or lmtorch-macos-x64); curl downloads carry no quarantine attribute
curl -LO https://github.com/lmctl-ai/lmtorch/releases/latest/download/lmtorch-macos-arm64
chmod +x lmtorch-macos-arm64 && mv lmtorch-macos-arm64 /usr/local/bin/lmtorch
```

```powershell
# Windows (PowerShell)
New-Item -ItemType Directory -Force "$env:LOCALAPPDATA\lmtorch" | Out-Null
Invoke-WebRequest https://github.com/lmctl-ai/lmtorch/releases/latest/download/lmtorch-windows-x64.exe -OutFile "$env:LOCALAPPDATA\lmtorch\lmtorch.exe"
# then add %LOCALAPPDATA%\lmtorch to PATH
```

`lmtorch-<platform>.sha256` lists the checksums. The archive (`lmtorch-<platform>.tar.gz`, `lmtorch-windows-x64.zip` on
Windows) holds the same executable with `LICENSE`, `THIRD_PARTY_NOTICES.md` and every bundled license. The binaries are
not code-signed: a macOS download through a browser needs `xattr -d com.apple.quarantine lmtorch`, and Windows
SmartScreen may ask once. Compiling `_dsl_` programs needs `lmscript` >= 0.2.359 (`npm install -g @lmctl-ai/lmctl`,
Node.js >= 24.15).

## Build from source

```sh
scripts/build-libtorch.sh          # once: static CPU-only libtorch 2.9.1 from source into ~/.cache/lmtorch (~1.5 h, 4 cores)
scripts/build-native.sh            # build/bin/lmtorch: libtorch and the C++ runtime static, only system libraries shared
./test.sh                          # the native test suite
```

Requirements: a C++17 compiler (GCC on Linux, AppleClang on macOS, MSVC 2022 on Windows: libtorch does not build with
MinGW), CMake >= 3.27 (PyTorch's minimum), Ninja, git, nlohmann/json (`nlohmann-json3-dev`, `brew install nlohmann-json`,
`vcpkg install nlohmann-json` with `CMAKE_PREFIX_PATH` pointing at it), the lmcpp checkout containing
revision `93a1847c081894ab1e9ed02ad5ea6350b5d5bd06` (`LMCPP_DIR`, default `../../lmctl-ai/lmcpp` or `../lmcpp`), and for the
libtorch build only a Python 3 with `pyyaml` and `typing_extensions` (torchgen code generation; nothing Python is linked).
On Windows the scripts run under Git Bash inside a Visual Studio x64 developer environment (`vcvars64.bat`) and build
with the static MSVC runtime (`/MT`); on macOS `MACOSX_DEPLOYMENT_TARGET` (default 13.0) sets the floor.
Compiling `_dsl_` programs needs `lmscript` >= 0.2.359 (`npm install -g @lmctl-ai/lmctl`, Node.js >= 24.15).

For development, `scripts/build-native.sh --dev` builds `build-dev/bin/lmtorch` against a PyTorch wheel's shared libtorch
(`TORCH_PYTHON`, default `~/.cache/lmtorch/oracle-venv/bin/python`). `test.sh` tests whichever binary exists
(`LMTORCH_BIN` overrides; `LMTORCH_REQUIRE_RELEASE=1` refuses any shared library that is not part of the operating
system: glibc's own on Linux, libc++ and libSystem on macOS, Windows system DLLs) and, when `ORACLE_PYTHON` (or that venv) has
torch, safetensors and numpy, compares both front ends with Python PyTorch. Details and decisions:
[durable-memory/native-build.md](durable-memory/native-build.md), [durable-memory/dsl-host.md](durable-memory/dsl-host.md).

```sh
export PATH="$PWD/build/bin:$PATH"
lmtorch --dsl-schema                       # the DSL API, grouped by type
lmscript examples/classify-moons-dsl.lms    # a four-step train/evaluate/predict pipeline
lmtorch tensor_create --shape 2 --shape 2 --fill 1 --out artifacts/a.safetensors   # an lmapi operation
```

All artifact paths, in both front ends, must stay inside the working directory.

CI (`.github/workflows/ci.yml`) builds and tests the release form on Linux x64 and arm64, macOS arm64 and x64 and
Windows x64 (plus the development form on Linux); a `v*` tag publishes the release assets (`scripts/package-release.sh`). License: MIT (`LICENSE`); bundled components:
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
