# torchcli status and hand-off (2026-10-20)

## State
- One C++17 binary, two front ends. lmapi: the original 12 operations (lmcpp, `--schema` byte-identical to
  `cli.schema.dev.json`). DSL: `torchcli --dsl prog.dslb` runs lmscript `_dsl_` bytecode (v5, cpp-dsl-sdk VM pinned in
  `sdk.lock`) against `schema/torchcli.dsl.json`: 218 signatures over Torch/Tensor/TensorList/StrList/Bundle/Model/
  Optimizer/Out (`docs/dsl-api.md`, decisions in dsl-host.md).
- Native build without docker or Python at run time: `scripts/build-libtorch.sh` (static CPU-only libtorch 2.9.1 from
  source, `libtorch.lock`) then `scripts/build-native.sh` (fully static). `scripts/build-native.sh --dev` builds against the
  PyTorch wheel. See native-build.md.
- `examples/`: the four lmapi recipes plus a `-dsl` version of each (same numbers: mlp 0.886624 → 0.057287, train-report
  R² 0.989220) and five multi-step DSL tasks (classify-moons, lr-sweep, checkpoint-resume, cnn-bars, regression-csv);
  `examples/lib/torchdsl.lms` is the compile-and-run helper. Index: `examples/README.md`.
- Tests: `./test.sh` (testing.md): 27 gates (26 on a dynamic dev build) including 137 DSL API checks, 60 refusal cases, every
  example, and two PyTorch oracles (DSL: RNG and init bit-identical to Python, training within 1e-5, model files
  interchangeable; lmapi: the original 12-operation oracle, now native). All green on both builds (2026-10-20).
- Bugs fixed this round: an lmcpp::Error thrown inside a DSL call (path containment in read_csv/write_text) aborted the
  process (not a std::exception); the safetensors temp-file error now names the path and errno; integration.py accepts a
  static binary (ldd exits 1), and so does test.sh's own static gate (pipefail); docker staging (`scripts/build.sh`) now copies `schema/`, `third_party/`, `sdk.lock`, and the
  Dockerfile enables `TORCHCLI_PYTHON_TESTS` so its CTest keeps the integration and shim tests.
- Second round (2026-10-20, each with a regression test): `torch.tensor("[3, [1, 2]]")` was accepted and returned
  uninitialized memory (a scalar before a list escaped the ragged check; all scalars must now share the final depth and
  their count must equal the shape); the shared safetensors reader accepted BOOL bytes other than 0/1 (now
  INVALID_ARTIFACT for both front ends); `torch.grad_enabled(false)` in one file silently disabled autograd in later files
  of the same `--dsl` run (grad mode is reset per file; the RNG still continues); saving a parameterless model said
  "put() at least one tensor"; an unknown layer with arguments in saved metadata reported an argument count instead of
  "unknown layer"; CSV files with a UTF-8 byte-order mark failed to parse.
- Static release binary: DONE on aarch64 Linux (2026-10-20): `build/bin/torchcli`, 185 MB (122 MB stripped), statically
  linked, no shared-library dependencies; `TORCHCLI_REQUIRE_STATIC=1 ./test.sh` passes all 27 gates, including both
  PyTorch oracles. amd64 static is untested (native.yml would cover it).

## Not done / open
- CI: `.github/workflows/native.yml` (static with libtorch cache + dev, arm64/amd64, running test.sh) is written but has
  never run; the docker CI (`ci.yml`) was not re-run after the CMake changes (no docker on this host).
- No BLAS: GEMM is ATen's portable kernel. A static OpenBLAS would speed up larger models; linalg/LAPACK ops are absent.
- macOS/Windows native builds are not attempted (libtorch static on those platforms is untested here).
- Licensing: repo is PRIVATE with no LICENSE. The static binary contains libtorch (BSD-3) and its bundled third-party code,
  glibc (LGPL-2.1+, static: relinking materials as lmcad ships them would be needed before redistribution) and the GCC
  runtime (GPLv3 + runtime exception). `THIRD_PARTY_NOTICES.md` lists what is linked; the go-public decision is the owner's.
- Nothing is committed or published by this round unless the owner asks.

## Rules that apply
- Commit identity Mike Ma <mike.ma@lmctl.com>, one Co-Authored-By per contributing model, explicit `git add`, push after every commit. Staging/deployment belongs to another team.
- Private repo: verify locally; gate any paid macOS/Windows CI legs with `github.event.repository.private == false`.
