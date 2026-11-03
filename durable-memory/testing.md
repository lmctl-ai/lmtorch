# Native tests

`./test.sh` is the primary CPU-only gate. It runs on Linux, macOS and Windows (Git Bash on Windows). The release matrix
builds static CPU libtorch; development jobs use CPU PyTorch wheels for the numerical oracle.

The gate checks:

1. Every vendored lmcode-cpp file matches the reviewed full SHA and per-file hashes in `sdk.lock`.
2. The embedded host schema matches `schema/lmtorch.dsl.json`, the compiler and registry signatures agree, and the
   host-flag negative controls reject retired options before effects.
3. Invocation manifest v1 and the terminal return frame work through both direct host calls and `run_lmcode`. Dry-mode
   controls assert no sentinel file or safetensors artifact is created, while real-mode controls assert those effects.
4. C++ safetensors unit vectors cover round trips, atomic no-replace writes, permissions, malformed headers, BOOL data,
   offset/length bounds and duplicate keys.
5. `tests/dsl/api.lms` checks the Torch, Tensor, Bundle, Model, Optimizer and Out host surface. `tests/dsl/negative.lms`
   checks load/run/application refusals, locations, file containment, resource failures and multi-part invocation behavior.
6. All examples run from scratch workspaces and pass `lmscript --check`; compatibility entry filenames now use the same
   lmcode host path as the other examples.
7. When `ORACLE_PYTHON` contains CPU Torch, safetensors and NumPy, `tests/integration.py` checks the numerical oracle and
   model checkpoint round trips against Python PyTorch, along with manifest and return-frame controls.

`LMTORCH_PYTHON_TESTS=ON` registers the native C++ unit suite and the optional Python/lmscript integration suite with
CTest. Set `LMSCRIPT_EXECUTABLE` when `lmscript` is not on PATH.
