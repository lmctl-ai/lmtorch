# Native tests

`./test.sh` is the primary CPU-only gate. It runs on Linux, macOS and Windows (Git Bash on Windows). The release matrix
builds static CPU libtorch; development jobs use CPU PyTorch wheels for the numerical oracle.

The gate checks:

1. Every vendored lmcode-cpp file matches the reviewed full SHA and per-file hashes in `sdk.lock`.
2. The embedded host schema matches `schema/lmtorch.dsl.json`, the compiler and registry signatures agree, and the
   host-flag negative controls reject retired options before effects.
3. Invocation manifest v1 and the terminal return frame work through both direct host calls and `run_lmcode`. Dry-mode
   controls preserve an existing sentinel and create no safetensors artifact, while real-mode controls assert those effects. Missing/nonboolean `dry` fails with exit 2 before dispatch, without overwriting the sentinel.
4. C++ safetensors unit vectors cover round trips, atomic no-replace writes, permissions, malformed headers, BOOL data,
   offset/length bounds, integer overflow and duplicate keys. Valid headers over 1 MiB and ranks above 32 round-trip. `check-unbounded-artifacts.py` additionally exercises artifacts and CSV text above 256 MiB, output saves, and zero-storage shapes above the former element ceiling on every native OS. Model dimensions/padding, tensor JSON results above one million values, and long error messages also have regression controls.
5. `tests/dsl/api.lms` checks the Torch, Tensor, Bundle, Model, Optimizer and Out host surface. `tests/dsl/negative.lms`
   checks load/run/application refusals, locations, file containment, resource failures and multi-part invocation behavior.
6. All examples run from scratch workspaces and pass `lmscript --check`; compatibility entry filenames now use the same
   lmcode host path as the other examples. Cached `schemas/*.json` must match the local host schema; executable/archive caches and disguised payloads are refused by `tool_cache.py`, with independent controls in `test_tool_cache.py`.
7. When `ORACLE_PYTHON` contains CPU Torch, safetensors and NumPy, `tests/integration.py` checks the numerical oracle and
   model checkpoint round trips against Python PyTorch, along with manifest and return-frame controls.

`LMTORCH_PYTHON_TESTS=ON` registers the native C++ unit suite and the optional Python/lmscript integration suite with
CTest. Set `LMSCRIPT_EXECUTABLE` when `lmscript` is not on PATH.

CI resolves npm `@lmctl-ai/lmctl@latest` into its private prefix and logs the concrete version. Qualification for this fix used 0.2.457. The required invocation boolean is validated by the pinned SDK; there is no fallback for older malformed producers. Artifact lengths are checked against actual file boundaries, format integer ranges and C++ representations, with no application byte or rank ceiling. Explicit caller settings such as Torch thread selection are unchanged.
