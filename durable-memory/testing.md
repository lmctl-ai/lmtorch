# Testing

`./test.sh` is the native suite (about one minute on the dev build). It runs under bash on Linux, macOS and Windows
(Git Bash: `.exe` binaries, drive-letter paths for anything a native program reads; see native-build.md). In order:

1. `scripts/sync-sdk.sh --verify` (every vendored lmcode-cpp file equals `sdk.lock`); `schema/lmtorch.dsl.json` equals
   `schema/lmtorch.host.json` byte for byte; the compiler emits bytecode v5; a std
   block (strings, math, lists, files, `Bundle.keys().join`) runs.
2. Schema gates: `--dsl-schema --json` equals `schema/lmtorch.dsl.json` byte for byte; `--dsl-signatures` equals
   `lmscript --dsl-signatures`; the `--dsl-schema` summary lists every member; lmapi `--schema` equals `cli.schema.dev.json`.
3. Release form: Linux: the binary's NEEDED libraries must be glibc's own (libc, libm, libpthread, libdl, librt, ld-linux)
   or none, and the gate prints the glibc floor; macOS: `otool -L` lists only /usr/lib/libc++.1.dylib and
   /usr/lib/libSystem.B.dylib, and the gate prints minos; Windows: `dumpbin -dependents` lists only Windows system DLLs
   (no vcruntime/msvcp/ucrtbase/api-ms-win-crt). `LMTORCH_REQUIRE_RELEASE=1` (CI release jobs) fails anything else; a
   dev build only notes it. Before all gates a smoke run reports a binary that cannot start with its exit status (and,
   on Windows, the imported DLLs and a cdb stack).
4. `lmtorch_unit` (C++ safetensors reader/writer, malformed headers, modes, BOOL payload bytes).
5. `tests/dsl/api.lms`: one DSL program with an `out.check` per behavior (137 checks: creation, operators incl. reversed
   forms, broadcasting, reductions, shapes, losses, autograd accumulation, RNG reproducibility, every layer kind through
   save/load_model, optimizer steps, bundles/CSV/files incl. BOM and CRLF, BOOL round trip, view write-through, live
   Model.param, float layer arguments through load_model, handle release, text/JSON forms). A failure exits 4 and names the
   check and its DSL line.
6. `tests/dsl/negative.lms`: load refusals (bad magic, missing, truncated, trailing bytes, schema-hash mismatch via
   `tests/dsl/fixtures/other-schema-bytes.dsl.json` = the schema plus one trailing space, a bad later file stops the run
   before the first runs), usage errors (exit 2, no result line), and run/app failures asserting exit, phase, code, a message
   fragment and the reported DSL line (60 cases, incl. ragged literals, corrupt BOOL bytes written with printf, unknown
   saved layers, unexpected tensors in Model.load, double backward, non-contiguous set, and a two-file run proving grad
   mode resets per file while the RNG continues). Expected lines are the line before each case's `end _lmcode_`; when
   inserting a case, recompute them (a small Python pass did this: find the preceding `end _lmcode_`, use its line number - 1),
   because the faulting statement is always that last body line.
6b. test.sh also pins the front-end boundary: a BOOL artifact written by the DSL is an lmapi INVALID_DTYPE refusal (exit 2).
7. An ill-typed chunk is a located lmscript compile error (exit 2).
8. Every `*-dsl.lms` example and the three lmapi recipes run from a scratch workspace and their key output lines are
   checked to six significant digits (0.886624/0.057287, R² 0.989220: the static build's BLAS-free GEMM changes the last
   float32 bits, so full-precision lmapi output differs between builds); every example
   passes `lmscript --check` (train-report.lms needs lmcreator and is skipped).
9. With `ORACLE_PYTHON` (default `~/.cache/lmtorch/oracle-venv/bin/python`): `tests/dsl_oracle.py` (seeded rand/randn/
   randint/randperm and nn.Linear/Conv2d init bit-identical to Python; Adam, SGD momentum, conv, softmax, GELU, losses
   within 1e-5; DSL-saved models load into Python nn.Sequential strictly and Python-saved ones through torch.load_model)
   and `tests/integration.py` (the original 12-operation oracle and 21 regression cases, now against the native binary).

CTest (`LMTORCH_PYTHON_TESTS=ON`) registers the unit test and integration.py; test.sh is the primary entry point.
