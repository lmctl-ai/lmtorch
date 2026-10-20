# Testing

`./test.sh` is the native suite (about one minute on the dev build). In order:

1. `scripts/sync-sdk.sh --verify` (vendored `dslvm.hpp` equals `sdk.lock`), and the compiler emits bytecode v5.
2. Schema gates: `--dsl-schema --json` equals `schema/torchcli.dsl.json` byte for byte; `--dsl-signatures` equals
   `lmscript --dsl-signatures`; the `--dsl-schema` summary lists every member; lmapi `--schema` equals `cli.schema.dev.json`.
3. Static check: a statically linked binary must make `ldd` say "not a dynamic executable"; `TORCHCLI_REQUIRE_STATIC=1`
   fails a dynamic build (release gate). The dev build only notes it.
4. `torchcli_unit` (C++ safetensors reader/writer, malformed headers, modes, BOOL payload bytes).
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
   mode resets per file while the RNG continues). Expected lines are the line before each case's `end _dsl_`; when
   inserting a case, recompute them (a small Python pass did this: find the preceding `end _dsl_`, use its line number - 1),
   because the faulting statement is always that last body line.
6b. test.sh also pins the front-end boundary: a BOOL artifact written by the DSL is an lmapi INVALID_DTYPE refusal (exit 2).
7. An ill-typed chunk is a located lmscript compile error (exit 2).
8. Every `*-dsl.lms` example and the three lmapi recipes run from a scratch workspace and their key output lines are
   checked to six significant digits (0.886624/0.057287, R² 0.989220: the static build's BLAS-free GEMM changes the last
   float32 bits, so full-precision lmapi output differs between builds); every example
   passes `lmscript --check` (train-report.lms needs lmcreator and is skipped).
9. With `ORACLE_PYTHON` (default `~/.cache/torchcli/oracle-venv/bin/python`): `tests/dsl_oracle.py` (seeded rand/randn/
   randint/randperm and nn.Linear/Conv2d init bit-identical to Python; Adam, SGD momentum, conv, softmax, GELU, losses
   within 1e-5; DSL-saved models load into Python nn.Sequential strictly and Python-saved ones through torch.load_model)
   and `tests/integration.py` (the original 12-operation oracle and 21 regression cases, now against the native binary).

The docker CTest path (`scripts/build.sh test`, CI) is unchanged in intent: it configures with `TORCHCLI_PYTHON_TESTS=ON`
to register integration.py and shim_test.py. It was not run on this host (no docker).
