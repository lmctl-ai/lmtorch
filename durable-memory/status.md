# lmtorch status and hand-off (2026-10-20)

Public at https://github.com/lmctl-ai/lmtorch since 2026-10-20: lmctlhq/torchcli (repo id 1417876046) was transferred and
renamed with its full history after an owner-approved audit (no credentials or home paths; history keeps mentions of
private repo/folder names such as lmctlhq/lmcreator, cppwrapper and m4g). Formerly `torchcli`. Renamed to lmtorch (binary, schemas, docs, env vars `LMTORCH_*`,
cache `~/.cache/lmtorch`) on 2026-10-20 for the move to the public lmctl-ai/lmtorch. The safetensors metadata key
`torchcli_format` is a file-format key and keeps its name so existing checkpoints stay readable.

## State
- One C++17 binary, two front ends. lmapi: the original 12 operations (lmcpp, `--schema` byte-identical to
  `cli.schema.dev.json`, tool name `lmtorch`). DSL: `lmtorch --dsl prog.dslb` runs lmscript `_dsl_` bytecode (v5,
  cpp-dsl-sdk VM pinned in `sdk.lock`) against `schema/lmtorch.dsl.json`: 218 signatures (`docs/dsl-api.md`, decisions in
  dsl-host.md).
- Build: `scripts/build-libtorch.sh` (static CPU-only libtorch 2.9.1 from source) then `scripts/build-native.sh`. Release
  form (owner decision 2026-10-20): libtorch, libstdc++ and libgcc static, glibc shared, so no LGPL relinking materials are
  needed and a release is the binary plus license texts. `LMTORCH_FULL_STATIC=ON` still links glibc statically.
  Docker, the host shim, wheel locks and the docker CI were removed the same day.
- Distribution: GitHub Releases only (no npm). `.github/workflows/ci.yml` builds release and dev forms on ubuntu-22.04
  and ubuntu-22.04-arm (glibc floor 2.35) and runs test.sh; a `v*` tag runs `scripts/package-release.sh` and attaches
  `lmtorch-linux-{x64,arm64}`, `.tar.gz` (binary + LICENSE + THIRD_PARTY_NOTICES.md + licenses/) and `.sha256`.
- License: MIT (`LICENSE`, byte-identical to lmcpp/lmrust/lmcad, sha256 949eb9b7...). No Rust is involved.
- `examples/`: three lmapi recipes plus a `-dsl` version of each, train-report-dsl (its lmapi original was removed: it needed
  the private lmcreator), and five multi-step DSL tasks.
- Tests: `./test.sh` (testing.md): 27 gates on a release build, 26 on a dev build; all green locally on both
  (2026-10-20, aarch64: glibc-only dynamic, needs glibc >= 2.38 when built on Ubuntu 24.04).
- CI evidence: in the private repo, run 39412874928 passed test.sh for the dev form on amd64 and arm64 (first amd64
  confirmation); its static jobs were cancelled to save private-repo minutes. The release form has not run in CI yet.
- Bugs fixed (all with regression tests): see the commit history of 2026-10-20 and testing.md (ragged literals reading
  uninitialized memory, uncaught lmcpp::Error aborting the process, BOOL payload bytes, grad mode leaking across files,
  CSV byte-order marks, and several error messages).

- `_dsl_ cli="lmtorch"` (lmscript R42, lmctl-mono 0795e69f, 2026-10-20): the compiler takes the schema from
  `lmtorch --dsl-schema --json`; lmtorch adds `--cli-schema`. Verified end to end with the mono lmscript (29 gates); the
  published npm lmscript 0.2.347 predates it (28 gates, cli= skipped).

## Pending
- First CI run of the release form (cold libtorch build) and the first `v*` release.
- Switch the examples from `schema=` to `cli="lmtorch"` once an lmscript release contains R42.
- No BLAS (portable GEMM; no linalg); macOS/Windows not attempted.

## Rules that apply
- Commit identity Mike Ma <mike.ma@lmctl.com>, one Co-Authored-By per contributing model, explicit `git add`, push after every commit.
- Transfers and visibility changes are operator-approved actions, never done on an agent's own initiative.
