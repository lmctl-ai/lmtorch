# lmtorch status and hand-off (2026-10-21)

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
- Build: `scripts/build-libtorch.sh` (static CPU-only libtorch 2.9.1 from source) then `scripts/build-native.sh`, on
  Linux (GCC), macOS (AppleClang) and Windows (MSVC from Git Bash). Release form: libtorch and the C++ runtime static,
  only OS libraries dynamic: Linux glibc (owner decision 2026-10-20: no LGPL relinking duty; `LMTORCH_FULL_STATIC=ON`
  still links glibc statically), macOS libc++ + libSystem (deployment target 13.0), Windows KERNEL32 + dbghelp with the
  static MSVC runtime (/MT, no VC++ redistributable). Per-OS details and gotchas (Accelerate found despite USE_BLAS=OFF,
  mimalloc forced on Windows and faulting before main): native-build.md. lmcpp pin moved f6c6f51 -> 93a1847 (native
  Windows support, same API) on 2026-10-21.
- Distribution: GitHub Releases only (no npm). `.github/workflows/ci.yml` builds the release form on linux-x64/arm64
  (ubuntu-22.04, glibc floor 2.35), macos-arm64 (macos-15), macos-x64 (macos-15-intel) and windows-x64 (windows-2022),
  plus the dev form on Linux, and runs test.sh on each; `workflow_dispatch` with `only=<platform>` runs one platform; a
  `[skip ci]` commit plus a dispatch is the cheap way to iterate. Docs-only pushes skip CI. A `v*` tag attaches
  `lmtorch-<os>-<arch>[.exe]`, the archive (`.tar.gz`; `.zip` on Windows: binary + LICENSE + THIRD_PARTY_NOTICES.md +
  licenses/) and `.sha256` for all five platforms. The libtorch cache key is runner + libtorch.lock + build-libtorch.sh:
  any edit to that script means about 40 min (linux-arm64, macos-arm64) to 2 h (windows-x64, macos-x64) of cold builds.
- License: MIT (`LICENSE`, byte-identical to lmcpp/lmrust/lmcad, sha256 949eb9b7...). No Rust is involved.
- `examples/`: three lmapi recipes plus a `-dsl` version of each, train-report-dsl (its lmapi original was removed: it needed
  the private lmcreator), and five multi-step DSL tasks.
- Tests: `./test.sh` (testing.md): 31 gates on a release build (29 on macos-x64, where PyTorch publishes no oracle wheel),
  30 on a dev build. CI run 39545682710 (2026-10-21) passed all seven jobs; mlp-train stays at 6 digits on every OS.
- CI evidence (public repo): runs 39418714985 and 39423349623 passed all four jobs (release and dev, x64 and arm64), with
  both PyTorch oracle suites running; the libtorch caches for both runners are saved, so a release run takes minutes.
- Released: v0.4.0 (2026-10-21, tag run 39559055428): 15 assets, `lmtorch-{linux-x64,linux-arm64,macos-arm64,macos-x64}`
  + `.tar.gz` + `.sha256` and `lmtorch-windows-x64.exe` + `.zip` + `.sha256` (stripped sizes 147/123/105/144 MB, Windows
  105 MB). Smoke-tested on aarch64: checksums OK, `--dsl-schema --json` byte-identical, version string
  `lmtorch 0.4.0 (libtorch 2.9.1, CPU)`, checkpoint-resume-dsl (lmscript 0.2.359, release binary first on PATH, nothing
  downloaded) prints `checkpoints bit-identical: true`. macOS and Windows assets are CI-tested only (no local hardware).
- Earlier: v0.3.0 (2026-10-20, run 39447192609): `lmtorch-linux-{x64,arm64}` (147 MB / 123 MB stripped), `.tar.gz`,
  `.sha256`. Smoke-tested on aarch64: checksum OK, `--dsl-schema --json` byte-identical, checkpoint-resume-dsl correct.
- Bugs fixed (all with regression tests): see the commit history of 2026-10-20 and testing.md (ragged literals reading
  uninitialized memory, uncaught lmcpp::Error aborting the process, BOOL payload bytes, grad mode leaking across files,
  CSV byte-order marks, and several error messages).

- `_dsl_ cli="lmtorch"` (lmscript R42, lmctl-mono 0795e69f, 2026-10-20): the compiler takes the schema from
  `lmtorch --dsl-schema --json`; lmtorch adds `--cli-schema`. lmscript 0.2.359 (npm) ships R42 + R43.
- Examples (2026-10-20): every `_dsl_` block is `cli="lmtorch" install="github:lmctl-ai/lmtorch@latest"` (R45, owner 2026-10-21); the helpers
  run `system ${dsl_cli(prog)} --dsl ...` (lib/torchdsl.lms: `run(prog, files)`, `run_block(prog, path)`; LMTORCH_BIN no
  longer selects the binary for examples, PATH does). New install-and-run-dsl.lms is the minimal form. test.sh puts the
  binary under test first on PATH, gives lmscript a private LMSCRIPT_TOOL_CACHE and fails if any example downloaded.
  lmapi recipes (and mlp-train-dsl's cross-check) import `github:lmctl-ai/lmtorch@latest lmtorch` (lmscript R44,
  >= 0.2.366): schema from `lmtorch --cli-schema --json`, calls run that binary; the test.sh no-download guard runs
  after them. `@latest` needs `lmtorch --version` (added in 0.4.1; v0.4.0 and older lack it, so @latest works from v0.4.1 on). No tag bumps are needed in the examples or docs at a release.

## Pending
- No BLAS (portable GEMM; no linalg) on any OS. Release binaries are not code-signed or notarized.

## Rules that apply
- Commit identity Mike Ma <mike.ma@lmctl.com>, one Co-Authored-By per contributing model, explicit `git add`, push after every commit.
- Transfers and visibility changes are operator-approved actions, never done on an agent's own initiative.
- Owner standing permission (2026-10-20): push, tag and publish GitHub Releases without asking.
