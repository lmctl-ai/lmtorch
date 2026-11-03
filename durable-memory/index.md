# lmtorch durable memory

Navigation for agents. Code and `./test.sh` take precedence over these notes.

- [status.md](status.md): current state, what is verified, what is open, rules that apply. Read first.
- [native-build.md](native-build.md): static CPU-only libtorch from source, the fully static link, the `--dev` wheel build,
  measurements and gotchas.
- [dsl-host.md](dsl-host.md): the lmcode host, invocation-manifest v1, return frame, CPU tensor handles and file rules.
- [testing.md](testing.md): current native gates, CPU PyTorch oracle and manifest/frame controls.

Elsewhere in the repository: `docs/dsl-api.md` (lmcode host API), `docs/parity.md` (native host boundary),
`examples/README.md` (lmcode examples), `DESIGN.md` (historical product notes).

The maintained SDK pin is recorded by full SHA in `sdk.lock`; its vendored files are checked by `scripts/sync-sdk.sh --verify`.
Historical checkout notes and pre-migration behavior are retained in the dated status/build documents, not used by the build.
