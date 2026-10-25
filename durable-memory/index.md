# lmtorch durable memory

Navigation for agents. Code and `./test.sh` take precedence over these notes.

- [status.md](status.md): current state, what is verified, what is open, rules that apply. Read first.
- [native-build.md](native-build.md): static CPU-only libtorch from source, the fully static link, the `--dev` wheel build,
  measurements and gotchas.
- [dsl-host.md](dsl-host.md): the `_lmcode_` bytecode host: decisions (one binary/two front ends, handles and release, file
  rules, model metadata, RNG parity, result protocol) and DSL authoring gotchas.
- [testing.md](testing.md): what `test.sh` runs, the PyTorch oracles, and how to extend the API and refusal suites.

Elsewhere in the repository: `docs/dsl-api.md` (DSL API reference), `docs/parity.md` (lmapi operation boundary),
`examples/README.md` (both example families), `DESIGN.md` (the original lmcpp/lmapi design).

Related checkouts on the development host: `../poc/cpp-dsl-sdk` (a checkout of lmctl-ai/lmcode-cpp, the VM and std; durable-memory/index.md there; `LMCODE_CPP=../poc/cpp-dsl-sdk scripts/sync-sdk.sh` re-vendors),
`../poc/lmcad`, `../poc/lmhtml` (sibling DSL hosts this one copies), `../lmctl-mono/lmscript/durable-memory/dsl-framework.md`
(the `_lmcode_` framework and language spec), `../lmcpp` (the lmapi C++ SDK; lmtorch pins revision f6c6f51).
