# torchcli status and hand-off (2026-10-19)

## State
- `main`: C++17 lmcpp front end for libtorch/PyTorch in a pinned offline docker image (hash-locked wheels, arm64 + amd64 locks), 12 ops, lmapi schema via `--schema` / `--schema --json`. Same architecture as pycli/rcli.
- `examples/`: `tensor-chain`, `grad-check`, `mlp-train`, `train-report` (the last needs the private lmcreator checkout and native helpers; local-only). Run as `lmscript examples/x.lms` with the tool shim on PATH (never `lmscript run`).
- CI (`.github/workflows/ci.yml`, `lock-drift.yml`): arm64 + amd64, secrets scan, test image, CTest 2/2 on both arches (run 38927178576), schema gate, recipes, image sizes. The weekly wheel-hash drift workflow has never fired. No amd64 timings. Nothing is published.
- Repo is PRIVATE and has no LICENSE file. Decide licensing/visibility before any go-public (torch is BSD-3; check wheel notices).

## Not done (deliberately)
- The re-engineering requested by the owner for pycli/rcli/torchcli: one binary with no Python/torch runtime dependency, scripted by lmscript. A separate team owns this (hosted on the m4g machine). torchcli has NO native plan yet: pycli/rcli have `durable-memory/native-plan.md` (the `lmnum` multicall design) which does not cover libtorch. Likely start: link libtorch (LibTorch C++ distribution, BSD-3) or a Rust/Candle-style backend directly; decide static vs. bundled-shared-lib tradeoffs (size, CUDA absence, macOS/Windows availability) first.
- The one-binary acceptance checklist: static or self-contained binary, zero external processes, one version, `--schema` and `--schema --json` byte-identical to the current schema, builds on Linux, macOS and Windows. Keep the docker image as the frozen test oracle with explicit numeric tolerance tiers.

## Rules that apply
- Commit identity Mike Ma <mike.ma@lmctl.com>, one Co-Authored-By per contributing model, explicit `git add`, push after every commit. Staging/deployment belongs to another team.
- Private repo: verify locally; gate any paid macOS/Windows CI legs with `github.event.repository.private == false`.
