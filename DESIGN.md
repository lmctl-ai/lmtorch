> Historical design record (2026-10). Its docker delivery and wheel locks were replaced by the native build in
> `scripts/build-libtorch.sh` and `scripts/build-native.sh`; see README.md and durable-memory/ for the current state.

# lmcpp and a stateless libtorch native tool

Decision: build `~/repos/lmctl-ai/lmcpp` as a small C++17 SDK, then `lmtorch` here as a CPU-only multi-call binary. Use direct C++ handlers and durable safetensors artifacts; no daemon in v1. This is a design, not an implementation or build report. Commands below are acceptance targets for files to be implemented.

## Contract findings that affect the design

- Read: `gemini.txt`; lmapi `docs/protocol.md` including §6.1; lmrust README and `types`, `inputs`, `publish`, `server/{cli,output}`, `clap_bridge/{mod,generate,child}`; lmts README; physim overlay, published schema and recipes.
- **Envelope correction:** the current SDK wire format is raw result JSON on exit 0, or `{ "success": false, "error": { "code", "message", "data" } }` on failure. `lmscript/clitools-binding.ts:156–166` wraps these into `{status:"ok",result}` / `{status:"error",error}`. Do not print a second status envelope from C++; that would nest `result` and misclassify failures.
- Publish `cli.schema.v1`, `tool:"lmtorch"`, flat snake_case operations, explicit `access`, `returns:"object"`, version, globals and `inputs:["flags","json"]`; omit `product`. Refuse GraphQL with `FORM_NOT_ALLOWED` until implemented.
- `--schema` must need no artifact directory, model, network or handler initialization. Embed schema/overlay data; never read a runtime overlay file. Loading linked DSOs still has a measurable startup cost.
- New tools can use only the lmapi runner. Existing CLIs need `maybe_run_cli`: only lmapi-shaped argv enters it; all other argv retains legacy parsing/output. Native domain statuses stay inside successful results; standardized failures use protocol exits and `error.data.tool_code`.
- Physim's temporary `result_file` adapter reads CSV/JSON/text and deletes the file; its 16 MiB cap is not an artifact transport. Persistent tensor output changes filesystem state and must be `write`, unlike physim's private disposable simulation output.

## lmcpp: smallest useful SDK, with an explicit parity boundary

Use a compiled static `lmcpp::core` target, small public headers and nlohmann ordered JSON. Do **not** make all parsing/runtime code header-only: rebuilding it alongside torch templates wastes this host's RAM/time. CMake >=3.22, C++17, GCC/Clang; pinned dependency versions/hashes, install/export package and `find_package(lmcpp CONFIG)`.

| Component | First release | Later parity work |
|---|---|---|
| Schema | Typed Param/Operation/PublishedSchema, deterministic publish/parse, names/reserved names, globals, defaults, enums, occurs, positional/items and schema flags | Preserve all lmrust optional metadata and warnings |
| Inputs | argv + JSON inline/file, direct params or op/params envelope; one shared validator | GraphQL parser and encoder |
| Validation | Alias/boolean/repeated/positional semantics; defaults, coercion by form, unknown/duplicate/missing/type/enum errors; mixed forms and op mismatch | Full differential edge coverage |
| Serve | `run_cli`, injectable `run_cli_io`, `maybe_run_cli`; schema/list/help/explain; stdout JSON, stderr diagnostics | Generic subprocess/product-version helpers |
| Policy/errors | Readonly/dryrun/confirm plus env floors; standardized exits 0–10/127; tool_code; exception boundary | Client tokenizer/interpreter/prompt formatter, module cache |
| Results | Raw JSON success; SDK error envelope; bounded output; RAII private temp files, JSON/text `result_file` adapter | CSV/auto adapter and all legacy bridge forms |
| Overlay | Embedded TOML semantic overlay, fail unknown keys; required access, published names, returns, exclusions | Full clap_bridge overlay compatibility |

No silent coercion differences: use lmapi fixtures as authority and compare issue code + param/op identity, not wording; still include nonempty message/synopsis/usage in validation issues. Reject unsupported schema features explicitly. Stage 0 is a documented native-server subset, **not full lmrust parity**.

Proposed public surface (sketch; `Result<T>` is a C++17 value/error variant):
```cpp
using Handler = std::function<Result<Json>(const Params&, Context&)>;
CliSpec spec = CliSpec::native("toy", "0.1.0").inputs({Flags, JsonForm});
spec.op(Operation("add").access(Read).returns(Object)
  .param(Param("a", Number).required()).param(Param("b", Number).required()),
  [](const Params& p, Context&) -> Result<Json> {
    return Json{{"value", p.number("a") + p.number("b")}};
  });
return run_cli(spec, argc, argv); // validates, enforces policy, dispatches, prints
```
Also expose `publish_schema`, `parse_published_schema`, `validate`, `resolve_input`, `encode_argv/json`, `CliError`, and `read_result_file`. Context supplies policy and stderr; runner owns output. Apply policy before any handler/file creation; dryrun validates request semantics and reports a plan without invoking the mutating handler. Tool-specific shape checks live outside schema validation.

Add optional `lmcpp::cli11_bridge` after the direct API works. Use CLI11's definition tree (not only parsed subcommands) plus explicit overlay metadata; parser type labels/custom validators cannot reliably supply all schema types/enums. An overlay/typed registration must fill gaps or schema generation fails. Public `Bridge(app, embedded_overlay).schema()/maybe_run()/assert_coverage()` mirrors lmrust. Require every leaf to be mapped or excluded, reject duplicate published names, and snapshot schema ordering. [CLI11 API](https://raw.githubusercontent.com/CLIUtils/CLI11/main/include/CLI/App.hpp)

For lmtorch, use direct handlers; it has no legacy CLI to self-exec. A later legacy bridge may self-exec with argv arrays, a recursion guard, bounded stdout/stderr, timeout/signal handling and exit mapping. Never use shell command strings. Keep `result_file` temporary parsing distinct from persistent artifact references; a size overflow is `COMMAND_FAILED` with `data.tool_code:"OUTPUT_TOO_LARGE"`, matching lmrust.

Suggested layout:
```text
lmcpp/
  CMakeLists.txt; cmake/lmcppConfig.cmake.in
  include/lmcpp/{schema,validate,inputs,server,result_file,cli11_bridge}.hpp
  src/{schema,validate,argv,json,server,policy,result_file,overlay}.cpp
  src/cli11_bridge/; tests/{conformance,differential,native,bridge}/
  examples/toy/{toy.hpp,main.cpp,overlay.toml,toy.lms}
  docs/{parity.md,protocol.md}; third_party/locks/
pytorch/
  CMakeLists.txt; src/{main,tensor_io,tensor_ops,mlp}.cpp
  lmapi/overlay.toml; cli.schema.dev.json; examples/*.lms
  tests/; docker/{Dockerfile,requirements.lock}; bin/lmtorch
```

## Artifact boundary and twelve operations

Choose **safetensors**, one file holding one or several named tensors. Its embedded dtype/shape/offset header avoids two-file consistency problems of raw+JSON; avoid `torch::save` as the public interchange contract. Follow the upstream binary specification: little-endian size prefix, JSON header, contiguous tensor data. Implement a narrow C++ reader/writer initially (F32/F64/I64, dense contiguous CPU only), tested against Python safetensors; reject other encodings. [Format specification](https://github.com/safetensors/safetensors#format)

Handle string: `artifacts/run01/a.safetensors#tensor`; path resolves against working directory, key is explicit, and `#` in paths is disallowed. No pointer IDs, registry or implicit sessions. Return `{ref,path,key,dtype,shape,nbytes}`; multi-tensor results return a name-to-descriptor map. Model tensors use `w1,b1,w2,b2`; safetensors string metadata records `torchcli_format=1`, architecture `mlp_relu_v1` and step. Include seed/version in result metadata.

Every producing op takes required `out`; write a sibling temporary, flush and atomically install without replacing an existing destination (CONFLICT/5). Never update input files; training emits a new checkpoint. Caller owns retention and run-directory cleanup. Validate header size (<=1 MiB), ranks/dimensions, checked byte arithmetic, offsets/overlap/full coverage, duplicate keys and exact file length before allocation. Default artifact cap 256 MiB and bounded preview; fail before allocation where possible. No preserved strides, aliasing, device or autograd graph. Start with copies, not mmap lifetime complexity.

| Published op | Parameters and behavior |
|---|---|
| `tensor_create` | shape, dtype=F32, fill or bounded values, out; create tensor (mutually exclusive fill/values) |
| `tensor_rand` | shape, seed, distribution=uniform, dtype, out; seeded local generator |
| `tensor_matmul` | a, b, out; initially rank-2 compatible matrices |
| `tensor_add` | a, b, out; initially equal shape/dtype, no implicit broadcasting |
| `tensor_softmax` | input, dim, out; stable softmax, checked dimension |
| `tensor_linear` | input, weight, optional bias, out; X Wᵀ + b |
| `tensor_save` | tensors map of name -> ref, out; pack tensors into a portable file |
| `tensor_load` | path, optional key, out; validate/import selected tensors into a new canonical file |
| `tensor_inspect` | input, limit<=32; metadata + bounded values/nonfinite counts; read-only |
| `autograd_grad` | input, function=`sum_square`, out; rebuild graph and return dx=2x plus loss |
| `mlp_forward` | checkpoint, input, out; fixed two-layer ReLU MLP, no-grad inference |
| `mlp_train_step` | checkpoint, input, target, lr, out; MSE, backward, plain SGD, new weights + loss |

All except inspect are `write` because they persist files. Shapes/dtypes remain domain validation, with standard code plus precise tool_code. Map NaN/Inf previews to null plus explicit counts; never emit invalid JSON. Creation accepts only finite JSON values. Seed reproducibility is within the pinned image/thread configuration, not promised across PyTorch versions.

Autograd is deliberately scoped: saving tensors detaches their history. `autograd_grad` cannot differentiate an earlier CLI matmul; compound differentiation must reconstruct that computation in one invocation. MLP train_step reconstructs forward/loss/backward, uses no momentum/dropout, and persists only weights and step metadata. Initialize its four weights with create/rand then tensor_save. This proves training without serializing a graph or arbitrary executable models.

No daemon in v1: measure process/import and artifact I/O separately from kernel time. Small tensor calls may be dominated by startup and copies; gemini.txt's assumption that compute always dominates is unsafe. Only add a session/UDS service if measured repeated workloads justify its lifecycle, replay and concurrency costs. Keep per-call fixtures below lmscript's current 60-second default tool timeout.

## Docker: native linux/arm64 CPU, no PyTorch source build

Verified availability: the official CPU index lists `torch-2.9.1+cpu-cp311-cp311-manylinux_2_28_aarch64.whl`. Pin this known available release for the spike, not an unverified “latest”. The inspected libtorch CPU zip index has no `aarch64` entry; macOS arm64 zips are unusable here. [CPU wheels](https://download.pytorch.org/whl/cpu/torch/), [libtorch zip index](https://download.pytorch.org/libtorch/cpu/)

PyTorch documents CMake consumption of a pip installation through `torch.utils.cmake_prefix_path`; v2.9.1 packaging includes headers, shared libraries and Torch/Caffe2 CMake files. This establishes the supported route, but the exact image still needs the stage-1 compile/link probe. [C++ installation](https://docs.pytorch.org/cppdocs/installing.html), [v2.9.1 packaging](https://raw.githubusercontent.com/pytorch/pytorch/v2.9.1/setup.py)

Use `python:3.11-slim-bookworm` for builder and initial runtime, with the arm64 manifest digest pinned at implementation time. Debian glibc exceeds the wheel's 2.28 floor. Builder installs g++, cmake, ninja-build and pkg-config; install only binary wheels, initially:
```sh
python -m pip install --no-cache-dir --only-binary=:all: 'torch==2.9.1+cpu' --index-url https://download.pytorch.org/whl/cpu
cmake -S /src/pytorch -B /build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')" \
  -Dlmcpp_DIR=/opt/lmcpp/lib/cmake/lmcpp
cmake --build /build --parallel 1
```
Turn the resolved wheels (including dependencies) into a hash-locked requirements file before release. `find_package(Torch REQUIRED)` supplies includes/libraries; apply `TORCH_CXX_FLAGS` to every ABI-sharing target, never guess `_GLIBCXX_USE_CXX11_ABI`. Link `${TORCH_LIBRARIES}`, not torch_python. Use narrow headers, a few translation units, no LTO/debug symbols; begin at -j1, increase to -j2 only after measuring peak RSS.

Multi-stage image copies the installed wheel/dependencies and binary at identical paths, retains required system DSOs, and excludes compiler/build tree/headers when safe. Keeping Python in the initial runtime simplifies packaging; **invocations execute the C++ binary directly**, without Python. Configure loader search for torch/lib (including bundled OpenMP dependencies); `ldd` and a native matmul probe must show no missing libraries. Delay aggressive DSO trimming until this works.

Budget estimates, not measurements: runtime 0.6–1.5 GB unpacked, builder 2–4 GB, peak working/cache disk <10 GB. Cold dependency download/install + build about 10–30 minutes on 2 cores; wrapper compilation about 2–10 minutes, cached incremental 10–90 seconds. Torch-heavy translation units may take longer; serial build targets <5 GB RSS and leaves host headroom. Record actual image bytes, wall time and RSS before accepting these budgets.

Run with two CPU threads/intra-op and one inter-op thread; pass `OMP_NUM_THREADS=2`. A host `bin/lmtorch` shim execs Docker with argument forwarding (`"$@"`), no tty, same absolute workspace mount and cwd, host uid/gid, `--network=none`, `--cpus=2 --memory=5g`. Send Docker failures to a protocol error mapper. Artifact refs remain valid across CLI and recipe calls. Put lmscript TMPDIR under that mounted workspace so generated `--input-file` paths are visible inside the container; never rewrite JSON path strings ad hoc.

## Stages and acceptance gates

**Stage 0 — lmcpp + tiny C++ library, no torch.** Implement toy.add and toy.scale as pure read handlers, plus a write fixture for policy tests. Generate schema from the same spec. Ship `examples/toy/toy.lms` importing `./cli.schema.dev.json`: call add(2,3), then scale(value,4); explicitly exit 1 on error/wrong result and print 20. Its dry pass must tolerate `status:"simulated"` and use deterministic placeholder values instead of reading missing results.
```text
import "./cli.schema.dev.json" as toy;
a = toy.add({a:2, b:3});
if (a.status == "error") { println(a.error.message); exit(1); }
v = 5;
if (a.status == "ok") { v = a.result.value; }
b = toy.scale({value:v, factor:4});
if (b.status == "error") { println(b.error.message); exit(1); }
if (b.status == "ok") {
  if (b.result.value != 20) { exit(1); }
  println(b.result.value);
}
```
```sh
cd ~/repos/lmctl-ai/lmcpp
cmake -S . -B build -G Ninja -DLMAPI_DIR=../lmapi -DLMCPP_BUILD_TESTS=ON
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure
build/bin/toy --schema > examples/toy/cli.schema.dev.json
build/bin/toy add --a 2 --b 3
build/bin/toy add --input-json '{"a":2,"b":3}'
PATH="$PWD/build/bin:$PATH" lmscript --check examples/toy/toy.lms
PATH="$PWD/build/bin:$PATH" lmscript --dry-run examples/toy/toy.lms
PATH="$PWD/build/bin:$PATH" lmscript examples/toy/toy.lms
```
Gate: both CLI forms return `{"value":5}`, recipe returns 20; schema succeeds offline/from unrelated cwd. CTest tests malformed JSON/unknown/mixed/duplicate inputs, readonly=6 and dryrun=no writes, error envelopes/exit mapping, native guard behavior and bounded result_file cleanup. Validate schema against lmapi's JSON Schema. Pin fixture revisions; run all supported conformance groups without silent skips and report deferred groups. Differential harness uses existing lmrust/lmts oracles for canonical values/issues; later full-parity milestone requires every vector, GraphQL/client APIs included.

**Stage 1a — wheel/link probe, then tensor arithmetic.** Dockerfile uses the repositories root as context to copy both lmcpp and this project (with a strict allowlist `docker/Dockerfile.dockerignore`). Add CTest integration targets for actual wheel headers/CMake discovery, no missing DSOs, no CUDA, dtype roundtrips and malformed artifacts. Add safetensors and Python oracle scripts only to the test target; use its installed torch for numerical reference checks. No source-build fallback.
```sh
cd ~/repos/cppwrapper/pytorch
docker build --platform linux/arm64 -f docker/Dockerfile --target test -t lmtorch:test ~/repos
docker run --rm --network=none lmtorch:test ctest --test-dir /build --output-on-failure
docker build --platform linux/arm64 -f docker/Dockerfile --target runtime -t lmtorch:dev ~/repos
mkdir -p artifacts/run01 .tmp
export PATH="$PWD/bin:$PATH" TMPDIR="$PWD/.tmp"
lmtorch --schema > cli.schema.dev.json
lmtorch tensor_create --shape 2 --shape 2 --fill 1 --out artifacts/run01/a.safetensors
lmtorch tensor_matmul --a 'artifacts/run01/a.safetensors#tensor' --b 'artifacts/run01/a.safetensors#tensor' --out artifacts/run01/b.safetensors
lmtorch tensor_inspect --input 'artifacts/run01/b.safetensors#tensor'
```
Gate: inspect reports shape [2,2], values [2,2,2,2]; separate processes and restarted containers produce identical results. Declare shape as repeated numbers (`occurs:'+'`), then enforce integral nonnegative dimensions in the handler. Existing output yields CONFLICT and no changes. Record startup, I/O and matmul timings; fresh-schema discovery must not create artifacts.

**Stage 1b — remaining operations and lmscript training.** Add recipes with distinct output paths per iteration and explicit simulated-result handling. Arithmetic recipe reproduces the preceding chain; gradient checks sum_square against 2x and finite differences; training recipe initializes a fixed 2→4→1 MLP, performs 20 small SGD steps and verifies fixture-specific loss reduction, then forwards from its final saved checkpoint.
```sh
lmscript --check examples/tensor-chain.lms
lmscript --dry-run examples/tensor-chain.lms
lmscript examples/tensor-chain.lms
lmscript --check examples/mlp-train.lms
lmscript examples/mlp-train.lms
```
Gate: numerical tests compare all twelve ops with the pinned Python torch oracle (float tolerances, not byte equality), checkpoint reload reproduces forward output, split/resumed SGD equals uninterrupted steps, and readonly/dryrun leave artifact tree unchanged. Each call stays below 60 s. Reruns use fresh run directories; recipes accept a run-directory argument.

**Stage 2 — SDK breadth only after libtorch proof.** CLI11 bridge/overlay coverage, CSV result_file and client/GraphQL/module features have their own parity gates; do not delay the first library proof for them. Test SDK on GCC/Clang x86_64 CI and native aarch64, use ASan/UBSan for SDK/artifact parser, and fuzz offsets/shape overflow/JSON limits. Tests must fail if required conformance fixtures are absent.

Main risks: ABI/loader mismatch (pin/probe first), parser bugs (strict bounds + cross-language fixtures), host/container paths (same absolute mount), policy misclassification (all persistent output is write), lost autograd history (explicit compound ops), and startup/I/O overhead (measure). Cut daemon, GPU, arbitrary graph/model loading, optimizer momentum, mmap, broad dtype/operator coverage and full legacy bridge support from v1. If compile memory is tight, reduce translation units/includes and parallelism; never build PyTorch on this host.
