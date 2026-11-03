# lmcode host (lmtorch --lmcode)

## Current boundary (WP175, 2026-11-02)

`lmtorch` exposes one native host surface: `--lmcode`, `--lmcode-schema`, and `--lmcode-signatures`. The C++ host vendors
lmcode-cpp at the full reviewed SHA in `sdk.lock`, consumes invocation manifest v1 with `InvocationResults`, propagates
the manifest dry flag to its VM context, and emits the terminal `LMCODE_RETURN` frame. Its public filesystem work
remains contained to the invocation directory. CMake, public headers, build scripts and CI no longer depend on the former
command SDK. See `docs/parity.md` and `durable-memory/testing.md` for the active contract.

## Historical implementation notes (pre-WP175)

What it is: lmtorch embeds the lmcode-cpp VM (`third_party/dslvm/dslvm.hpp` plus `stdlib/*.hpp`,
the std schema and `scripts/merge-std.py`, all pinned per file by `sdk.lock`) and binds the host-only
`schema/lmtorch.dsl.json` to libtorch in `src/dsl_host.cpp`. lmscript compiles `_lmcode_` blocks
against that schema; `run_lmcode` compiles and invokes the host with manifest v1. The host validates every submitted bytecode file before running effects, collects each result slot, and emits one terminal `LMCODE_RETURN` frame. API reference: `docs/dsl-api.md`.

## Retained implementation decisions (originally recorded 2026-10-20; revised by WP175)

- **lmcode std (2026-10-24):** the default `dslvm::Registry` installs the 72-signature lmcode std (strings, math,
  collections, files). lmtorch registers none of it and dropped its own `StrList` (`Bundle.keys()` returns the std one via
  `registry.std_collections().make_str_list`, valid for the run). `schema/lmtorch.host.json` is the source for
  `schema/lmtorch.dsl.json`; keep the two byte-identical. The embedded schema and bytecode hash now contain only the host
  declarations. lmscript 0.2.420 or later and the default runtime registry provide std automatically. `_lmcode_` is the
  current spelling; `_dsl_` remains accepted as an alias.

- **Current compiler integration (WP175):** `_lmcode_ cli="lmtorch"` discovers the embedded schema, and `run_lmcode` compiles and invokes a program with its manifest. Portable examples may still set `schema=` explicitly when running from a scratch directory.
- **One host surface (WP175):** `main.cpp` accepts `--version`, `--lmcode`, `--lmcode-schema`, and `--lmcode-signatures`. The former lmcpp/lmapi CLI and operations were removed; invalid legacy flags return a framed CLI error.
- **Context:** `torch` (Torch) for everything tensor-shaped and `out` (Out) for the result object. No free functions:
  reads like PyTorch (`torch.zeros`, `torch.manual_seed`).
- **Schema 2 operators** map `+ - * neg r+ r- r*` to Tensor methods; `/` has no schema-2 operator key, so `div()`.
- **Shapes are 1-4 int overloads** (`zeros(2, 3)`): the DSL has no lists. Literal data enters as JSON text
  (`torch.tensor("[[...]]")`), usually via `${json_stringify(list)}` from lmscript.
- **dtype is an enum** (`DType.F32|F64|I64|BOOL`); enum values arrive as strings, so the argument checker accepts STR for
  `DType` and validates membership. BOOL was added to the safetensors reader/writer (1 byte) for comparison results.
- **Handles never die during a run** (deques keep addresses stable). `torch.mark()/release(mark)` empty unkept handles so a
  long loop keeps memory flat; using a released handle is a clear error, not a dangling read. Holders themselves are never
  freed (about 30 bytes each). Model params, optimizer state, bundles and lists hold their own references.
  Measured (dev build, 2026-10-20): 20000 Adam steps of an 8-32-1 MLP on 64 rows keep 40002 live tensors and 255 MB max
  RSS without release, 2 live tensors and 126 MB with `mark()`/`release()` per step; the same 5 s either way.
- **Every registry entry re-checks argument kinds** (the VM does not) and converts c10::Error (first line, max 300 chars),
  lmtorch::Error, bad_alloc, std::exception and unknown exceptions into located `RUN_HOST` errors.
  The safetensors error type is owned by lmtorch and translated at the host boundary.
- **Files stay inside the working directory** (`lmtorch::check_contained`) and lmcode saves replace atomically (rename), allowing re-runnable scripts.
  Windows (`#ifdef _WIN32`, `lmtorch::win_atomic_write`): CREATE_NEW temp file next to the target, FlushFileBuffers,
  MoveFileExW with REPLACE_EXISTING|WRITE_THROUGH; no mode bits or directory fsync. Containment compares components case-insensitively and refuses ADS, device
  names and trailing dots/spaces. Paths are in the ANSI code page there (argv and std::filesystem::path(std::string)).
- **Models are `torch::nn::Sequential`.** State dict names equal PyTorch's (`0.weight`). `Model.save` stores
  `torchcli_format=model-1` and `layers=<summary>`; `torch.load_model` replays that spec through the same `add_layer`
  used by the builder methods, so the summary text is a format: change it only together with `replay_layers`.
- **RNG is the default CPU generator**: `torch.manual_seed(s); torch.rand(...)` equals Python, and preserves the prior CPU recipe values.
- **Result protocol:** one terminal return frame for the invocation, with a result per builder slot and each program's `out` object. All files are loaded and verified before any runs.
- **Per-file state:** grad mode is reset to enabled before each file of a multi-file run; the RNG, thread count and `out`
  object carry over (a later file continues the random stream).
- **Literals are strict:** all scalars of `torch.tensor(json)` must share the final depth and fill the inferred shape;
  overflowing numbers are refused by the JSON parser; nonfinite values are refused.
- **Threads:** the lmcode path keeps libtorch's default intra-op thread count (`torch.set_num_threads` to change it).

## Gotchas

- `_lmcode_` closers (`end _lmcode_`) must be at column 1 even inside an lmscript `while`/`for`; body lines may be indented.
- A statement cannot start with `(`: write `let s = (w * 1.0).sum(); s.backward();`.
- `${n}` of an integral lmscript number is a DSL int: pass rates as scaled ints (`float(${round(lr * 1e6)}) / 1000000.0`).
- lmscript property names cannot be keywords (`res.fn` is a parse error): result keys like `fn` are unusable.
- `grad()` returns a copy; `zero_grad()` sets the gradient to none (libtorch 2.x default), so `has_grad()` is false after it.
- `Tensor.set` writes in place through views (a `slice` shares storage), and refuses non-contiguous tensors.
- A DSL rebuild of `src/dsl_host.cpp` takes several minutes on the shared 4-core host (torch/torch.h); other teams' builds
  often load the machine to 10-40.

## Hard host flag cutover

WP175 removed the lmcpp/lmapi implementation and retained only the lmcode host protocol. The vendored SDK revision and file hashes are recorded in `sdk.lock`; `scripts/sync-sdk.sh --verify` checks them. `test.sh` validates manifest v1, dry and real effects, result frames, schema discovery, examples, and CPU numerical oracles when available. Local native qualification is unavailable without Torch and nlohmann headers; the CI matrix remains the native build gate.
