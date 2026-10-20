# The `_dsl_` host (lmtorch --dsl)

What it is: lmtorch embeds the cpp-dsl-sdk VM (`third_party/dslvm/dslvm.hpp`, pinned by `sdk.lock`, same header hash as
lmcad/lmhtml) and binds `schema/lmtorch.dsl.json` to libtorch in `src/dsl_host.cpp`. lmscript compiles `_dsl_` blocks
against that schema (`savedsl`); `lmtorch --dsl A.dslb [B.dslb ...]` runs them. API reference: `docs/dsl-api.md`.
Pattern copied from ../poc/lmcad and ../poc/lmhtml (embedded schema bytes, `--dsl-signatures` parity, one JSON line).

## Decisions (2026-10-20)

- **Host flag standard (lmscript R42, 2026-10-20):** `--dsl-schema [--json]` (DSL schema), `--cli-schema [--json]` (lmapi
  document; `--schema` stays as the lmapi protocol name), `--dsl-signatures`. lmscript's `_dsl_ cli="lmtorch"` runs
  `lmtorch --dsl-schema --json` at compile time, so scripts need no schema file; examples keep `schema=` until an lmscript
  release ships R42 (test.sh runs a cli= smoke test whenever the compiler supports it).
- **One binary, two front ends.** `main.cpp` sends `--dsl`, `--dsl-schema`, `--dsl-signatures` to the DSL host; everything
  else is the unchanged lmcpp/lmapi CLI (`--schema` stays byte-identical to `cli.schema.dev.json`, the acceptance gate
  from status.md). The DSL schema flag is `--dsl-schema` because `--schema` already belongs to lmapi.
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
  lmcpp::Error, bad_alloc, std::exception and unknown exceptions into located `RUN_HOST` errors.
  Bug found by tests/dsl/negative.lms: lmcpp::Error is not a std::exception, and an uncaught one aborted the process
  (read_csv outside cwd). The wrapper catches it now.
- **Files stay inside the working directory** (`lmtorch::check_contained`, the lmapi rule) and DSL saves replace
  atomically (rename), unlike the lmapi operations' exclusive link() writes. Re-runnable scripts need replacement.
- **Models are `torch::nn::Sequential`.** State dict names equal PyTorch's (`0.weight`). `Model.save` stores
  `torchcli_format=model-1` and `layers=<summary>`; `torch.load_model` replays that spec through the same `add_layer`
  used by the builder methods, so the summary text is a format: change it only together with `replay_layers`.
- **RNG is the default CPU generator**: `torch.manual_seed(s); torch.rand(...)` equals Python, and equals the lmapi
  `tensor_rand` (which seeds a fresh generator per call), so ported recipes reproduce their old numbers exactly.
- **Result protocol:** exactly one stdout line for the whole invocation (lmcad prints one per file; lmtorch does not), the
  `out` object accumulated across files. All files are loaded and verified before any runs (lmhtml rule).
- **Per-file state:** grad mode is reset to enabled before each file of a multi-file run; the RNG, thread count and `out`
  object carry over (a later file continues the random stream).
- **Literals are strict:** all scalars of `torch.tensor(json)` must share the final depth and fill the inferred shape;
  overflowing numbers are refused by the JSON parser; nonfinite values are refused.
- **Threads:** the DSL path keeps libtorch's default intra-op thread count (`torch.set_num_threads` to change it); the lmapi
  path keeps its fixed 2 threads.

## Gotchas

- `_dsl_` closers (`end _dsl_`) must be at column 1 even inside an lmscript `while`/`for`; body lines may be indented.
- A statement cannot start with `(`: write `let s = (w * 1.0).sum(); s.backward();`.
- `${n}` of an integral lmscript number is a DSL int: pass rates as scaled ints (`float(${round(lr * 1e6)}) / 1000000.0`).
- lmscript property names cannot be keywords (`res.fn` is a parse error): result keys like `fn` are unusable.
- `grad()` returns a copy; `zero_grad()` sets the gradient to none (libtorch 2.x default), so `has_grad()` is false after it.
- `Tensor.set` writes in place through views (a `slice` shares storage), and refuses non-contiguous tensors.
- A DSL rebuild of `src/dsl_host.cpp` takes several minutes on the shared 4-core host (torch/torch.h); other teams' builds
  often load the machine to 10-40.
