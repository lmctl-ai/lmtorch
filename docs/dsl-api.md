# lmtorch `_dsl_` API

`schema/lmtorch.dsl.json` is the contract (dsl.schema 2, bytecode v5). `lmtorch --dsl-schema` prints this list grouped by
type; `lmtorch --dsl-schema --json` prints the exact schema bytes the compiler needs, so a block can name the binary
instead of a schema file: `prog = _dsl_ cli="lmtorch" install="github:lmctl-ai/lmtorch@v0.3.0";` (lmscript >= 0.2.359)
compiles against the binary's own schema, which can never drift from the binary that runs it. `install=` downloads that
release (verified against GitHub's SHA-256 digest) when lmtorch is not on PATH, and `dsl_cli(prog)` is the argv prefix of
the binary the block was compiled against. `lmtorch --cli-schema` prints the
lmapi document (the same as lmapi's `--schema`); `lmtorch --dsl-signatures` prints
the canonical signatures, which `test.sh` diffs against `lmscript --dsl-signatures schema/lmtorch.dsl.json`.

A program is an lmscript `_dsl_` block compiled to bytecode and run by the native host:

```lmscript
prog = _dsl_ cli="lmtorch" install="github:lmctl-ai/lmtorch@v0.3.0";
let x = torch.tensor("[[1.0, 2.0], [3.0, 4.0]]").requires_grad(true);
let loss = (x.matmul(x) + 1.0).square().mean();
loss.backward();
out.set("loss", loss.item()).set("grad", x.grad());
end _dsl_
savedsl(prog, "prog.dslb");
r = system ${dsl_cli(prog)} --dsl prog.dslb;   # stdout: {"ok": true, "result": {"loss": ..., "grad": [[...], [...]]}}
```

The language is the lmscript DSL (typed `let`, `if`/`else`, `switch`, `while`, `for (i in range(a, b))`, `for (x in list)`,
compiler-inlined `fn`, `int`/`float`/`string`/`bool`, `str()`, `float()`, `int()`, `abs`/`min`/`max`/`sqrt`/... on
primitives). No lists or maps exist in the DSL itself: tensors and the host collections below carry data.
`${expr}` in the block substitutes an lmscript value as a literal before compiling (strings, ints, finite floats,
bools); pass lists as JSON text (`torch.tensor(${json_stringify(rows)})`) and a float that may be integral as scaled
int (`float(${round(lr * 1000000)}) / 1000000.0`).

## Context

| Name | Type | Role |
|---|---|---|
| `torch` | `Torch` | tensor creation, files, models, optimizers, process settings |
| `out` | `Out` | the JSON result object, log lines, application failures |

## Torch

| Method | Result | Notes |
|---|---|---|
| `zeros(d1[, d2[, d3[, d4]]])`, `ones(...)`, `rand(...)`, `randn(...)` | Tensor | F32; shapes have 1 to 4 dimensions (reshape for more); dims >= 0, at most 2^31 elements |
| `full(d1..d4, float)` | Tensor | F32 filled with the value |
| `randint(low, high, d1[, d2])` | Tensor | I64 in [low, high) |
| `randperm(n)` | Tensor | I64 permutation of 0..n-1 |
| `arange(int end)` / `arange(float start, float end, float step)` | Tensor | I64 / F32 |
| `linspace(float a, float b, int steps)`, `eye(n)`, `scalar(float)` | Tensor | F32; `scalar` is zero-dimensional |
| `tensor(string json[, DType])` | Tensor | nested JSON list or scalar; shape inferred; dtype inferred like PyTorch (any float → F32, all integers → I64, booleans → BOOL) |
| `where(BOOL cond, a, b)` | Tensor | elementwise select, broadcasting |
| `list()` | TensorList | an ordered list of tensor handles |
| `cat(list, dim)`, `stack(list, dim)` | Tensor | non-empty list |
| `manual_seed(int)` | void | seeds the default CPU generator: the same draws as Python `torch.manual_seed` |
| `set_num_threads(1..256)`, `grad_enabled(bool)`, `is_grad_enabled()` | | intra-op threads; autograd recording on/off (`torch.no_grad()`) |
| `load(path[, key])`, `save(path, Tensor)` | Tensor / void | safetensors; the single-tensor key is `tensor` (the lmapi format: `path#tensor`) |
| `bundle()`, `open(path)` | Bundle | named tensors plus string metadata in one safetensors file |
| `read_csv(path, bool header)` | Tensor | F32 `[rows, cols]`; every cell a finite number; blank lines skipped |
| `sequential()` | Model | an empty `nn.Sequential` |
| `load_model(path)` | Model | rebuilds a model saved by `Model.save` (architecture from metadata) and loads its weights |
| `sgd(Model, lr[, momentum])`, `sgd(TensorList, lr)` | Optimizer | `torch.optim.SGD` |
| `adam(Model, lr)`, `adam(TensorList, lr)`, `adamw(Model, lr, weight_decay)` | Optimizer | `torch.optim.Adam` / `AdamW` defaults otherwise |
| `mark()`, `release(mark)`, `live_tensors()` | int | handle lifetime, see below |
| `version()` | string | `lmtorch 0.3.0 (libtorch 2.9.1, CPU)` |

## Tensor

Operators: `a + b`, `a - b`, `a * b`, `-a` with Tensor or float operands on either side (`2.0 * t`, `1.0 - t`); there is
no `/` operator mapping in schema 2, so division is `a.div(b)`. Arithmetic broadcasts and promotes like PyTorch.

| Group | Methods |
|---|---|
| information | `numel()`, `dim()`, `size(dim)`, `shape()` (`"[2, 3]"`), `dtype()` (DType), `item()`, `item_int()`, `value(i)`, `value_int(i)` (flat row-major index), `to_text()` (6 digits), `to_json()` (round-trip digits; nonfinite → `null`) |
| arithmetic | `add`, `sub`, `mul`, `div` (Tensor or float), `rsub(float)` (= float − t), `pow(float)`, `neg`, `matmul`, `maximum`, `minimum`, `clamp(lo, hi)` |
| elementwise | `abs`, `exp`, `log`, `sqrt`, `square`, `sin`, `cos`, `sign`, `floor`, `round` (half to even), `relu`, `sigmoid`, `tanh`, `gelu`, `leaky_relu(slope)`, `softmax(dim)`, `log_softmax(dim)` |
| comparison | `gt`, `lt`, `eq` (Tensor or float), `ge(float)`, `le(float)` → BOOL; `allclose(t, rtol, atol)` (false when shapes differ), `equal(t)` (same dtype, shape and values), `is_finite()`, `any()`, `all()` |
| reduction | `sum()`, `sum(dim[, keepdim])`, `mean()`, `mean(dim[, keepdim])`, `std()`, `std(dim, keepdim)` (unbiased), `var()`, `norm()`, `max()`, `min()`, `amax(dim)`, `amin(dim)`, `argmax([dim])`, `argmin(dim)` |
| shape | `reshape(d1..d4)` (one `-1` allowed), `flatten()`, `t()`, `transpose(a, b)`, `unsqueeze(dim)`, `squeeze([dim])`, `slice(dim, start, end)` (clamped, step 1), `select(dim, index)`, `index_select(dim, I64 indices)` |
| dtype | `to(DType)`; `DType.F32`, `F64`, `I64`, `BOOL` |
| nn functions | `linear(weight[, bias])` (x Wᵀ + b), `mse_loss`, `l1_loss`, `smooth_l1_loss`, `bce_with_logits` (target shape must equal input shape), `cross_entropy(target)` (class indices or probabilities), `nll_loss(target)` |
| autograd | `requires_grad()`, `requires_grad(bool)` (float leaves only), `backward()` (one-element result), `grad()` (a copy of the accumulated gradient), `has_grad()`, `zero_grad()` (sets the gradient to none), `detach()`, `clone()` |
| mutation | `set(i, float)`: writes one element in place (contiguous tensors only; a view writes through to its base) |
| files | `save(path)`, `write_csv(path)` (1-D: one column; 2-D: rows) |
| lifetime | `keep()` |

Dimensions may be negative (from the end). Tensors are CPU-only.

## TensorList, StrList, Bundle

`TensorList`: `add(t)` (chains), `len()`, `at(i)`; `for (t in list) { ... }` iterates it. `StrList`: `len()`, `at(i)`.
`Bundle`: `put(name, t)`, `get(name)`, `has(name)`, `keys()` (sorted StrList), `meta(key)`, `meta(key, value)`,
`has_meta(key)`, `save(path)`. A bundle stores handles; `save` writes the values current at that moment.

## Model and Optimizer

`Model` layers chain: `linear(in, out)`, `conv2d(in, out, k)`, `conv2d(in, out, k, stride, padding)`, `max_pool2d(k)`,
`flatten()`, `relu()`, `tanh()`, `sigmoid()`, `gelu()`, `leaky_relu(slope)`, `softmax(dim)`, `log_softmax(dim)`,
`dropout(p)`, `layer_norm(n)`. Then `forward(x)`, `parameters()`, `param(name)` (`"0.weight"`, `"2.bias"`: PyTorch
`nn.Sequential` names), `num_parameters()`, `layers()`, `summary()` (`"linear(2, 8) -> relu -> linear(8, 1)"`), `train()`,
`eval()`, `zero_grad()`, `save(path)`, `load(path)`. Initialization is libtorch's and therefore PyTorch's: the same seed
gives the same weights as Python `nn.Linear`/`nn.Conv2d`.

`Model.save` writes the state dict (parameters and buffers) with metadata `torchcli_format=model-1` and `layers=<summary>`;
Python `safetensors.torch.load_file` + `load_state_dict(strict=True)` reads it into the equivalent `nn.Sequential`, and a
Python state dict saved with that metadata loads through `torch.load_model`. `load` is strict: missing, unexpected or
differently shaped tensors are errors.

`Optimizer`: `zero_grad()`, `step()`, `lr()`, `set_lr(lr)` (all parameter groups).

## Out and the result line

`out.set(key, float|int|string|bool|Tensor)` and `out.append(key, float|int)` build one JSON object (tensors become nested
lists, at most 1e6 elements; nonfinite floats become `null`). `out.log(text)` writes a line to stderr.
`out.check(cond, message)` and `out.fail(message)` stop the run with an application failure. `out.write_text(path, text)`
writes a file.

stdout carries exactly one line:

    {"ok": true, "result": {...}}                                           exit 0
    {"ok": false, "phase": "load", "code": "LOAD_...", "message": ...}      exit 3  (nothing ran)
    {"ok": false, "phase": "run", "code": "RUN_HOST|RUN_ERROR", "message": "dsl line N:C: ...", "line": N, "column": C}   exit 3
    {"ok": false, "phase": "app", "code": "APP_FAIL", "message": ..., "line": N, ...}                                       exit 4

Usage errors exit 2 with no result line; lmscript compile errors also exit 2. `--dsl A.dslb B.dslb ...` loads and verifies
every file before running any, then runs them in order with one host state (the `out` object, the RNG, thread settings;
autograd recording is reset to enabled at the start of each file);
each file has its own DSL locals, so files hand tensors to each other through safetensors files.

## Files

Every path is relative to, and must stay inside, the working directory (symlinks are resolved; `..` escapes and
absolute paths elsewhere are refused). DSL saves replace an existing file atomically (temporary file, then rename).
Formats: safetensors (F32, F64, I64, BOOL; header ≤ 1 MiB, file ≤ 256 MiB), CSV, text.

## Handle lifetime

Every call that returns a tensor creates a handle that lives for the whole run. A long loop should release its
temporaries: `let m = torch.mark();` at the top of the body, `torch.release(m);` at the end. Release empties every handle
created since the mark except those marked `keep()`; model parameters, optimizer state, bundles and lists keep their own
references and are unaffected. Using a released handle is an error naming `torch.release()`, never a dangling read.

## Not provided

CUDA and other devices, BLAS/LAPACK-backed linear algebra (`linalg`, `solve`, `inverse`), FFT-style spectral ops,
sparse/quantized tensors, JIT/TorchScript, data loaders and image/audio I/O. GEMM uses ATen's portable kernels.
