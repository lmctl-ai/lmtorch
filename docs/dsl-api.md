# lmtorch lmcode API

`schema/lmtorch.dsl.json` is the host contract (schema 2, bytecode v6), identical to lmtorch's declarations in
`schema/lmtorch.host.json`. The pinned lmcode-cpp SDK and lmscript 0.2.450 or later supply the built-in std
schema automatically (see [lmcode std](#lmcode-std)); it does not need duplicate host declarations. `_lmcode_` is the
lmcode block spelling. `lmtorch --lmcode-schema` prints this list grouped by type; `lmtorch --lmcode-schema --json` prints
the exact host schema bytes. A block can select the executable and bind compilation to its schema:
`prog = _lmcode_ cli="lmtorch" install="github:lmctl-ai/lmtorch@latest";` (lmscript >= 0.2.450). `install=` downloads
the newest release (`@latest`; pin `@vX.Y.Z` for reproducibility and digest verification) when lmtorch is not on PATH.
`lmtorch --lmcode-signatures` prints canonical signatures, which `test.sh` diffs against
`lmscript --lmcode-signatures schema/lmtorch.dsl.json`.

A program is an lmscript `_lmcode_` block compiled to bytecode and run by the native host:

```lmscript
prog = _lmcode_ cli="lmtorch" install="github:lmctl-ai/lmtorch@latest";
let x = torch.tensor("[[1.0, 2.0], [3.0, 4.0]]").requires_grad(true);
let loss = (x.matmul(x) + 1.0).square().mean();
loss.backward();
out.set("loss", loss.item()).set("grad", x.grad());
end _lmcode_
r = run_lmcode(prog);   # uses the v1 invocation manifest and decodes the terminal return frame.
```

The language is lmscript (typed `let`, `if`/`else`, `switch`, `while`, `for (i in range(a, b))`, `for (x in list)`,
compiler-inlined `fn`, `int`/`float`/`string`/`bool`, `str()`, `float()`, `int()`, `abs`/`min`/`max`/`sqrt`/... on
primitives). lmcode programs use tensors and the host collections below for structured data.
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
| `load(path[, key])`, `save(path, Tensor)` | Tensor / void | safetensors; the single-tensor key is `tensor`, referenced as `path#tensor` |
| `bundle()`, `open(path)` | Bundle | named tensors plus string metadata in one safetensors file |
| `read_csv(path, bool header)` | Tensor | F32 `[rows, cols]`; every cell a finite number; blank lines skipped |
| `sequential()` | Model | an empty `nn.Sequential` |
| `load_model(path)` | Model | rebuilds a model saved by `Model.save` (architecture from metadata) and loads its weights |
| `sgd(Model, lr[, momentum])`, `sgd(TensorList, lr)` | Optimizer | `torch.optim.SGD` |
| `adam(Model, lr)`, `adam(TensorList, lr)`, `adamw(Model, lr, weight_decay)` | Optimizer | `torch.optim.Adam` / `AdamW` defaults otherwise |
| `mark()`, `release(mark)`, `live_tensors()` | int | handle lifetime, see below |
| `version()` | string | `lmtorch 0.5.1 (libtorch 2.9.1, CPU)` |

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

## TensorList, Bundle

`TensorList`: `add(t)` (chains), `len()`, `at(i)`; `for (t in list) { ... }` iterates it.
`Bundle`: `put(name, t)`, `get(name)`, `has(name)`, `keys()` (sorted std `StrList`), `meta(key)`, `meta(key, value)`,
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

The host writes one terminal frame containing a compact JSON envelope:

    LMCODE_RETURN_BEGIN
    {"ok":true,"result":{...}}
    LMCODE_RETURN_END

Failures use the same frame, with `phase`, `code`, `message`, optional source location/part, and exit codes 2 (CLI), 3
(load/run), or 4 (application). A manifest dry invocation adds `"dry":true`. `run_lmcode` supplies manifest v1 and
decodes this frame. All bytecode inputs are loaded before any host operation; each part has a separate result slot while
Torch state is shared for the invocation. Autograd recording resets to enabled at each part.

## Files

Every path is relative to, and must stay inside, the working directory (symlinks are resolved; `..` escapes and
absolute paths elsewhere are refused; on Windows the comparison ignores case, and alternate data streams, device names
such as `CON` and names ending in a dot or space are refused). lmcode saves replace an existing file atomically (temporary
file, then rename; `MoveFileExW` with `MOVEFILE_REPLACE_EXISTING|MOVEFILE_WRITE_THROUGH` on Windows).
Formats: safetensors (F32, F64, I64, BOOL; header ≤ 1 MiB, file ≤ 256 MiB), CSV, text.

## lmcode std

The lmcode-cpp VM registers the lmcode std catalog itself (206 signatures; lmtorch implements none of it), so every
block can use, besides lmtorch's API:

- strings: `str_len`, `str_upper`, `str_lower`, `str_trim`, `str_find`, `str_contains`, `str_starts_with`,
  `str_ends_with`, `str_substr`, `str_replace`, `str_repeat`, `str_split` (-> `StrList`), `str_match`, `str_is_int`,
  `str_is_float`, `str_to_int`, `str_to_float`;
- math (int and float overloads): `math_abs`, `math_ceil`, `math_floor`, `math_round`, `math_sqrt`, `math_pow`,
  `math_min`, `math_max`;
- collections: `str_list()`, `int_list()`, `float_list()` (`add`, `at`, `set`, `len`, `clear`, `contains`, `index_of`;
  `StrList.join(sep)`), `str_map()`;
- files: `file_read`, `file_write`, `file_append`, `file_exists`, `file_lines` (relative paths resolve against the
  working directory). Unlike lmtorch's own loads and saves, std file calls are **not** confined to the working directory
  and std writes are not atomic (a failed one can leave a partial file).

Std collections live for one run (one `lmtorch --lmcode` file); the generated reference with exact semantics and errors is
[lmcode-cpp docs/std.md](https://github.com/lmctl-ai/lmcode-cpp/blob/main/docs/std.md).

## Handle lifetime

Every call that returns a tensor creates a handle that lives for the whole run. A long loop should release its
temporaries: `let m = torch.mark();` at the top of the body, `torch.release(m);` at the end. Release empties every handle
created since the mark except those marked `keep()`; model parameters, optimizer state, bundles and lists keep their own
references and are unaffected. Using a released handle is an error naming `torch.release()`, never a dangling read.

## Not provided

CUDA and other devices, BLAS/LAPACK-backed linear algebra (`linalg`, `solve`, `inverse`), FFT-style spectral ops,
sparse/quantized tensors, JIT/TorchScript, data loaders and image/audio I/O. GEMM uses ATen's portable kernels.
