# lmtorch examples

Two ways to drive the same `lmtorch` binary from lmscript:

- **lmapi operations** (`*.lms` without `-dsl`): `import "../cli.schema.dev.json" as lmtorch;` makes each of the twelve
  operations a function; every call is one lmtorch process and tensors travel through safetensors files.
- **`_dsl_` programs** (`*-dsl.lms`): a `_dsl_` block is compiled by lmscript into bytecode (`savedsl`), and
  `lmtorch --dsl FILE.dslb` runs it against libtorch in one process: loops, autograd, models and optimizers stay in
  memory. `lib/torchdsl.lms` holds the shared compile-and-run plumbing; the API is [docs/dsl-api.md](../docs/dsl-api.md).

Every block is `_dsl_ cli="lmtorch" install="github:lmctl-ai/lmtorch@v0.4.0"` (lmscript >= 0.2.359): no schema file and
no manual install.

- `cli="lmtorch"`: lmscript asks the binary for its schema (`lmtorch --dsl-schema --json`) and compiles against exactly
  those bytes, so the bytecode is bound to the binary that runs it.
- `install=`: when `lmtorch` is not on PATH, lmscript downloads the release asset `lmtorch-OS-ARCH[.exe]` once into its
  tool cache (`~/.cache/lmscript/tools`, or `LMSCRIPT_TOOL_CACHE`), verified against GitHub's SHA-256 digest.
- `dsl_cli(p)`: the argv prefix of that exact binary; the helpers run `system ${dsl_cli(p)} --dsl FILE...`.

[install-and-run-dsl.lms](install-and-run-dsl.lms) is the minimal version of this. Run from the repository root; outputs
go to `artifacts/<example>/` below the working directory, which lmtorch requires. To use a local build instead of the
release, put it first on PATH:

```sh
export PATH="$PWD/build/bin:$PATH"   # optional: a local build wins over install=
lmscript examples/classify-moons-dsl.lms
```

## The original recipes and their `_dsl_` versions

| lmapi recipe | `_dsl_` version | What changes |
|---|---|---|
| [tensor-chain.lms](tensor-chain.lms) | [tensor-chain-dsl.lms](tensor-chain-dsl.lms) | four processes and four artifacts become one program; operators (`b + a`) |
| [grad-check.lms](grad-check.lms) | [grad-check-dsl.lms](grad-check-dsl.lms) | adds a finite-difference check in a DSL loop |
| [mlp-train.lms](mlp-train.lms) | [mlp-train-dsl.lms](mlp-train-dsl.lms) | 28 processes become one (11.9 s → 1.1 s on the dev host, including the lmapi cross-check); identical losses (0.886624 → 0.057287); the lmapi `mlp_forward` scores the DSL checkpoint |
| train-report.lms (removed: it needed the private lmcreator) | [train-report-dsl.lms](train-report-dsl.lms) | 72 calls plus a separate chart and DOCX tool became one program plus a Markdown report; identical metrics (R² 0.989220) |

## Multi-step tasks

| Example | Steps |
|---|---|
| [classify-moons-dsl.lms](classify-moons-dsl.lms) | prepare (generate, shuffle, split, standardize, save) → train (MLP, Adam, minibatches) → evaluate (`torch.load_model`, confusion) → lmscript retries with more epochs below target → predict a grid chosen by lmscript → text decision map and report |
| [lr-sweep-dsl.lms](lr-sweep-dsl.lms) | one program compiled per learning rate, all run in one process → lmscript discards diverged runs and picks the best → refine 5x longer → verify on held-out points (`out.check`) |
| [checkpoint-resume-dsl.lms](checkpoint-resume-dsl.lms) | uninterrupted 40 steps vs 20 + save + reload + 20 in separate processes → bit-identical checkpoints |
| [cnn-bars-dsl.lms](cnn-bars-dsl.lms) | synthesize labelled images in the DSL → conv2d/max_pool2d network → held-out accuracy → classify pictures lmscript draws as ASCII art |
| [regression-csv-dsl.lms](regression-csv-dsl.lms) | lmscript writes a CSV → DSL reads, standardizes, fits weights with Adam on hand-made parameters, writes predictions CSV → lmscript parses it and checks the recovered formula |

Every example needs only lmtorch and lmscript. `../test.sh` runs every example and checks its output.
