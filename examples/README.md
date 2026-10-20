# torchcli examples

Two ways to drive the same `torchcli` binary from lmscript:

- **lmapi operations** (`*.lms` without `-dsl`): `import "../cli.schema.dev.json" as torchcli;` makes each of the twelve
  operations a function; every call is one torchcli process and tensors travel through safetensors files.
- **`_dsl_` programs** (`*-dsl.lms`): a `_dsl_` block is compiled by lmscript into bytecode (`savedsl`), and
  `torchcli --dsl FILE.dslb` runs it against libtorch in one process: loops, autograd, models and optimizers stay in
  memory. `lib/torchdsl.lms` holds the shared compile-and-run plumbing; the API is [docs/dsl-api.md](../docs/dsl-api.md).

Run from the repository root with the binary on PATH (or `TORCHCLI_BIN=/path/to/torchcli`); outputs go to
`artifacts/<example>/` below the working directory, which torchcli requires:

```sh
export PATH="$PWD/build/bin:$PATH"
lmscript examples/classify-moons-dsl.lms
```

## The four recipes, in both forms

| lmapi recipe | `_dsl_` version | What changes |
|---|---|---|
| [tensor-chain.lms](tensor-chain.lms) | [tensor-chain-dsl.lms](tensor-chain-dsl.lms) | four processes and four artifacts become one program; operators (`b + a`) |
| [grad-check.lms](grad-check.lms) | [grad-check-dsl.lms](grad-check-dsl.lms) | adds a finite-difference check in a DSL loop |
| [mlp-train.lms](mlp-train.lms) | [mlp-train-dsl.lms](mlp-train-dsl.lms) | 28 processes become one (11.9 s → 1.1 s on the dev host, including the lmapi cross-check); identical losses (0.886624 → 0.057287); the lmapi `mlp_forward` scores the DSL checkpoint |
| [train-report.lms](train-report.lms) | [train-report-dsl.lms](train-report-dsl.lms) | 72 torchcli calls plus 5 lmcreator calls become one program plus a Markdown report; identical metrics (R² 0.989220) |

## Multi-step tasks

| Example | Steps |
|---|---|
| [classify-moons-dsl.lms](classify-moons-dsl.lms) | prepare (generate, shuffle, split, standardize, save) → train (MLP, Adam, minibatches) → evaluate (`torch.load_model`, confusion) → lmscript retries with more epochs below target → predict a grid chosen by lmscript → text decision map and report |
| [lr-sweep-dsl.lms](lr-sweep-dsl.lms) | one program compiled per learning rate, all run in one process → lmscript discards diverged runs and picks the best → refine 5x longer → verify on held-out points (`out.check`) |
| [checkpoint-resume-dsl.lms](checkpoint-resume-dsl.lms) | uninterrupted 40 steps vs 20 + save + reload + 20 in separate processes → bit-identical checkpoints |
| [cnn-bars-dsl.lms](cnn-bars-dsl.lms) | synthesize labelled images in the DSL → conv2d/max_pool2d network → held-out accuracy → classify pictures lmscript draws as ASCII art |
| [regression-csv-dsl.lms](regression-csv-dsl.lms) | lmscript writes a CSV → DSL reads, standardizes, fits weights with Adam on hand-made parameters, writes predictions CSV → lmscript parses it and checks the recovered formula |

`train-report.lms` (lmapi) needs the private lmcreator checkout for charts and DOCX; everything else needs only torchcli and
lmscript. `../test.sh` runs every example and checks its output.
