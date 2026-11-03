# lmtorch examples

All examples compile lmcode programs and invoke the native host through `run_lmcode`. The invocation manifest and
terminal return frame are handled by lmscript, while tensor values, gradients, models and optimizers stay in the CPU
libtorch host.

Every program declares `cli="lmtorch"` and `install="github:lmctl-ai/lmtorch@latest"`. A local binary on `PATH` wins;
otherwise lmscript installs the selected release into its private tool cache and verifies its digest. Run from the
repository root so generated artifacts stay in a scratch workspace:

```sh
export PATH="$PWD/build/bin:$PATH"   # optional local native build
lmscript examples/classify-moons.lms
```

## Recipes

| Entry point | lmcode program | Coverage |
|---|---|---|
| [tensor-chain.lms](tensor-chain.lms) | [tensor-chain-dsl.lms](tensor-chain-dsl.lms) | Tensor creation, matmul, add, softmax, save and inspect |
| [grad-check.lms](grad-check.lms) | [grad-check-dsl.lms](grad-check-dsl.lms) | Autograd gradient and central-difference check |
| [mlp-train.lms](mlp-train.lms) | [mlp-train-dsl.lms](mlp-train-dsl.lms) | 20-step SGD training, portable model checkpoint and reload check |
| [train-report-dsl.lms](train-report-dsl.lms) | same | Loss table and report generation |
| [classify-moons-dsl.lms](classify-moons-dsl.lms) | same | Dataset preparation, MLP training, prediction and decision map |
| [lr-sweep-dsl.lms](lr-sweep-dsl.lms) | same | Multi-program invocation, sweep selection and refinement |
| [checkpoint-resume-dsl.lms](checkpoint-resume-dsl.lms) | same | Uninterrupted versus save/reload training equivalence |
| [cnn-bars-dsl.lms](cnn-bars-dsl.lms) | same | CPU convolutional image classifier |
| [regression-csv-dsl.lms](regression-csv-dsl.lms) | same | CSV input, fitting and prediction output |
| [install-and-run-dsl.lms](install-and-run-dsl.lms) | same | Private release install and native invocation |

The original short recipe filenames remain runnable entry points and use the same lmcode host as the longer example names.

Every example needs lmtorch and lmscript only. `../test.sh` runs the examples, host checks, invocation-manifest controls,
CPU-only C++ unit tests, and (when a CPU PyTorch wheel is available) numerical comparisons against Python PyTorch.
