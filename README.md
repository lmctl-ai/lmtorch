<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/readme/hero-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="docs/readme/hero-light.png">
  <img alt="A native lmtorch model classifies a two-moons dataset with 97.5% test accuracy." src="docs/readme/hero-light.png" width="100%">
</picture>

# lmtorch

Run CPU tensors, automatic differentiation, and small neural networks from LMScript with a single native C++ host.

[![Latest release](https://img.shields.io/github/v/release/lmctl-ai/lmtorch?label=release)](https://github.com/lmctl-ai/lmtorch/releases/latest)
[![Build and test](https://github.com/lmctl-ai/lmtorch/actions/workflows/ci.yml/badge.svg)](https://github.com/lmctl-ai/lmtorch/actions/workflows/ci.yml)

## How it works

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/readme/architecture-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="docs/readme/architecture-light.png">
  <img alt="LMScript compiles against the lmtorch DSL schema, then the native CPU host returns tensor results and artifacts." src="docs/readme/architecture-light.png" width="100%">
</picture>

The same executable supports stateless safetensors operations and compiled `_dsl_` programs. Models, gradients, and tensor data stay in the native host; no Python runtime is needed.

## Examples

![The tensor-sum example returns 10; the autograd example returns the gradient 2, -4, 6.](docs/readme/gallery.png)

This gallery runs and checks both examples before drawing their values. [Tensor sum source](examples/tensor-sum-dsl.lms) · [Tensor sum book page](https://lmctl.com/tools/lmtorch/examples/tensor-sum.html) · [Gradient source](examples/autograd-gradient-dsl.lms) · [Gradient book page](https://lmctl.com/tools/lmtorch/examples/autograd-gradient.html)

The hero shows the [two-moons classifier](examples/classify-moons-dsl.lms), including its actual predicted decision grid.

## Quick start

```sh
npm install -g @lmctl-ai/lmctl
lmctl init
lmctl autorun https://lmctl.com/tools/lmtorch/lmtorch_solo.lms
```

## Learn more

- [lmtorch tool book](https://lmctl.com/tools/lmtorch/)
- [Get started](https://lmctl.com/tools/lmtorch/get_started.html)
- [DSL API](docs/dsl-api.md) · [Example guide](examples/README.md)
- [Releases and downloads](https://github.com/lmctl-ai/lmtorch/releases)

## License

MIT. Maintained by Lattice Systems LLC.
