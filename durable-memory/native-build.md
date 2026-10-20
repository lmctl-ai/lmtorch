# Native build (no Python at run time)

Goal from the owner (2026-10-19): one lmtorch binary with no runtime dependencies, built from pure C/C++ libraries,
without docker, Python runtime, GUI or display code. Status of that goal: see status.md.

## How

1. `scripts/build-libtorch.sh` clones PyTorch at `libtorch.lock` (tag v2.9.1, commit d38164a5) with submodules into
   `~/.cache/lmtorch/pytorch-src` and builds a **static, CPU-only libtorch** into `~/.cache/lmtorch/libtorch-static`
   (outside the repo so `rm -rf build` never discards it). Python is used only by torchgen code generation at build time
   (needs `pyyaml`, `typing_extensions`). Everything optional is off: CUDA/ROCm/XPU/MPS/Vulkan, distributed, MKLDNN, FBGEMM,
   KleidiAI, NNPACK/QNNPACK/XNNPACK, Kineto, ITT, NumPy, OpenMP, MKL/BLAS/LAPACK, MAGMA, flash/mem-efficient attention,
   Python bindings, tests. `BLAS=Eigen` with `USE_BLAS=OFF` is required: `BLAS=None` is a configure error
   ("Unrecognized BLAS option"); with USE_BLAS off ATen uses its portable GEMM.
2. `scripts/build-native.sh` extracts lmcpp at its pinned revision (f6c6f51, via `git archive`, so the checkout's branch
   does not matter) into the build tree, verifies the SDK pin, and configures CMake with `LMTORCH_LIBTORCH_PREFIX`.
   CMake links `libtorch.a`, `libtorch_cpu.a` and `libc10.a` with `--whole-archive` (operator/dispatcher registrations are
   static initializers), every other archive in the prefix inside `--start-group/--end-group` (protoc, gtest, gmock and
   benchmark excluded), plus `-static-libstdc++ -static-libgcc`. Release form: glibc stays shared (owner decision
   2026-10-20: no LGPL relinking duty, same one-file install); `LMTORCH_FULL_STATIC=ON` adds `-static`.
   `scripts/package-release.sh` strips the binary and bundles LICENSE, THIRD_PARTY_NOTICES.md and the component licenses
   that build-libtorch.sh copies into `$PREFIX/licenses`.
3. `scripts/build-native.sh --dev` builds `build-dev/` against the PyTorch wheel's shared libtorch instead (minutes, not
   hours); the wheel lives in `~/.cache/lmtorch/oracle-venv`, which is also the test oracle (torch 2.9.1+cpu, safetensors,
   numpy; installed without hashes, it is a dev tool, not a shipped input).

## Measurements (aarch64 dev host, 4 cores, 15 GiB, shared with other teams' builds, 2026-10-20)

- PyTorch source with submodules: about 2.6 GB. Configure + build + install at -j3: 1537 ninja steps, 1 h 33 min wall
  (configure 23:56, install 01:29) while other agents loaded the machine to 10-40. Peak memory stayed below ~10 GiB.
- Installed archives: libtorch_cpu.a 398 MB, then onnx, protobuf, protoc (not linked), sleef, fmt, cpuinfo, c10, torch.
- lmtorch static link: 48 s. Executable 184,599,624 bytes unstripped (122,127,864 stripped), `file`: statically linked,
  `ldd`: not a dynamic executable. `--dsl-schema` starts in 0.21 s with 94 MB max RSS.
- Static vs dev (wheel) build: classify-moons-dsl 1.33 s vs 1.90 s, cnn-bars-dsl 0.74 s vs 0.94 s (startup dominates).
- Numerics: the static build's BLAS-free GEMM differs from the wheel's kernels in the last float32 bits (lmapi mlp-train
  first loss 0.886623740 static vs 0.886623621 wheel); the PyTorch oracles (tolerance 1e-5; RNG/init exact) pass on both.

## Gotchas

- `torch/torch.h` (C++ frontend: nn, optim) is part of libtorch_cpu; nothing extra is needed for Sequential/Adam.
- The static link warns once: `Using 'dlopen' in statically linked applications` (c10 DynamicLibrary); expected, nothing is
  loaded at run time on CPU.
- libprotobuf and libonnx are really linked (libtorch_cpu's ONNX code references them); see THIRD_PARTY_NOTICES.md.
- Docker remains the frozen oracle path (`scripts/build.sh`, `docker/Dockerfile`, CI); its staged context must include
  `schema/`, `third_party/` and `sdk.lock` now that the binary embeds the DSL host, and it enables
  `-DLMTORCH_PYTHON_TESTS=ON` to keep the Python CTest entries.
