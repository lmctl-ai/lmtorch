# Third-party components in the static torchcli executable

torchcli itself has no LICENSE file yet (a pending owner decision; see durable-memory/status.md). This file lists what the
fully static Linux executable from `scripts/build-native.sh` contains, so that the decision and any redistribution can
ship the right texts. Linked components were confirmed by symbol inspection of the aarch64 build (2026-10-20). License
texts live in the pinned sources: PyTorch v2.9.1 (`libtorch.lock`) under `~/.cache/torchcli/pytorch-src`.

| Component | Where it comes from | License | Text in the PyTorch source tree |
|---|---|---|---|
| PyTorch libtorch (c10, ATen, autograd, C++ frontend; `libtorch*.a`, `libc10.a`, `libCaffe2_perfkernels_sve.a`) | v2.9.1 | BSD-3-Clause (plus its NOTICE) | `LICENSE`, `NOTICE` |
| ONNX (`libonnx*.a`, pulled in by libtorch_cpu) | PyTorch submodule | Apache-2.0 | `third_party/onnx/LICENSE` |
| Protocol Buffers (`libprotobuf.a`, used by the ONNX code) | PyTorch submodule | BSD-3-Clause | `third_party/protobuf/LICENSE` |
| SLEEF (vectorized math) | PyTorch submodule | Boost Software License 1.0 | `third_party/sleef/LICENSE.txt` |
| cpuinfo | PyTorch submodule | BSD-2-Clause | `third_party/cpuinfo/LICENSE` |
| {fmt} | PyTorch submodule | MIT | `third_party/fmt/LICENSE` |
| miniz (compiled into libtorch_cpu) | PyTorch third_party | MIT | `third_party/miniz-3.0.2/LICENSE` |
| pocketfft (header code in ATen) | PyTorch submodule | BSD-3-Clause | `third_party/pocketfft/LICENSE.md` |
| nlohmann/json | system `nlohmann-json3-dev` 3.11.3 (and PyTorch's copy) | MIT | upstream `LICENSE.MIT` |
| cpp-dsl-sdk `dslvm.hpp` | vendored, `sdk.lock` | MIT | `third_party/dslvm/LICENSE` |
| lmcpp | pinned revision f6c6f51 | MIT | lmcpp `LICENSE` |
| GNU C Library | Ubuntu libc6 2.39-0ubuntu8.9 static objects | LGPL-2.1-or-later | — |
| libstdc++, libgcc | GCC 13.3.0 | GPL-3.0-or-later with GCC Runtime Library Exception 3.1 | — |

glibc in a static executable: LGPL-2.1 requires that recipients can relink the application against a modified library.
lmcad ships its application object files and a link recipe for that purpose (`../poc/lmcad/licenses/LINUX-RELINKING.md`);
torchcli would need the same before any binary is redistributed. The link warning about `dlopen` in a static executable
(from c10's DynamicLibrary) is expected; CPU-only torchcli loads no shared library at run time.
