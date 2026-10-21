# Third-party components in the lmtorch executable

lmtorch is MIT-licensed (`LICENSE`). The release executables from `scripts/build-native.sh` (Linux, macOS and Windows)
link the components below statically; release archives (`scripts/package-release.sh`) carry each license text in `licenses/`. Linked components were
confirmed by symbol inspection of the aarch64 build (2026-10-20); libtorch comes from PyTorch v2.9.1 (`libtorch.lock`).

| Component | License | File in `licenses/` |
|---|---|---|
| PyTorch libtorch (c10, ATen, autograd, C++ frontend) | BSD-3-Clause, with NOTICE | `pytorch-LICENSE`, `pytorch-NOTICE` |
| ONNX (pulled in by libtorch_cpu) | Apache-2.0 | `onnx-LICENSE` |
| Protocol Buffers (used by the ONNX code) | BSD-3-Clause | `protobuf-LICENSE` |
| SLEEF (vectorized math) | Boost Software License 1.0 | `sleef-LICENSE` |
| cpuinfo | BSD-2-Clause | `cpuinfo-LICENSE` |
| {fmt} | MIT | `fmt-LICENSE` |
| miniz | MIT | `miniz-LICENSE` |
| pocketfft | BSD-3-Clause | `pocketfft-LICENSE` |
| nlohmann/json | MIT | `nlohmann-json-LICENSE` |
| cpp-dsl-sdk `dslvm.hpp` (vendored, `sdk.lock`) | MIT | `cpp-dsl-sdk-LICENSE` |
| lmcpp (pinned revision 93a1847) | MIT | `lmcpp-LICENSE` |
| libstdc++, libgcc (GCC; Linux executables) | GPL-3.0-or-later with GCC Runtime Library Exception 3.1 | (the exception lets compiled programs use any license) |
| Microsoft C/C++ runtime, static (`libcmt`, `libvcruntime`, `libucrt`, `libcpmt`; Windows executable) | Visual Studio license terms (Distributable Code) | (redistribution as part of a program is permitted; no notice is required) |

The GNU C Library is not part of the Linux executables: it is linked dynamically from the user's system (glibc 2.35 or
newer for the CI-built releases). `LMTORCH_FULL_STATIC=ON` would link glibc statically too, which brings LGPL-2.1
relinking duties; releases do not use it. The macOS executables use the system's libc++ and libSystem (part of macOS,
not bundled); the Windows executable imports only Windows system DLLs.
