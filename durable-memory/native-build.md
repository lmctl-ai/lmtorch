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

## Per-OS release builds (2026-10-21)

All five release forms come from the same two scripts and the same cmake option list; only the toolchain differs.
CI runners, cold libtorch build at the given -j (CI run 39531212351): linux-arm64 (ubuntu-22.04-arm, -j4) 38 min,
linux-x64 (ubuntu-22.04, -j4) 61 min, macos-arm64 (macos-15, -j3) 39 min, macos-x64 (macos-15-intel, -j4) 52 min,
windows-x64 (windows-2022, -j4) 93 min (with mimalloc; run 39531212351). lmtorch itself builds in under a minute and test.sh runs in about one.

- **BLAS stays off everywhere** only with `-DCMAKE_DISABLE_FIND_PACKAGE_BLAS=ON`: PyTorch's Dependencies.cmake calls
  `find_package(BLAS)` (and then `find_package(LAPACK)`) even with `USE_BLAS=OFF`, which finds Accelerate on macOS and
  made libtorch_cpu reference LAPACK symbols (`zunmqr_`, `cblas_sdot`, ...). Linux had been BLAS-free only because the
  runners have no system BLAS. Do not also disable the LAPACK lookup: configure then dies with `Unknown CMake command
  "CHECK_FUNCTION_EXISTS"` (FindLAPACK is what includes it); FindLAPACK only accepts a LAPACK matching the BLAS found.
- **macOS** (AppleClang from Xcode 16): `CMAKE_OSX_DEPLOYMENT_TARGET` 13.0 for libtorch (`MACOSX_DEPLOYMENT_TARGET`,
  default 13.0) and lmtorch (CMakeLists default), so the floor is macOS 13 Ventura (`otool -l`: minos 13.0). Link:
  `-Wl,-force_load,<lib>` for torch/torch_cpu/c10, the other archives plain (ld64 rescans archives; no groups). Only
  `/usr/lib/libc++.1.dylib` and `/usr/lib/libSystem.B.dylib` are imported (test.sh gate: `otool -L`). nlohmann-json from
  Homebrew (header-only). x64: PyTorch publishes no macOS x86_64 wheel since 2.2, so the oracles are skipped there (29
  gates instead of 31); arm64 runs them against the `torch==2.9.1` wheel.
- **Windows** (MSVC 2022 from Git Bash after ilammy/msvc-dev-cmd; libtorch does not support MinGW): static runtime
  `/MT` everywhere: `-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded` (PyTorch 2.9 requires CMake 3.27, so CMP0091 is NEW and
  its own `/MD`->`/MT` flag rewriting finds nothing to rewrite; subprojects with old policies get no runtime flag, which
  is cl's `/MT` default; a mismatch would be a LNK2038 link error) and the same in lmtorch's CMakeLists. Paths go to CMake
  in drive-letter form (`cygpath -m`); source and build trees on `D:/pt` (short paths, MAX_PATH; `core.longpaths`).
  Archives are `*.lib`; link with `/WHOLEARCHIVE:` per whole library. Gate: `dumpbin -dependents` (`-` not `/`, which
  Git Bash would rewrite into a path) must list only Windows system DLLs, no vcruntime/msvcp/ucrtbase/api-ms-win-crt.
  nlohmann-json from vcpkg via `CMAKE_PREFIX_PATH`. `.gitattributes` forces LF (sdk.lock hash, byte-identical schemas).
  Stdout/stderr/stdin are switched to binary mode in main (CRLF would break `--dsl-schema --json` and JSON lines).
  File writes use CreateFileW(CREATE_NEW) + FlushFileBuffers + MoveFileExW (see dsl-host.md).
- test.sh under Git Bash: binaries are `*.exe`; anything a native program reads from a file, argument it builds or
  environment variable gets the drive-letter form (`npath`); the oracles get npm's `lmscript.cmd`; the venv python is
  `Scripts/python.exe`.
- **mimalloc** (Windows only): PyTorch forces `USE_MIMALLOC ON` for WIN32 regardless of `-DUSE_MIMALLOC=OFF`. Statically
  linked into an /MT executable it initializes from a TLS callback before the static CRT heap exists: lmtorch.exe died
  before main with an access violation (exit 139 under Git Bash; cdb stack mi_process_init -> __dyn_tls_init ->
  operator new -> RtlAllocateHeap). build-libtorch.sh patches that line of the pinned tree to OFF (sed without `$`: the
  Windows checkout may be CRLF); c10 then uses `_aligned_malloc`. Verified first with a two-function shim
  (`mi_malloc_aligned`/`mi_free` -> `_aligned_malloc`/`_aligned_free`) against the cached build, then removed.
  The Windows release imports only KERNEL32.dll and dbghelp.dll (c10 stack traces).
- A PDB is written next to lmtorch.exe (`/DEBUG /OPT:REF /OPT:ICF`, not shipped); test.sh's smoke run prints the exit
  status and, on Windows, a cdb stack when the binary cannot start.
- The Python oracles on Windows: safetensors keeps loaded files memory-mapped, so temp directories are created with
  `ignore_cleanup_errors=True`; JSON built by string concatenation must escape backslash paths (`json.dumps`).
- mlp-train keeps the 6-digit check on every OS: all five platforms print loss first 0.886624 / last 0.057287 (DSL) and
  0.88662... / 0.057286 (lmapi); no tolerance change was needed.

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
- CI saves the libtorch cache with actions/cache/save right after the cold build, so a later failing step does not
  throw the hours away (the combined actions/cache only saves when the whole job succeeds).
