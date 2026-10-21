#!/usr/bin/env bash
# Build a static, CPU-only, Python-free libtorch (ATen + c10 + autograd) from the pinned PyTorch source tag.
# Python is a build-time code generator here (torchgen); nothing Python is linked or shipped.
# Defaults keep the multi-hour build outside the repository so `rm -rf build` never discards it:
#   LIBTORCH_SRC=~/.cache/lmtorch/pytorch-src  LIBTORCH_BUILD=~/.cache/lmtorch/libtorch-build
#   LIBTORCH_PREFIX=~/.cache/lmtorch/libtorch-static  LIBTORCH_JOBS=3
# Linux: GCC. macOS: AppleClang, deployment target MACOSX_DEPLOYMENT_TARGET (default 13.0, the release floor).
# Windows: MSVC from Git Bash inside a Visual Studio developer environment (libtorch does not support MinGW), with the
# static runtime (/MT) so that nothing needs the Visual C++ redistributable.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAG="$(sed -n 's/^tag=//p' "$HERE/libtorch.lock")"
COMMIT="$(sed -n 's/^commit=//p' "$HERE/libtorch.lock")"
CACHE="${XDG_CACHE_HOME:-$HOME/.cache}/lmtorch"
SRC="${LIBTORCH_SRC:-$CACHE/pytorch-src}"
BUILD="${LIBTORCH_BUILD:-$CACHE/libtorch-build}"
PREFIX="${LIBTORCH_PREFIX:-$CACHE/libtorch-static}"
JOBS="${LIBTORCH_JOBS:-3}"
PYTHON="${LIBTORCH_PYTHON:-python3}"
case "$(uname -s)" in Darwin) OS=macos;; MINGW*|MSYS*|CYGWIN*) OS=windows;; *) OS=linux;; esac
# CMake and cl.exe want drive-letter paths on Windows, not Git Bash's /c/... form.
native() { if [ "$OS" = windows ]; then cygpath -m "$1"; else printf '%s\n' "$1"; fi; }
PLATFORM_ARGS=()
case "$OS" in
  macos) PLATFORM_ARGS=(-DCMAKE_OSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-13.0}");;
  windows) PLATFORM_ARGS=(-DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl
                          -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded -DCAFFE2_USE_MSVC_STATIC_RUNTIME=ON);;
esac

if [ ! -d "$SRC/.git" ]; then
  git clone --depth 1 --branch "$TAG" --recurse-submodules --shallow-submodules -j4 \
    https://github.com/pytorch/pytorch.git "$SRC"
fi
have="$(git -C "$SRC" rev-parse HEAD)"
[ -z "$COMMIT" ] || [ "$have" = "$COMMIT" ] || { echo "pytorch source is at $have, libtorch.lock pins $COMMIT" >&2; exit 1; }

# PyTorch forces USE_MIMALLOC ON for WIN32 (CMakeLists.txt `if(WIN32) set(USE_MIMALLOC ON)`, ignoring -DUSE_MIMALLOC=OFF).
# Linked statically into an /MT executable, mimalloc initializes from a TLS callback before the static CRT has created its
# heap, and the process faults before main (cdb: mi_process_init -> operator new -> RtlAllocateHeap). The source tree is
# patched to honour the option; the checked-out commit stays the pinned one. Without mimalloc c10 uses _aligned_malloc.
if [ "$OS" = windows ]; then
  sed -i 's/^  set(USE_MIMALLOC ON)/  set(USE_MIMALLOC OFF)/' "$SRC/CMakeLists.txt"   # no $: the checkout may have CRLF
  grep -q '^  set(USE_MIMALLOC OFF)' "$SRC/CMakeLists.txt" || { echo "cannot turn mimalloc off in $SRC/CMakeLists.txt" >&2; exit 1; }
fi

# torchgen needs PyYAML and typing_extensions at build time only.
"$PYTHON" -c 'import yaml, typing_extensions' 2>/dev/null || {
  echo "build-time Python needs pyyaml and typing_extensions (pip install pyyaml typing_extensions)" >&2; exit 1; }

# Every optional backend, accelerator, distributed, profiler and Python binding is off: only the CPU
# tensor library, its native kernels and autograd remain. No BLAS/LAPACK: ATen's portable GEMM is used. USE_BLAS=OFF alone
# is not enough: Dependencies.cmake falls back to find_package(BLAS) and find_package(LAPACK) unconditionally, which find
# Accelerate on macOS (and would find any system BLAS on Linux). Disabling the BLAS lookup is enough: FindLAPACK only
# accepts a LAPACK matching the BLAS found (and disabling LAPACK itself breaks configure: it includes CheckFunctionExists).
cmake -S "$(native "$SRC")" -B "$(native "$BUILD")" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$(native "$PREFIX")" \
  -DPython_EXECUTABLE="$(native "$(command -v "$PYTHON")")" \
  -DBUILD_SHARED_LIBS=OFF -DBUILD_PYTHON=OFF -DBUILD_TEST=OFF -DBUILD_BINARY=OFF \
  -DBUILD_LITE_INTERPRETER=OFF -DBUILD_CUSTOM_PROTOBUF=ON -DBUILD_LAZY_TS_BACKEND=OFF \
  -DUSE_CUDA=OFF -DUSE_ROCM=OFF -DUSE_XPU=OFF -DUSE_MPS=OFF -DUSE_VULKAN=OFF -DUSE_CUDNN=OFF -DUSE_NCCL=OFF \
  -DUSE_DISTRIBUTED=OFF -DUSE_GLOO=OFF -DUSE_MPI=OFF -DUSE_TENSORPIPE=OFF \
  -DUSE_MKLDNN=OFF -DUSE_FBGEMM=OFF -DUSE_FBGEMM_GENAI=OFF -DUSE_KLEIDIAI=OFF \
  -DUSE_NNPACK=OFF -DUSE_QNNPACK=OFF -DUSE_PYTORCH_QNNPACK=OFF -DUSE_XNNPACK=OFF \
  -DUSE_KINETO=OFF -DUSE_ITT=OFF -DUSE_NUMPY=OFF -DUSE_OPENMP=OFF -DUSE_MIMALLOC=OFF \
  -DUSE_BLAS=OFF -DUSE_LAPACK=OFF -DBLAS=Eigen -DUSE_MAGMA=OFF -DUSE_NUMA=OFF \
  -DCMAKE_DISABLE_FIND_PACKAGE_BLAS=ON \
  -DUSE_GFLAGS=OFF -DUSE_GLOG=OFF -DUSE_OPENCV=OFF -DUSE_PRECOMPILED_HEADERS=OFF \
  -DUSE_FLASH_ATTENTION=OFF -DUSE_MEM_EFF_ATTENTION=OFF -DUSE_PRIORITIZED_TEXT_FOR_LD=OFF \
  -DATEN_NO_TEST=ON ${PLATFORM_ARGS[@]+"${PLATFORM_ARGS[@]}"}
cmake --build "$(native "$BUILD")" -j"$JOBS"
cmake --install "$(native "$BUILD")"
# The static install omits several archives the static link needs; copy every archive the build produced
# (lib*.a with GCC and AppleClang, *.lib with MSVC).
mkdir -p "$PREFIX/lib"
find "$BUILD/lib" -maxdepth 1 \( -name '*.a' -o -name '*.lib' \) -exec cp -p {} "$PREFIX/lib/" \;
# License texts of every component the executable links, kept with the cached prefix for release packaging.
mkdir -p "$PREFIX/licenses"
for pair in "LICENSE:pytorch-LICENSE" "NOTICE:pytorch-NOTICE" "third_party/onnx/LICENSE:onnx-LICENSE" \
            "third_party/protobuf/LICENSE:protobuf-LICENSE" "third_party/sleef/LICENSE.txt:sleef-LICENSE" \
            "third_party/cpuinfo/LICENSE:cpuinfo-LICENSE" "third_party/fmt/LICENSE:fmt-LICENSE" \
            "third_party/miniz-3.0.2/LICENSE:miniz-LICENSE" "third_party/pocketfft/LICENSE.md:pocketfft-LICENSE" \
            "third_party/nlohmann/LICENSE.MIT:nlohmann-json-LICENSE"; do
  cp "$SRC/${pair%%:*}" "$PREFIX/licenses/${pair##*:}"
done
git -C "$SRC" rev-parse HEAD > "$PREFIX/SOURCE_COMMIT"
echo "static libtorch installed at $PREFIX"
