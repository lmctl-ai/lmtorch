# Stage 1a parity boundary

| Area | Implemented in this stage | Deferred |
|---|---|---|
| Build/package | ARM64 CPU image; C++17 `torchcli`; lmcpp installed and found with its exported CMake package | Hash-locked Python wheels and multi-platform images |
| Artifact format | Single-tensor safetensors files; F32/F64/I64 dense CPU tensors; checked header, bounds, offsets, overlap, coverage, duplicate keys, and exact length; atomic no-replace writes | Multi-key files, metadata, alternate encodings, mmap, and other dtypes/devices |
| Operations | `tensor_create` and `tensor_matmul` publish write access; `tensor_inspect` publishes read access; descriptors include ref/path/key/dtype/shape/nbytes | Remaining Stage 1b+ operations and policy floors |
| Verification | CTest checks C++ roundtrip/malformed files, `ldd`, CPU-only wheel, Python safetensors interoperability, Torch matmul oracle, and output conflict | Larger numerical/property suites and performance profiling of kernel versus file I/O |
| Runtime | Host shim runs the C++ entrypoint in a bounded, offline Docker container | Long-lived service/session mode |

The output formatter preserves floating-point values as JSON decimals (for example,
`2.0`); the acceptance result is numerically equivalent to `[2,2,2,2]`. The C++
program runs directly as the container entrypoint; Python is present in the runtime
image because PyTorch's pip distribution and CMake package are installed there.

The Docker image pins the available `torch==2.9.1+cpu` CPython 3.11 ARM64 wheel,
but dependency hashes are not locked in this spike. The test stage adds Python
`safetensors` and NumPy only for interoperability tests. The runtime retains the
Torch Python package tree and its shared libraries, while excluding compiler,
headers, and build tree.
