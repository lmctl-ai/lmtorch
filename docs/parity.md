# lmcpp / lmtorch parity boundary

| Area | Implemented | Deferred / difference from lmrust or DESIGN.md |
|---|---|---|
| C++ SDK | Exported `lmcpp::core` CMake package; tool schema, shared validation, argv/JSON/file input, error envelopes and protocol exits | CLI11 bridge, GraphQL, client APIs, broad native-tool compatibility |
| Artifacts | Safetensors F32/F64/I64, dense CPU, multi-tensor files, string metadata, 1 MiB header and 256 MiB artifact bounds, header-first bounded reads, integer-only shape/offset fields, checked offsets/overlap/coverage/length and duplicate keys, no-replace writes, file and directory fsync, umask-respecting 0644 mode (Windows: `FlushFileBuffers` and `MoveFileExW` without `REPLACE_EXISTING`, inherited ACL, no directory sync) | Uncommon safetensors encodings rejected; directory sync failure removes the linked output and returns an error |
| Operations | All 12 Stage 1a/1b ops are published; only `tensor_inspect` is read, all producers are write | Twelve-op table is the current product boundary; additional tensor/device features deferred |
| Numerical behavior | CPU Torch operations; local seeded generator; no broadcasting in add; stable softmax; linear is X Wᵀ+b; graph is rebuilt for sum_square and each MLP training step | Random results are reproducible for the pinned Torch/CPU build; no cross-version RNG guarantee |
| MLP | Shared checkpoint validation for F32 tensors with `w1[h,in]`, `b1[h]`, `w2[out,h]`, `b2[out]`, metadata and strict unsigned step; input/target batch dimensions and dtypes are checked and violations return short validation envelopes | No arbitrary model/optimizer state, momentum, dropout, or serialized autograd graph |
| Recipes | tensor chain, gradient check, and 20-step MLP recipe; each accepts a run directory, uses distinct output paths, checks tool errors, and handles simulated calls | Recipes are fixtures, not a general training framework |
| Verification | `./test.sh`: the 12-operation Python Torch/safetensors oracle and 21 CLI regression cases (tests/integration.py), C++ safetensors malformed-header/permission checks, deterministic seed, checkpoint resume, policy floors and output conflicts, on the native binary in CI (Linux x64 and arm64) | Test-only Python with the torch wheel; nothing Python ships |
| Runtime | One native executable (libtorch, libstdc++ and libgcc static; glibc shared); lmapi operations use 2 intra-op threads and 1 inter-op thread; artifact paths are canonicalized and constrained to cwd | The docker image and host shim were removed (2026-10-20) |

Float outputs are JSON decimals (such as `2.0`); comparisons use numeric tolerances. The tensor rand generator is Torch's CPU generator seeded per call, never the process-global RNG. `tensor_rand` uses uniform `[0,1)` floats or uniform integers `[0,1000)` for I64.

`tensor_create` rejects nonfinite values, F32 overflow, nonnumeric values and I64 values that are not integral; integer JSON tokens are accepted exactly when they fit I64. Signal exits and empty-result failures do not leak raw process statuses through the protocol. `--schema`, `--list`, `--help`, and `--explain` skip Torch thread-pool setup.
