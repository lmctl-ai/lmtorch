# lmcpp / torchcli parity boundary

| Area | Implemented | Deferred / difference from lmrust or DESIGN.md |
|---|---|---|
| C++ SDK | Exported `lmcpp::core` CMake package; tool schema, shared validation, argv/JSON/file input, error envelopes and protocol exits | CLI11 bridge, GraphQL, client APIs, broad native-tool compatibility |
| Artifacts | Safetensors F32/F64/I64, dense CPU, multi-tensor files, string metadata, 1 MiB header and 256 MiB artifact bounds, checked offsets/overlap/coverage/length and duplicate JSON keys, atomic no-replace writes | Multiple filesystem durability guarantees beyond fsync+atomic link; uncommon safetensors encodings rejected |
| Operations | All 12 Stage 1a/1b ops are published; only `tensor_inspect` is read, all producers are write | Twelve-op table is the current product boundary; additional tensor/device features deferred |
| Numerical behavior | CPU Torch operations; local seeded generator; no broadcasting in add; stable softmax; linear is X Wᵀ+b; graph is rebuilt for sum_square and each MLP training step | Random results are reproducible for the pinned Torch/CPU build; no cross-version RNG guarantee |
| MLP | Fixed 2→4→1 ReLU checkpoint with `w1,b1,w2,b2`; metadata `torchcli_format=1`, `arch=mlp_relu_v1`, decimal `step`; plain SGD/MSE | No arbitrary model/optimizer state, momentum, dropout, or serialized autograd graph |
| Recipes | tensor chain, gradient check, and 20-step MLP recipe; each accepts a run directory, uses distinct output paths, checks tool errors, and handles simulated calls | Recipes are fixtures, not a general training framework |
| Verification | ARM64 Docker CTest uses Python Torch and safetensors as numerical/interchange oracles; tests all 12 ops, deterministic seed, resume equivalence, policy floors, malformed files and output conflict; Torch runtime and test wheels are hash-locked | OS apt repository packages are not snapshot-pinned |
| Runtime | C++ binary is the container entrypoint; host shim uses no network, mounted absolute workspace/cwd, host uid/gid, 2 CPUs/5 GiB | Python and the full Torch Python package tree remain in the runtime image |

Float outputs are JSON decimals (such as `2.0`); comparisons use numeric tolerances. The tensor rand generator is Torch's CPU generator seeded per call, never the process-global RNG. `tensor_rand` uses uniform `[0,1)` floats or uniform integers `[0,1000)` for I64.
