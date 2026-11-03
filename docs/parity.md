# lmtorch host boundary

| Area | Implemented | Deferred / boundary |
|---|---|---|
| Native host | C++ host built against the reviewed lmcode-cpp SHA in `sdk.lock`; `--lmcode`, `--lmcode-schema`, `--lmcode-signatures`, version, v1 invocation manifest and terminal return frame | No second command-operation schema or alternate CLI surface |
| Artifacts | Safetensors F32/F64/I64/BOOL, dense CPU tensors, bundles, string metadata, 1 MiB header and 256 MiB artifact bounds, checked offsets/coverage/length/duplicate keys, atomic no-replace writes and path confinement | Uncommon safetensors encodings are refused; outputs must remain beneath the invocation working directory |
| Numerical behavior | CPU Torch operations, deterministic seeded generator, tensor arithmetic, autograd, models, optimizers, CSV and safetensors through lmcode host methods | CPU only; no cross-version Torch RNG guarantee |
| Models | Generic `nn.Sequential` model construction, supported layers/optimizers, portable `Model.save`/`Model.load` checkpoints, strict tensor shape/dtype checks | Arbitrary model/optimizer state, device transfers and serialized autograd graphs are outside the contract |
| Invocation results | `InvocationResults` validates manifest v1, selects one result sink per program, preserves builder order, and emits one `LMCODE_RETURN` frame; manifest dry mode reaches the VM context | Logs and body output remain passthrough before the final frame |
| Verification | `./test.sh` checks the SDK lock, schema/signatures, flag retirement, manifest/dry/frame behavior, safetensors unit vectors, lmcode API/negative suite, examples and CPU PyTorch oracle | Python tests require a CPU PyTorch/safetensors environment; nothing Python ships in the executable |

The C++ host exposes the `Torch`, `Tensor`, `TensorList`, `Bundle`, `Model`, `Optimizer`, and `Out` schema. All filesystem
access is confined to the working directory and all native work uses CPU libtorch. The lmcode schema is the only public
host contract.
