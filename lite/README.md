# Portable lmtorch core (experimental ML browser profile)

C++17 tensors and reverse-mode differentiation without libtorch. `tensor.hpp` owns F32/F64/I64/BOOL storage,
broadcasting, reductions, rank-2 matrix products, views, losses and the graph. `bindings.hpp` shares the lmcode host
between `host.cpp` (native/wasm CLI) and lmlive's ML extension. Both use the native host's byte-identical DSLB v6 VM.

Build with `lite/build.sh` (or `native` / `wasm`). Set `EMSDK` to an Emscripten 7.0.0 installation.
The script downloads nlohmann JSON 3.11.3 with a pinned SHA-256 and generates the byte-exact embedded host schema.
The separate `lite/CMakeLists.txt` also builds the target after those inputs have been staged.
Neither target finds or links libtorch. Native libtorch remains the reference and the ordinary release executable.
The browser compiler is not included in this public repository.

Run `python3 tests/lite/parity.py --native /path/to/native/lmtorch`. It discovers every example recipe, runs the full
lmscript orchestration, compares terminal results and safetensors tensors/metadata, and records a coverage manifest
in `build-lite/parity/report.json`. `examples/lib/torchdsl.lms` is a library helper, explicitly accounted for.
The local wasm CLI adapter mounts host paths with NODEFS solely for recipe testing. Browser lmlive uses MEMFS and
its explicit same-origin asset map. No native filesystem is exposed by the browser.

The host registers the complete compatibility schema, and unsupported calls fail explicitly. Initial limitations:
rank-2 matmul; no nll_loss or smooth_l1_loss; no GPU, external model execution, distributed or sparse tensors.
Not every registered operator has a full PyTorch conformance claim: the acceptance gate is the example suite and the
independent core fixtures. The ML track's capability profile must stay within that tested surface.
Tensor size is bounded at 16 million elements, output at one million, wasm memory at 512 MiB, one thread. The page
must also enforce a wall timeout and cancellation via its Worker. There are no internal long-running threads.

Numerical parity uses finite checks, exact integer/boolean/shape/metadata comparisons, F32 forward tolerance
`2e-5 + 2e-4*abs(reference)` and training `2e-4 + 2e-3*abs(reference)`; held-out accuracy difference <=0.005.
F64 core tests cover exact arithmetic and finite differences. Checkpoint resume is bit-identical within a backend.
The lr-sweep example now uses 0.1 instead of 0.2 as its largest candidate below the intentional divergent 3.0.
At 0.2 with momentum, tiny CPU-kernel roundoff led to different 750-step trajectories and final parameter valleys,
so that rate was unsuitable for a portable numerical demonstration. The tighter parity tolerances were retained.

The RNG follows the documented ATen CPU draw order (MT19937, low 24-bit F32 uniforms, randperm and normal blocks),
with seeded initialization tested through the recipes. Reference: PyTorch v2.9.1
[DistributionTemplates.h](https://github.com/pytorch/pytorch/blob/v2.9.1/aten/src/ATen/native/cpu/DistributionTemplates.h)
and [DistributionsHelper.h](https://github.com/pytorch/pytorch/blob/v2.9.1/aten/src/ATen/core/DistributionsHelper.h).
No PyTorch code or library is linked into lite.
