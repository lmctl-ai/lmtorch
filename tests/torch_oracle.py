import sys
import torch
from safetensors.torch import load_file
actual = load_file(sys.argv[1], device="cpu")["tensor"]
expected = torch.tensor([[2.0, 2.0], [2.0, 2.0]], dtype=torch.float32)
assert torch.allclose(actual, expected, rtol=0, atol=0), (actual, expected)
