"""PyTorch oracle for the DSL host: the same seeds, layers, losses and optimizer steps computed by `lmtorch --dsl` and by
Python torch must agree, and model files must load in both directions. Test-only: the shipped binary contains no Python.

Usage: python dsl_oracle.py LMTORCH LMSCRIPT   (python needs torch and safetensors; cwd is irrelevant)
"""
import json
import os
import subprocess
import sys
import tempfile

import torch
from safetensors.torch import load_file, save_file

binary, lmscript = sys.argv[1:3]
here = os.path.dirname(os.path.abspath(__file__))
schema = os.path.join(os.path.dirname(here), "schema", "lmtorch.dsl.json")

PROGRAM = r'''
p = _lmcode_ schema=${SCHEMA};
torch.manual_seed(123);
out.set("rand", torch.rand(3, 2)).set("randn", torch.randn(4)).set("randint", torch.randint(0, 10, 5)).set("randperm", torch.randperm(7));
torch.manual_seed(7);
let model = torch.sequential().linear(3, 4).relu().linear(4, 2);
out.set("w0", model.param("0.weight")).set("b0", model.param("0.bias")).set("w2", model.param("2.weight"));
let x = torch.tensor("[[0.5, -1.0, 2.0], [1.5, 0.25, -0.75], [-2.0, 1.0, 0.0], [0.1, 0.2, 0.3]]");
let y = torch.tensor("[1, 0, 1, 1]");
let opt = torch.adam(model, 0.05);
for (step in range(0, 5)) {
  opt.zero_grad();
  let loss = model.forward(x).cross_entropy(y);
  loss.backward();
  opt.step();
  out.append("adam_loss", loss.item());
}
model.save("dsl-model.safetensors");
out.set("logits", model.forward(x));
let sgdm = torch.sequential().linear(3, 1);
let opt2 = torch.sgd(sgdm, 0.1, 0.9);
for (step in range(0, 4)) {
  opt2.zero_grad();
  let l = sgdm.forward(x).square().mean();
  l.backward();
  opt2.step();
}
out.set("sgd_momentum_w", sgdm.param("0.weight"));
torch.manual_seed(11);
let conv = torch.sequential().conv2d(1, 2, 3, 1, 1).relu().max_pool2d(2).flatten();
let img = torch.randn(2, 1, 4, 4);
out.set("conv", conv.forward(img)).set("img", img);
out.set("softmax", x.softmax(1)).set("log_softmax", x.log_softmax(1)).set("std", x.std(0, false)).set("gelu", x.gelu());
out.set("mse", x.mse_loss(x.sin()).item()).set("bce", x.bce_with_logits(x.sigmoid()).item());
let py = torch.load_model("py-model.safetensors");
out.set("py_forward", py.forward(x));
end _lmcode_
savedsl(p, "oracle.dslb");
'''

def close(name, actual, expected, tol=1e-5):
    # JSON carries float32 values with 9 significant digits, which round-trips exactly at float32: tol=0 means bit-equal.
    e = expected.detach()
    a = torch.tensor(actual, dtype=e.dtype)
    assert a.shape == e.shape, (name, a.shape, e.shape)
    if tol == 0:
        assert torch.equal(a, e), (name, a, e)
    else:
        assert torch.allclose(a.to(torch.float64), e.to(torch.float64), rtol=tol, atol=tol), (name, a, e)

# ignore_cleanup_errors: on Windows safetensors keeps loaded files memory-mapped (and d is the cwd), so they cannot be
# deleted while the process runs.
with tempfile.TemporaryDirectory(prefix="lmtorch-oracle-", ignore_cleanup_errors=True) as d:
    os.chdir(d)
    # A Python-trained model saved with lmtorch's metadata must load through torch.load_model.
    torch.manual_seed(99)
    py_model = torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.Tanh(), torch.nn.Linear(5, 2))
    save_file({k: v.contiguous() for k, v in py_model.state_dict().items()}, "py-model.safetensors",
              metadata={"torchcli_format": "model-1", "layers": "linear(3, 5) -> tanh -> linear(5, 2)"})
    with open("oracle.lms", "w") as f:
        f.write(PROGRAM.replace("${SCHEMA}", json.dumps(schema)))
    subprocess.run([lmscript, "oracle.lms"], check=True)
    proc = subprocess.run([binary, "--dsl", "oracle.dslb"], capture_output=True, text=True)
    assert proc.returncode == 0, (proc.returncode, proc.stdout, proc.stderr)
    r = json.loads(proc.stdout)["result"]

    torch.manual_seed(123)
    close("rand", r["rand"], torch.rand(3, 2), 0)
    close("randn", r["randn"], torch.randn(4), 0)
    assert r["randint"] == torch.randint(0, 10, (5,)).tolist(), r["randint"]
    assert r["randperm"] == torch.randperm(7).tolist(), r["randperm"]

    torch.manual_seed(7)
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2))
    close("init 0.weight", r["w0"], model[0].weight, 0)
    close("init 0.bias", r["b0"], model[0].bias, 0)
    close("init 2.weight", r["w2"], model[2].weight, 0)
    x = torch.tensor([[0.5, -1.0, 2.0], [1.5, 0.25, -0.75], [-2.0, 1.0, 0.0], [0.1, 0.2, 0.3]])
    y = torch.tensor([1, 0, 1, 1])
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    losses = []
    for _ in range(5):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    close("adam losses", r["adam_loss"], torch.tensor(losses))
    close("logits after adam", r["logits"], model(x))
    # The DSL-saved model is a plain nn.Sequential state dict.
    saved = load_file("dsl-model.safetensors")
    twin = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2))
    twin.load_state_dict(saved, strict=True)
    close("python forward of the DSL model", r["logits"], twin(x))

    # The DSL builds sgdm right after the first model, and `twin` above drew from the generator since: replay the draws.
    torch.manual_seed(7)
    _ = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2))
    sgdm = torch.nn.Linear(3, 1)
    opt2 = torch.optim.SGD(sgdm.parameters(), lr=0.1, momentum=0.9)
    for _ in range(4):
        opt2.zero_grad()
        l = sgdm(x).square().mean()
        l.backward()
        opt2.step()
    close("sgd momentum", r["sgd_momentum_w"], sgdm.weight)

    torch.manual_seed(11)
    conv = torch.nn.Sequential(torch.nn.Conv2d(1, 2, 3, stride=1, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2), torch.nn.Flatten())
    img = torch.randn(2, 1, 4, 4)
    close("conv input", r["img"], img, 0)
    close("conv forward", r["conv"], conv(img))
    close("softmax", r["softmax"], x.softmax(1))
    close("log_softmax", r["log_softmax"], x.log_softmax(1))
    close("std", r["std"], x.std(0, unbiased=True))
    close("gelu", r["gelu"], torch.nn.functional.gelu(x))
    close("mse", [r["mse"]], torch.tensor([torch.nn.functional.mse_loss(x, x.sin()).item()]))
    close("bce", [r["bce"]], torch.tensor([torch.nn.functional.binary_cross_entropy_with_logits(x, x.sigmoid()).item()]))
    close("python model in the DSL", r["py_forward"], py_model(x))
print("oracle: DSL host matches PyTorch", torch.__version__)
