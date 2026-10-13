import hashlib
import json
import os
import subprocess
import sys
import tempfile
import torch
from safetensors.torch import load_file, save_file

binary, unit, oracle = sys.argv[1:]
if not os.path.exists(binary): binary = binary.replace("/build-torchcli/", "/build/")
if not os.path.exists(unit): unit = unit.replace("/build-torchcli/", "/build/")
ldd = subprocess.run(["ldd", binary], text=True, capture_output=True, check=True)
assert "not found" not in ldd.stdout, ldd.stdout
assert not torch.cuda.is_available(), "CPU wheel unexpectedly reports CUDA"
subprocess.run([unit], check=True)

active_cwd=None
def call(*args, env=None, ok=True):
    p = subprocess.run([binary, *map(str, args)], text=True, capture_output=True, env=env, cwd=active_cwd)
    if ok and p.returncode != 0: raise AssertionError((args, p.returncode, p.stdout, p.stderr))
    return p

def result(*args): return json.loads(call(*args).stdout)
def approx(path, name, expected):
    actual = load_file(path, device="cpu")[name]
    assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-6), (actual, expected)

with tempfile.TemporaryDirectory(prefix="torchcli-test-") as d:
    active_cwd=d
    paths = {name: os.path.join(d, name + ".safetensors") for name in (
        "a", "b", "mm", "sum", "soft", "linear", "lininput", "packed", "loaded", "grad",
        "rand1", "rand2", "randi64", "checkpoint", "fwd", "trained", "split1", "split2", "full1", "full2")}
    schema = json.loads(call("--schema").stdout)
    ops = {op["name"]: op for op in schema["operations"]}
    assert len(ops) == 12
    assert all((op["access"] == "read") == (name == "tensor_inspect") for name, op in ops.items())
    # 1 tensor_create and 2 tensor_rand (local generator reproducibility across processes).
    vals = [1.0, 2.0, 3.0, 4.0]
    result("tensor_create", "--input-json", json.dumps({"shape":[2,2],"values":vals,"out":paths["a"]}))
    approx(paths["a"], "tensor", torch.tensor(vals, dtype=torch.float32).reshape(2, 2))
    seed = 741
    for name in ("rand1", "rand2"):
        result("tensor_rand", "--shape", "2", "--shape", "3", "--seed", seed, "--out", paths[name])
    generated = torch.rand((2, 3), generator=torch.Generator(device="cpu").manual_seed(seed))
    approx(paths["rand1"], "tensor", generated); approx(paths["rand2"], "tensor", generated)
    assert load_file(paths["rand1"])["tensor"].equal(load_file(paths["rand2"])["tensor"])
    result("tensor_rand", "--shape", "2", "--seed", seed, "--dtype", "I64", "--out", paths["randi64"])
    expected_i64=torch.randint(0,1000,(2,),generator=torch.Generator(device="cpu").manual_seed(seed),dtype=torch.int64)
    assert load_file(paths["randi64"])["tensor"].equal(expected_i64)
    # 3 matmul, 4 add, 5 softmax, 6 linear.
    save_file({"tensor": torch.tensor([[2.0, 0.0], [0.0, 2.0]])}, paths["b"])
    result("tensor_matmul", "--a", paths["a"]+"#tensor", "--b", paths["b"]+"#tensor", "--out", paths["mm"])
    ma = torch.tensor([[1., 2.], [3., 4.]]) @ torch.tensor([[2., 0.], [0., 2.]])
    approx(paths["mm"], "tensor", ma)
    result("tensor_add", "--a", paths["a"]+"#tensor", "--b", paths["a"]+"#tensor", "--out", paths["sum"])
    approx(paths["sum"], "tensor", torch.tensor([[2., 4.], [6., 8.]]))
    result("tensor_softmax", "--input", paths["a"]+"#tensor", "--dim", 1, "--out", paths["soft"])
    approx(paths["soft"], "tensor", torch.softmax(torch.tensor([[1., 2.], [3., 4.]]), dim=1))
    save_file({"tensor": torch.tensor([[1., 2.], [3., 4.]]), "bias": torch.tensor([.5, -1.])}, paths["lininput"])
    result("tensor_linear", "--input", paths["lininput"]+"#tensor", "--weight", paths["b"]+"#tensor", "--bias", paths["lininput"]+"#bias", "--out", paths["linear"])
    approx(paths["linear"], "tensor", torch.nn.functional.linear(torch.tensor([[1.,2.],[3.,4.]]),torch.tensor([[2.,0.],[0.,2.]]),torch.tensor([.5,-1.])))
    # 7 tensor_save, 8 tensor_load (multi-key metadata survives both).
    model_meta={"torchcli_format":"1","arch":"mlp_relu_v1","step":"0"}
    saved = result("tensor_save", "--tensors", json.dumps({"left": paths["a"]+"#tensor", "right": paths["b"]+"#tensor"}), "--metadata", json.dumps(model_meta), "--out", paths["packed"])
    assert set(saved["tensors"]) == {"left", "right"}
    from safetensors import safe_open
    with safe_open(paths["packed"],framework="pt") as f: assert f.metadata()==model_meta
    loaded = result("tensor_load", "--path", paths["packed"], "--out", paths["loaded"])
    assert set(loaded["tensors"]) == {"left", "right"}
    approx(paths["loaded"], "left", torch.tensor([[1.,2.],[3.,4.]]))
    # 9 inspect.
    inspected = result("tensor_inspect", "--input", paths["loaded"]+"#right")
    assert inspected["shape"] == [2, 2] and inspected["values"] == [2.0, 0.0, 0.0, 2.0]
    # 10 autograd_grad.
    grad = result("autograd_grad", "--input", paths["a"]+"#tensor", "--out", paths["grad"])
    approx(paths["grad"], "tensor", 2*torch.tensor([[1.,2.],[3.,4.]]))
    assert abs(grad["loss"] - 30.0) < 1e-6
    # 11/12 fixed MLP inference and one-step MSE/SGD, checked against Python Torch.
    torch.manual_seed(3)
    model = {"w1":torch.rand((4,2)), "b1":torch.rand(4), "w2":torch.rand((1,4)), "b2":torch.rand(1)}
    metadata={"torchcli_format":"1", "arch":"mlp_relu_v1", "step":"0"}
    save_file(model, paths["checkpoint"], metadata=metadata)
    x=torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
    y=torch.tensor([[0.],[1.],[1.],[0.]])
    xfile=os.path.join(d,"x.safetensors"); yfile=os.path.join(d,"y.safetensors")
    save_file({"tensor":x},xfile); save_file({"tensor":y},yfile)
    fwd=result("mlp_forward","--checkpoint",paths["checkpoint"],"--input",xfile+"#tensor","--out",paths["fwd"])
    ref=((x@model["w1"].T+model["b1"]).relu()@model["w2"].T+model["b2"])
    approx(paths["fwd"],"tensor",ref)
    lr=0.03
    trained=result("mlp_train_step","--checkpoint",paths["checkpoint"],"--input",xfile+"#tensor","--target",yfile+"#tensor","--lr",lr,"--out",paths["trained"])
    assert trained["step"] == 1 and abs(trained["loss"]-torch.nn.functional.mse_loss(ref,y).item())<1e-5
    py={k:v.clone().requires_grad_(True) for k,v in model.items()}
    pred=(x@py["w1"].T+py["b1"]).relu()@py["w2"].T+py["b2"]
    torch.nn.functional.mse_loss(pred,y).backward()
    expected={k:(v.detach()-lr*v.grad).detach() for k,v in py.items()}
    actual=load_file(paths["trained"])
    for k in model: assert torch.allclose(actual[k],expected[k],rtol=1e-5,atol=1e-6),(k,actual[k],expected[k])
    saved_meta=__import__("safetensors").safe_open(paths["trained"],framework="pt").metadata()
    assert saved_meta == {"torchcli_format":"1","arch":"mlp_relu_v1","step":"1"}
    # Continue a 2-step checkpoint in a new process and compare with uninterrupted 2 steps.
    base=paths["checkpoint"]
    for idx,out in enumerate((paths["split1"],paths["split2"])):
        result("mlp_train_step","--checkpoint",base,"--input",xfile+"#tensor","--target",yfile+"#tensor","--lr",lr,"--out",out);base=out
    base=paths["checkpoint"]
    for idx,out in enumerate((paths["full1"],paths["full2"])):
        result("mlp_train_step","--checkpoint",base,"--input",xfile+"#tensor","--target",yfile+"#tensor","--lr",lr,"--out",out);base=out
    split=load_file(paths["split2"]);full=load_file(paths["full2"])
    assert all(torch.allclose(split[k],full[k],rtol=0,atol=0) for k in split)
    # Dry-run and readonly must leave every artifact byte-for-byte unchanged.
    before={f:hashlib.sha256(open(os.path.join(d,f),"rb").read()).hexdigest() for f in os.listdir(d) if f.endswith(".safetensors")}
    dry=call("--dryrun","tensor_create","--shape","2","--out",os.path.join(d,"dry.safetensors"));assert dry.returncode==0 and json.loads(dry.stdout)["dryRun"]
    env=os.environ.copy();env["TORCHCLI_READONLY"]="yes"
    blocked=call("tensor_create","--shape","2","--out",os.path.join(d,"blocked.safetensors"),env=env,ok=False);assert blocked.returncode==6 and json.loads(blocked.stdout)["error"]["code"]=="READONLY_VIOLATION"
    after={f:hashlib.sha256(open(os.path.join(d,f),"rb").read()).hexdigest() for f in os.listdir(d) if f.endswith(".safetensors")}
    assert before==after and not os.path.exists(os.path.join(d,"dry.safetensors")) and not os.path.exists(os.path.join(d,"blocked.safetensors"))
    # Ensure the exact same checkpoint forwards after serialization/reload.
    reload_path=os.path.join(d,"reloaded.safetensors")
    result("tensor_load","--path",paths["checkpoint"],"--out",reload_path)
    rerun=result("mlp_forward","--checkpoint",reload_path,"--input",xfile+"#tensor","--out",os.path.join(d,"reload_fwd.safetensors"))
    assert torch.allclose(load_file(os.path.join(d,"reload_fwd.safetensors"))["tensor"],load_file(paths["fwd"])["tensor"],rtol=0,atol=0)
    # Regression vectors for Opus findings 10-20. These assertions were added
    # before the fixes and intentionally fail against the prior implementation.
    def rejected(code, *args):
        p=call(*args,ok=False)
        assert p.returncode==2, (args,p.returncode,p.stdout,p.stderr)
        assert "Exception raised from" not in p.stderr and "backtrace" not in p.stdout.lower(), (p.stdout,p.stderr)
        envelope=json.loads(p.stdout)
        assert envelope["error"]["code"]==code, envelope
        if args and args[0] in ("mlp_forward","mlp_train_step"):
            assert envelope["error"]["data"].get("tool_code")=="INVALID_CHECKPOINT", envelope
        return envelope
    for badstep in ("-1", "7abc"):
        badmodel=os.path.join(d,"bad-step-"+str(abs(hash(badstep)))+".safetensors")
        save_file(model,badmodel,metadata={**metadata,"step":badstep})
        rejected("VALIDATION_ERROR","mlp_train_step","--checkpoint",badmodel,"--input",xfile+"#tensor","--target",yfile+"#tensor","--lr",lr,"--out",os.path.join(d,"bad-step-out.safetensors"))
        rejected("VALIDATION_ERROR","mlp_forward","--checkpoint",badmodel,"--input",xfile+"#tensor","--out",os.path.join(d,"bad-step-fwd.safetensors"))
    malformed={"w1":model["w1"],"b1":torch.zeros(1),"w2":torch.zeros((3,4)),"b2":torch.zeros(1)}
    badmodel=os.path.join(d,"bad-shape.safetensors");save_file(malformed,badmodel,metadata=metadata)
    rejected("VALIDATION_ERROR","mlp_forward","--checkpoint",badmodel,"--input",xfile+"#tensor","--out",os.path.join(d,"bad-fwd.safetensors"))
    rejected("VALIDATION_ERROR","mlp_train_step","--checkpoint",badmodel,"--input",xfile+"#tensor","--target",yfile+"#tensor","--lr",lr,"--out",os.path.join(d,"bad-train.safetensors"))
    for tensors,name in (({**model,"w1":model["w1"].double()},"bad-dtype"),({**model,"w2":torch.zeros((3,4))},"bad-w2"),({**model,"b2":torch.zeros(2)},"bad-b2")):
        path=os.path.join(d,name+".safetensors");save_file(tensors,path,metadata=metadata)
        rejected("VALIDATION_ERROR","mlp_forward","--checkpoint",path,"--input",xfile+"#tensor","--out",os.path.join(d,name+"-out.safetensors"))
    rejected("VALIDATION_ERROR","mlp_forward","--checkpoint",paths["checkpoint"],"--input",paths["randi64"]+"#tensor","--out",os.path.join(d,"bad-inputdtype.safetensors"))
    ybad=os.path.join(d,"bad-target.safetensors");save_file({"tensor":torch.zeros((3,1))},ybad)
    rejected("VALIDATION_ERROR","mlp_train_step","--checkpoint",paths["checkpoint"],"--input",xfile+"#tensor","--target",ybad+"#tensor","--lr",lr,"--out",os.path.join(d,"bad-batch.safetensors"))
    for val,dtype in (("not-a-number","F32"),(1e100,"F32"),(1.5,"I64"),(9223372036854775808,"I64")):
        rejected("VALIDATION_ERROR","tensor_create","--input-json",json.dumps({"shape":[1],"dtype":dtype,"values":[val],"out":os.path.join(d,"bad-value.safetensors")}))
    exact_i64=os.path.join(d,"exact-i64.safetensors")
    result("tensor_create","--input-json",'{"shape":[1],"dtype":"I64","values":[9007199254740993],"out":"'+exact_i64+'"}')
    assert load_file(exact_i64)["tensor"].item()==9007199254740993
    rejected("VALIDATION_ERROR","tensor_softmax","--input",paths["a"]+"#tensor","--dim",9223372036854775808,"--out",os.path.join(d,"bad-dim.safetensors"))
    rejected("VALIDATION_ERROR","tensor_create","--shape","2","--out","../torchcli-escape.safetensors")
    with tempfile.TemporaryDirectory(prefix="torchcli-outside-") as outside:
        os.symlink(outside,os.path.join(d,"escape"))
        rejected("VALIDATION_ERROR","tensor_create","--shape","2","--out",os.path.join(d,"escape","symlink-out.safetensors"))
        external=os.path.join(outside,"external.safetensors");save_file({"tensor":torch.ones(1)},external)
        rejected("VALIDATION_ERROR","tensor_inspect","--input",external+"#tensor")
    nf=os.path.join(d,"nonfinite.safetensors");save_file({"tensor":torch.cat((torch.ones(32),torch.full((8,),float("nan"))))},nf)
    nfresult=result("tensor_inspect","--input",nf+"#tensor","--limit",1)
    assert nfresult["values"]==[1.0] and nfresult["nonfinite_count"]==8,nfresult
    print("torchcli CLI regression cases: 21 passed (checkpoint shapes/metadata, values, softmax bounds, nonfinite count and path containment)")
    print("torchcli integration: all 12 operations, Torch numerical oracle, deterministic rand, checkpoint resume, readonly/dryrun, ldd and CPU-only passed")
