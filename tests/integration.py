import json
import os
import subprocess
import sys
import tempfile
import torch
from safetensors.torch import save_file
binary, unit, oracle = sys.argv[1:]
if not os.path.exists(binary): binary = binary.replace("/build-torchcli/", "/build/")
if not os.path.exists(unit): unit = unit.replace("/build-torchcli/", "/build/")
ldd = subprocess.run(["ldd", binary], text=True, capture_output=True, check=True)
assert "not found" not in ldd.stdout, ldd.stdout
assert not torch.cuda.is_available(), "CPU wheel unexpectedly reports CUDA"
subprocess.run([unit], check=True)
def call(*args, ok=True):
    p = subprocess.run([binary, *map(str, args)], text=True, capture_output=True)
    if ok and p.returncode != 0: raise AssertionError((args, p.returncode, p.stdout, p.stderr))
    return p
with tempfile.TemporaryDirectory(prefix="torchcli-test-") as d:
    a=os.path.join(d,"a.safetensors"); b=os.path.join(d,"b.safetensors"); c=os.path.join(d,"python.safetensors")
    schema=json.loads(call("--schema").stdout); assert schema["tool"]=="torchcli"
    created=json.loads(call("tensor_create","--shape","2","--shape","2","--fill","1","--out",a).stdout)
    assert created["shape"]==[2,2] and created["dtype"]=="F32"
    product=json.loads(call("tensor_matmul","--a",a+"#tensor","--b",a+"#tensor","--out",b).stdout)
    assert product["shape"]==[2,2]
    inspected=json.loads(call("tensor_inspect","--input",b+"#tensor").stdout)
    assert inspected["shape"]==[2,2] and inspected["values"]==[2.0,2.0,2.0,2.0]
    subprocess.run([sys.executable,oracle,b],check=True)
    save_file({"tensor":torch.tensor([[3.0,4.0]],dtype=torch.float64)},c)
    from_python=json.loads(call("tensor_inspect","--input",c+"#tensor").stdout)
    assert from_python["dtype"]=="F64" and from_python["values"]==[3.0,4.0]
    conflict=call("tensor_create","--shape","2","--shape","2","--out",a,ok=False)
    assert conflict.returncode==5 and json.loads(conflict.stdout)["error"]["code"]=="CONFLICT"
print("torchcli integration: ldd, CPU-only, safetensors, matmul oracle and conflict passed")
