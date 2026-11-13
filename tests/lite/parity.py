#!/usr/bin/env python3
"""Run every original recipe through libtorch and lite wasm; missing coverage fails."""
import argparse, csv, json, math, os, pathlib, shutil, struct, subprocess, tempfile, time
ROOT=pathlib.Path(__file__).resolve().parents[2]
def compare(a,b,path='result',training=False):
    if isinstance(a,bool) or isinstance(a,str) or a is None:
        if path.endswith('.engine'):return
        assert a==b,(path,a,b)
    elif isinstance(a,(int,float)):
        assert isinstance(b,(int,float)) and math.isfinite(a) and math.isfinite(b),(path,a,b)
        exact=isinstance(a,int) and isinstance(b,int)
        atol,rtol=(2e-4,2e-3) if training else (2e-5,2e-4)
        if 'accuracy' in path:atol,rtol=.005,0
        assert abs(a-b)<=(0 if exact else atol+rtol*abs(a)),(path,a,b)
    elif isinstance(a,list):
        assert isinstance(b,list) and len(a)==len(b),(path,'length',len(a),len(b))
        for i,(x,y) in enumerate(zip(a,b)):compare(x,y,f'{path}[{i}]',training)
    else:
        assert set(a)==set(b),(path,'keys',set(a),set(b))
        for k in a:compare(a[k],b[k],path+'.'+k,training)
def tensors(path):
    raw=path.read_bytes();n=struct.unpack('<Q',raw[:8])[0];header=json.loads(raw[8:8+n]);result={}
    for key,desc in header.items():
        if key=='__metadata__':result[key]=desc;continue
        start,end=desc['data_offsets'];fmt={'F32':'f','F64':'d','I64':'q','BOOL':'?'}[desc['dtype']]
        result[key]={'dtype':desc['dtype'],'shape':desc['shape'],'values':list(struct.unpack('<'+fmt*((end-start)//struct.calcsize(fmt)),raw[8+n+start:8+n+end]))}
    return result
def main():
    p=argparse.ArgumentParser();p.add_argument('--native',required=True);p.add_argument('--only');args=p.parse_args()
    recipes=sorted(ROOT.glob('examples/**/*.lms'));helpers=[x for x in recipes if 'lib' in x.parts];recipes=[x for x in recipes if x not in helpers]
    if args.only:recipes=[x for x in recipes if x.stem in args.only.split(',')]
    assert recipes
    base=ROOT/'build-lite/parity';base.mkdir(exist_ok=True);report={'helpers':[str(x.relative_to(ROOT)) for x in helpers],'recipes':[]}
    for recipe in recipes:
        name=recipe.stem;entry={'source':str(recipe.relative_to(ROOT)),'runs':{}};report['recipes'].append(entry)
        for backend in ['native','wasm']:
            work=base/backend/name
            if work.exists():shutil.rmtree(work)
            work.mkdir(parents=True);log=work/'frames.jsonl'
            env=os.environ.copy();env['PATH']=str(ROOT/'build-lite/cli')+os.pathsep+env['PATH'];env['LMTORCH_PARITY_LOG']=str(log)
            if backend=='native':env['LMTORCH_PARITY_NATIVE']=str(pathlib.Path(args.native).resolve())
            else:env.pop('LMTORCH_PARITY_NATIVE',None)
            start=time.monotonic();r=subprocess.run(['lmscript',str(recipe)],cwd=work,env=env,capture_output=True,text=True,timeout=240)
            (work/'stdout.txt').write_text(r.stdout);(work/'stderr.txt').write_text(r.stderr)
            entry['runs'][backend]={'seconds':time.monotonic()-start,'exit':r.returncode}
            print(name,backend,r.returncode,round(time.monotonic()-start,3),flush=True)
            assert r.returncode==0,(name,backend,r.stdout[-3000:],r.stderr[-3000:])
        def frames(backend):return [json.loads(x) for x in (base/backend/name/'frames.jsonl').read_text().splitlines()]
        training=name not in ['tensor-sum-dsl','autograd-gradient-dsl','grad-check-dsl','grad-check','tensor-chain-dsl','tensor-chain']
        compare(frames('native'),frames('wasm'),name,training)
        native=base/'native'/name;wasm=base/'wasm'/name
        files=sorted(native.glob('artifacts/**/*'));assert sorted(str(x.relative_to(native)) for x in files if x.is_file())==sorted(str(x.relative_to(wasm)) for x in wasm.glob('artifacts/**/*') if x.is_file()),(name,'artifact set')
        for f in files:
            if f.suffix=='.safetensors':compare(tensors(f),tensors(wasm/f.relative_to(native)),str(f.relative_to(native)),training)
            elif f.suffix=='.csv':
                def cells(path):
                    return [[float(v) if v.strip().replace('.','',1).replace('-','',1).isdigit() else v for v in row] for row in csv.reader(path.open())]
                compare(cells(f),cells(wasm/f.relative_to(native)),str(f.relative_to(native)),training)
        entry['parity']='pass';(base/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    assert args.only or len(report['recipes'])+len(helpers)==len(list(ROOT.glob('examples/**/*.lms')))
    print('PASS',len(recipes),'recipes;',len(helpers),'library helpers accounted for')
if __name__=='__main__':main()
