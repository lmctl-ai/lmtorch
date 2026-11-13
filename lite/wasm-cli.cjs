#!/usr/bin/env node
// Local parity adapter only. Browser hosts use MEMFS and explicit assets, never NODEFS.
const fs = require('node:fs');
const path = require('node:path');
const {spawnSync} = require('node:child_process');
const root = path.resolve(__dirname, '..');
(async () => {
  let lines = [];
  let code;
  if (process.env.LMTORCH_PARITY_NATIVE) {
    const r = spawnSync(process.env.LMTORCH_PARITY_NATIVE, process.argv.slice(2), {encoding:'utf8'});
    if (r.error) throw r.error;
    process.stdout.write(r.stdout);process.stderr.write(r.stderr);lines=r.stdout.split('\n');code=r.status;
  } else {
    const create = require(path.join(root,'build-lite/wasm/lmtorch-lite.cjs'));
    const module = await create({noInitialRun:true,print:x=>{lines.push(x);process.stdout.write(x+'\n');},printErr:x=>process.stderr.write(x+'\n')});
    for (const directory of ['/tmp','/home']) {
      if (!module.FS.analyzePath(directory).exists) module.FS.mkdir(directory);
      module.FS.mount(module.NODEFS,{root:directory},directory);
    }
    module.FS.chdir(process.cwd());
    if (process.env.LMCODE_INVOCATION_FILE) module.ENV.LMCODE_INVOCATION_FILE=process.env.LMCODE_INVOCATION_FILE;
    code = module.callMain(process.argv.slice(2));
  }
  if (process.env.LMTORCH_PARITY_LOG && process.argv.includes('--lmcode')) {
    const begin=lines.indexOf('LMCODE_RETURN_BEGIN'),end=lines.indexOf('LMCODE_RETURN_END');
    if(begin<0||end<0)throw Error('missing terminal return frame');
    fs.appendFileSync(process.env.LMTORCH_PARITY_LOG,JSON.stringify(JSON.parse(lines.slice(begin+1,end).join('\n')))+'\n');
  }
  process.exitCode=code;
})().catch(e=>{console.error(e);process.exitCode=1;});
