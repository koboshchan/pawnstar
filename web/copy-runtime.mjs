import {mkdir,copyFile,readFile} from 'node:fs/promises';
const {version}=JSON.parse(await readFile('node_modules/onnxruntime-web/package.json','utf8'));
const target='public/ort/'+version;
await mkdir(target,{recursive:true});
for(const type of ['jsep','asyncify']){
  for(const ext of ['wasm','mjs']){
    const name='ort-wasm-simd-threaded.'+type+'.'+ext;
    await copyFile('node_modules/onnxruntime-web/dist/'+name,target+'/'+name);
  }
}
