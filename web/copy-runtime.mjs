import {mkdir,copyFile} from 'node:fs/promises';
await mkdir('public/ort',{recursive:true});
for(const type of ['jsep','asyncify']){
  for(const ext of ['wasm','mjs']){
    const name='ort-wasm-simd-threaded.'+type+'.'+ext;
    await copyFile('node_modules/onnxruntime-web/dist/'+name,'public/ort/'+name);
  }
}
