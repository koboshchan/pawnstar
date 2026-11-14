import * as ort from 'onnxruntime-web/webgpu';
import {encodeFen} from './encoding.js';
ort.env.wasm.numThreads=globalThis.crossOriginIsolated
  ? Math.min(4,Math.max(1,navigator.hardwareConcurrency || 2)) : 1;
ort.env.wasm.simd=true;
ort.env.wasm.wasmPaths=new URL('../ort/1.31.1/', self.location.href).href;
let session, backend, url;
async function load(model, id) {
  const started=performance.now();
  let cache,cached;
  try{cache=await caches.open('pawnstar-models-v1');cached=await cache.match(model);}catch{}
  const response=cached || await fetch(model);
  const cacheCopy=!cached&&cache?response.clone():null;
  if(!response.ok)throw new Error('Model download failed: '+response.status);
  const total=Number(response.headers.get('Content-Length'))||55130763;
  const reader=response.body.getReader(),chunks=[];
  let loaded=0;
  while(true){
    const {done,value}=await reader.read();
    if(done)break;
    chunks.push(value);loaded+=value.byteLength;
    self.postMessage({id,progress:true,phase:'download',loaded,total});
  }
  const bytes=new Uint8Array(loaded);
  let offset=0;
  for(const chunk of chunks){bytes.set(chunk,offset);offset+=chunk.byteLength;}
  const downloadMs=performance.now()-started;
  if(cacheCopy)cache.put(model,cacheCopy).catch(()=>{});
  self.postMessage({id,progress:true,phase:'initializing',loaded,total,cached:!!cached});
  const initStarted=performance.now();

  if (session) await session.release();
  url=bytes;
  try {
    if (!navigator.gpu || !await navigator.gpu.requestAdapter()) throw new Error('WebGPU unavailable');
    session=await ort.InferenceSession.create(bytes,{executionProviders:['webgpu','wasm'],graphOptimizationLevel:'basic'});
    backend='WebGPU + WASM';
  } catch {
    session=await ort.InferenceSession.create(bytes,{executionProviders:['wasm'],graphOptimizationLevel:'basic'});
    backend='WASM';
  }
  return {downloadMs,sessionMs:performance.now()-initStarted,cached:!!cached,backend,threads:ort.env.wasm.numThreads,isolated:!!globalThis.crossOriginIsolated};
}
self.onmessage=async ({data})=>{
  try {
    if (data.type==='load') {
      const timings=await load(data.url,data.id);
      self.postMessage({id:data.id,backend,ready:true,timings});
    } else {
      const input=new ort.Tensor('int64',BigInt64Array.from(encodeFen(data.fen),BigInt),[1,69]);
      let outputs;
      try { outputs=await session.run({tokens:input}); }
      catch (e) {
        if (backend==='WASM') throw e;
        await session.release();
        session=await ort.InferenceSession.create(url,{executionProviders:['wasm'],graphOptimizationLevel:'basic'});
        backend='WASM';
        outputs=await session.run({tokens:input});
      }
      self.postMessage({id:data.id,policy:Array.from(outputs.policy.data),
        value:Number(outputs.value.data[0]),backend});
      input.dispose();
      for (const tensor of Object.values(outputs)) tensor.dispose();
    }
  } catch(e) {self.postMessage({id:data.id,error:e.message || String(e)});}
};
