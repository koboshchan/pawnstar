import * as ort from 'onnxruntime-web/webgpu';
import {encodeFen} from './encoding.js';
const mobile=/iPhone|iPad|iPod|Android|Mobile/i.test(navigator.userAgent||'') || (navigator.platform==='MacIntel' && navigator.maxTouchPoints>1);
ort.env.wasm.numThreads=!mobile && globalThis.crossOriginIsolated
  ? Math.min(4,Math.max(1,navigator.hardwareConcurrency || 2)) : 1;
ort.env.wasm.simd=true;
ort.env.wasm.wasmPaths=new URL('../ort/1.31.1/', self.location.href).href;
let session, backend, url;
async function load(model, id) {
  const started=performance.now();
  let cache,cached;
  try{cache=await caches.open('pawnstar-models-v1');cached=await cache.match(model);}catch{}
  const response=cached || await fetch(model);
  const cacheCopy=!mobile&&!cached&&cache?response.clone():null;
  if(!response.ok)throw new Error('Model download failed: '+response.status);
  const total=Number(response.headers.get('Content-Length'))||55130763;
  const reader=response.body.getReader();
  let bytes=new Uint8Array(total);
  let loaded=0;
  while(true){
    const {done,value}=await reader.read();
    if(done)break;
    if(loaded+value.byteLength>bytes.length)throw new Error('Unexpected model size');
    bytes.set(value,loaded);loaded+=value.byteLength;
    self.postMessage({id,progress:true,phase:'download',loaded,total});
  }
  reader.releaseLock();
  if(loaded!==total)throw new Error('Incomplete model download');
  const downloadMs=performance.now()-started;
  if(cacheCopy)cache.put(model,cacheCopy).catch(()=>{});
  self.postMessage({id,progress:true,phase:'initializing',loaded,total,cached:!!cached});
  const initStarted=performance.now();

  if (session) await session.release();
  url=model;
  try {
    if (mobile || !navigator.gpu || !await navigator.gpu.requestAdapter()) throw new Error('WebGPU unavailable');
    session=await ort.InferenceSession.create(bytes,{executionProviders:['webgpu','wasm'],graphOptimizationLevel:'basic'});
    backend='WebGPU + WASM';
  } catch {
    session=await ort.InferenceSession.create(bytes,{executionProviders:['wasm'],graphOptimizationLevel:'basic'});
    backend='WASM';
  }
  bytes=null;
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
      try {
        try { outputs=await session.run({tokens:input}); }
        catch (e) {
          if (backend==='WASM') throw e;
          await session.release();
          const fallbackBytes=new Uint8Array(await (await fetch(url)).arrayBuffer());
          session=await ort.InferenceSession.create(fallbackBytes,{executionProviders:['wasm'],graphOptimizationLevel:'basic'});
          backend='WASM';
          outputs=await session.run({tokens:input});
        }
        self.postMessage({id:data.id,policy:Array.from(outputs.policy.data),
          value:Number(outputs.value.data[0]),backend});
      } finally {
        input.dispose();
        if(outputs)for(const tensor of Object.values(outputs))tensor.dispose();
      }
    }
  } catch(e) {self.postMessage({id:data.id,error:e.message || String(e)});}
};
