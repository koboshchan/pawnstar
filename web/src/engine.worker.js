import * as ort from 'onnxruntime-web/webgpu';
import {encodeFen} from './encoding.js';
ort.env.wasm.numThreads=1;
ort.env.wasm.wasmPaths=new URL('../ort/', self.location.href).href;
let session, backend, url;
async function load(model) {
  if (session) await session.release();
  url=model;
  try {
    if (!navigator.gpu) throw new Error('WebGPU unavailable');
    session=await ort.InferenceSession.create(model,{executionProviders:['webgpu','wasm']});
    backend='WebGPU + WASM';
  } catch {
    session=await ort.InferenceSession.create(model,{executionProviders:['wasm']});
    backend='WASM';
  }
}
self.onmessage=async ({data})=>{
  try {
    if (data.type==='load') {
      await load(data.url);
      self.postMessage({id:data.id,backend,ready:true});
    } else {
      const input=new ort.Tensor('int64',BigInt64Array.from(encodeFen(data.fen),BigInt),[1,69]);
      let outputs;
      try { outputs=await session.run({tokens:input}); }
      catch (e) {
        if (backend==='WASM') throw e;
        await session.release();
        session=await ort.InferenceSession.create(url,{executionProviders:['wasm']});
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
