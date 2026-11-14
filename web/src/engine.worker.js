import * as ort from 'onnxruntime-web/webgpu';
import {encodeFen} from './encoding.js';
ort.env.wasm.numThreads=1;
ort.env.wasm.wasmPaths=new URL('../ort/', self.location.href).href;
let session, backend, url;
async function load(model, id) {
  const response=await fetch(model);
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
  self.postMessage({id,progress:true,phase:'initializing',loaded,total});

  if (session) await session.release();
  url=bytes;
  try {
    if (!navigator.gpu) throw new Error('WebGPU unavailable');
    session=await ort.InferenceSession.create(bytes,{executionProviders:['webgpu','wasm']});
    backend='WebGPU + WASM';
  } catch {
    session=await ort.InferenceSession.create(bytes,{executionProviders:['wasm']});
    backend='WASM';
  }
}
self.onmessage=async ({data})=>{
  try {
    if (data.type==='load') {
      await load(data.url,data.id);
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
