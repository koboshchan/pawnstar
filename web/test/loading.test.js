import {test} from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
test('stream loading caches bytes, reports progress and uses basic optimization',async()=>{
  const events=[],options=[],store=new Map();
  let fetches=0;
  const ctx=vm.createContext({
    performance,Uint8Array,Response,Number,Error,crossOriginIsolated:true,
    self:{location:{href:'https://chess.kjt.lol/assets/worker.js'},postMessage:x=>events.push(x)},
    navigator:{hardwareConcurrency:8},
    caches:{open:async()=>({match:async key=>store.get(key)?.clone(),put:async(key,r)=>{store.set(key,new Response(await r.arrayBuffer()));}})},
    fetch:async()=>{fetches++;return new Response(new Uint8Array([1,2,3]),{headers:{'Content-Length':'3'}});},
    ort:{env:{wasm:{}},InferenceSession:{create:async(bytes,o)=>{assert.equal(bytes.length,3);options.push(o);return {release:async()=>{}};}}}
  });
  let code=fs.readFileSync(new URL('../src/engine.worker.js',import.meta.url),'utf8')
    .replace(/^import .*;$/gm,'').replace("ort.env.wasm.wasmPaths=new URL('../ort/1.31.1/', self.location.href).href;",'');
  vm.runInContext(code,ctx);
  await ctx.self.onmessage({data:{type:'load',id:1,url:'https://chess.kjt.lol/model.onnx'}});
  await new Promise(resolve=>setImmediate(resolve));
  await ctx.self.onmessage({data:{type:'load',id:2,url:'https://chess.kjt.lol/model.onnx'}});
  assert.equal(fetches,1);
  assert.equal(options[0].graphOptimizationLevel,'basic');
  assert.equal(events.at(-1).timings.threads,4);
  assert.equal(events.at(-1).timings.isolated,true);
  assert.ok(events.some(e=>e.phase==='download'&&e.loaded===3&&e.total===3));
  assert.ok(events.some(e=>e.phase==='initializing'));
  assert.equal(events.at(-1).timings.cached,true);
  assert.equal(events.at(-1).backend,'WASM');
});
