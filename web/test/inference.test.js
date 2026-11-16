import {test} from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import * as ort from 'onnxruntime-web';
test('FP32 ONNX matches Python logits and masked argmax in JS/WASM',async t=>{
  const path=new URL('../../artifacts/export/pawnstar-fp32.onnx',import.meta.url);
  if(!fs.existsSync(path)){t.skip('Run training.export_onnx first');return;}
  ort.env.wasm.numThreads=1;
  const session=await ort.InferenceSession.create(fs.readFileSync(path),{executionProviders:['wasm']});
  const fixtures=JSON.parse(fs.readFileSync(new URL('./parity.json',import.meta.url)));
  for(const f of fixtures.slice(0,10)){
    const input=new ort.Tensor('int64',BigInt64Array.from(f.tokens,BigInt),[1,69]);
    const result=await session.run({tokens:input});
    const logits=result.policy.data;
    assert.ok(Math.max(...f.logits.map((x,i)=>Math.abs(x-logits[i])))<.0001);
    const best=f.legal.reduce((a,b)=>logits[b.id]>logits[a.id]?b:a);
    assert.equal(best.uci,f.argmax);
    input.dispose();Object.values(result).forEach(x=>x.dispose());
  }
  await session.release();
});
