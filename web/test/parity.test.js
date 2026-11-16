import {test} from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {Chess} from 'chess.js';
import {encodeFen,legalPolicy,choose,canonicalUci} from '../src/encoding.js';
const vocabulary=JSON.parse(fs.readFileSync(new URL('../src/moves.json',import.meta.url)));
const fixtures=JSON.parse(fs.readFileSync(new URL('./parity.json',import.meta.url)));
test('Python tokens, legal IDs and FP32 masked argmax match JS on all fixtures',()=>{
  for(const f of fixtures){
    assert.deepEqual(encodeFen(f.fen),f.tokens,f.fen);
    const game=new Chess(f.fen),legal=legalPolicy(game,vocabulary);
    assert.deepEqual(legal.map(x=>x.id).sort((a,b)=>a-b),f.legal.map(x=>x.id).sort((a,b)=>a-b),f.fen);
    const m=choose(legal,f.logits);
    assert.equal(m.from+m.to+(m.promotion||''),f.argmax,f.fen);
  }
});
test('castling normalization and black rank mirroring',()=>{
  assert.equal(canonicalUci({from:'e1',to:'h1',piece:'k'},false),'e1g1');
  assert.equal(canonicalUci({from:'e8',to:'a8',piece:'k'},true),'e1c1');
});
test('temperature only chooses legal moves including deterministic endpoints',()=>{
  const legal=[{id:0,move:'a'},{id:2,move:'b'}];
  assert.equal(choose(legal,[0,999,1]),'b');
  assert.equal(choose(legal,[0,999,1],1,()=>0),'a');
  assert.equal(choose([],[]),null);
});
