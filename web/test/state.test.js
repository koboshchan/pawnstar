import {test} from 'node:test';
import assert from 'node:assert/strict';
import {Chess} from 'chess.js';
import {saveGame,restoreGame} from '../src/state.js';
test('reload restores full moves, pending bot turn, side and orientation',()=>{
let saved;const storage={setItem:(_,v)=>saved=v,getItem:()=>saved};
const game=new Chess();for(const m of ['e4','c5','Nf3'])game.move(m);
saveGame(storage,game,'w','b');const next=new Chess();
assert.deepEqual(restoreGame(storage,next),{human:'w',orientation:'b'});
assert.equal(next.fen(),game.fen());assert.deepEqual(next.history(),game.history());
assert.equal(next.turn(),'b');
});
test('corrupt state resets and storage failures never block saving',()=>{
const game=new Chess();game.move('e4');
assert.deepEqual(restoreGame({getItem:()=>'{broken'},game),{human:'w',orientation:'w'});
assert.equal(game.history().length,0);
assert.doesNotThrow(()=>saveGame({setItem(){throw Error();}},game,'w','w'));
});
