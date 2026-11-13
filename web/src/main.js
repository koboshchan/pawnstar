import {Chess} from 'chess.js';
import vocabulary from './moves.json';
import {legalPolicy,choose} from './encoding.js';
import './style.css';
const $=s=>document.querySelector(s);
const game=new Chess();
let orientation='w', human='w', selected=null, pending=false, ready=false, version=0, request=0;
const worker=new Worker(new URL('./engine.worker.js',import.meta.url),{type:'module'});
const waiting=new Map();
worker.onmessage=({data})=>{
  const cb=waiting.get(data.id); waiting.delete(data.id);
  if(cb) data.error ? cb.reject(new Error(data.error)) : cb.resolve(data);
};
worker.onerror=e=>{for(const cb of waiting.values())cb.reject(new Error(e.message));waiting.clear();};
function rpc(data){return new Promise((resolve,reject)=>{const id=++request;waiting.set(id,{resolve,reject});worker.postMessage({...data,id});});}
const symbols={wk:'♚',wq:'♛',wr:'♜',wb:'♝',wn:'♞',wp:'♟',bk:'♚',bq:'♛',br:'♜',bb:'♝',bn:'♞',bp:'♟'};
function status(){
  if(game.isCheckmate()) return game.turn()===human?'Pawnstar wins.':'You win.';
  if(game.isStalemate())return 'Draw by stalemate.';
  if(game.isThreefoldRepetition())return 'Draw by repetition.';
  if(game.isInsufficientMaterial())return 'Insufficient material.';
  if(game.isDraw())return 'Draw · fifty-move rule.';
  if(pending)return 'Thinking in tensors…';
  if(!ready)return 'Load a network to play.';
  return game.turn()===human ? (game.inCheck()?'You’re in check.':'Your move.') : 'Pawnstar to move.';
}
function render(){
  const last=game.history({verbose:true}).at(-1);
  const dests=selected?game.moves({square:selected,verbose:true}).map(m=>m.to):[];
  $('#board').replaceChildren();
  for(let row=0;row<8;row++)for(let col=0;col<8;col++){
    const rank=orientation==='w'?8-row:row+1, file=orientation==='w'?col:7-col;
    const sq='abcdefgh'[file]+rank,p=game.get(sq);
    const b=document.createElement('button');b.className='square '+((file+rank)%2===0?'dark':'light');
    b.dataset.square=sq;b.setAttribute('aria-label',sq+(p?' '+p.color+p.type:' empty'));
    if(sq===selected)b.classList.add('selected');
    if(last&&(sq===last.from||sq===last.to))b.classList.add('last');
    if(dests.includes(sq))b.classList.add('destination');
    if(p){
      const s=document.createElement('span');s.textContent=symbols[p.color+p.type];s.className='piece '+p.color;b.append(s);
      b.draggable=p.color===human&&game.turn()===human&&!pending&&ready&&!game.isGameOver();
      b.ondragstart=e=>{selected=sq;e.dataTransfer.setData('text/plain',sq);};
    }
    if(col===0||row===7){const c=document.createElement('small');c.className='coordinate';c.textContent=(col===0?rank:'')+(row===7?'abcdefgh'[file]:'');b.append(c);}
    b.onclick=()=>click(sq);
    b.ondragover=e=>e.preventDefault();
    b.ondrop=e=>{e.preventDefault();attempt(e.dataTransfer.getData('text/plain'),sq);};
    $('#board').append(b);
  }
  $('#status').textContent=status();
  $('#color-label').textContent='Playing '+(human==='w'?'white':'black');
  const history=game.history();$('#move-count').textContent=history.length+' PLY';
  $('#moves').replaceChildren();
  if(!history.length){const p=document.createElement('p');p.className='empty';p.textContent='A blank page. For now.';$('#moves').append(p);}
  for(let i=0;i<history.length;i+=2){
    const row=document.createElement('div');row.className='move-row';
    for(const text of [String(i/2+1).padStart(2,'0'),history[i],history[i+1]||'·']){const span=document.createElement('span');span.textContent=text;row.append(span);}
    $('#moves').append(row);
  }
  $('#moves').scrollTop=$('#moves').scrollHeight;
}
function click(sq){
  if(pending||!ready||game.turn()!==human||game.isGameOver())return;
  if(selected&&selected!==sq){attempt(selected,sq);return;}
  selected=game.get(sq)?.color===human?sq:null;render();
}
async function attempt(from,to){
  if(pending||!ready||game.turn()!==human||game.isGameOver())return;
  const options=game.moves({square:from,verbose:true}).filter(m=>m.to===to);
  if(!options.length){selected=game.get(to)?.color===human?to:null;render();return;}
  let promotion;
  if(options.some(m=>m.promotion)){
    const dialog=$('#promotion'),old=version;dialog.showModal();
    promotion=await new Promise(resolve=>dialog.addEventListener('close',()=>resolve(dialog.returnValue),{once:true}));
    if(version!==old||!['q','r','b','n'].includes(promotion))return;
  }
  game.move({from,to,...(promotion?{promotion}:{})});selected=null;render();await bot();
}
async function bot(){
  if(!ready||pending||game.turn()===human||game.isGameOver())return;
  pending=true;const old=version;render();
  try{
    const result=await rpc({type:'infer',fen:game.fen({forceEnpassantSquare:true})});
    if(version!==old)return;
    const value=1/(1+Math.exp(-result.value)),white=game.turn()==='w'?value:1-value;
    $('#eval-fill').style.height=(white*100)+'%';$('#eval-label').textContent=Math.round(white*100)+'%';
    $('#backend').textContent=result.backend;
    const move=choose(legalPolicy(game,vocabulary),result.policy,Number($('#temp').value));
    if(move)game.move({from:move.from,to:move.to,...(move.promotion?{promotion:move.promotion}:{})});
  }catch(e){ready=false;$('#backend').textContent='Inference failed';$('#status').textContent=e.message;}
  finally{pending=false;if(ready)render();}
}
$('#flip').onclick=()=>{orientation=orientation==='w'?'b':'w';render();};
$('#new').onclick=()=>{version++;game.reset();selected=null;human=$('#color').value;orientation=human;$('#eval-fill').style.height='50%';$('#eval-label').textContent='½';render();if(pending){const timer=setInterval(()=>{if(!pending){clearInterval(timer);bot();}},30);}else bot();};
$('#temp').oninput=()=>$('#temp-value').value=Number($('#temp').value).toFixed(1);
$('#load').onclick=async()=>{
  if(pending)return;
  pending=true;ready=false;$('#load').disabled=true;$('#status').textContent='Loading the network…';
  try{
    const url=new URL($('#model').value,location.href);
    if(!['http:','https:'].includes(url.protocol))throw new Error('Use an HTTP(S) model URL');
    const result=await rpc({type:'load',url:url.href});
    ready=true;$('#backend').textContent=result.backend+' · quantized';
  }catch(e){$('#backend').textContent='Network unavailable';$('#status').textContent=e.message;}
  finally{pending=false;$('#load').disabled=false;if(ready){render();bot();}}
};
const model=new URLSearchParams(location.search).get('model');
if(model)$('#model').value=model;
render();

document.querySelectorAll('[data-color]').forEach(button=>button.onclick=()=>{
  $('#color').value=button.dataset.color;
  document.querySelectorAll('[data-color]').forEach(b=>{
    b.classList.toggle('active',b===button);b.setAttribute('aria-pressed',String(b===button));
  });
});
