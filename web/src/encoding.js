/** Exact preprocess/core.py encoding. FEN must preserve literal en passant. */
export function encodeFen(fen) {
  const [placement, turn, rights, ep] = fen.split(' ');
  const tokens = Array(69).fill(1), black = turn === 'b';
  let rank = 7, file = 0;
  for (const p of placement) {
    if (p === '/') { rank--; file = 0; continue; }
    if (/\d/.test(p)) { file += Number(p); continue; }
    const white = p === p.toUpperCase();
    const sq = (black ? 7-rank : rank)*8+file++;
    tokens[sq] = 2+'pnbrqk'.indexOf(p.toLowerCase()) + ((white !== black) ? 0 : 6);
  }
  const order = black ? 'kqKQ' : 'KQkq';
  for (let i=0;i<4;i++) tokens[64+i] = 14+Number(rights.includes(order[i]));
  tokens[68] = ep === '-' ? 16 : 17+ep.charCodeAt(0)-97;
  return tokens;
}
export const mirror = sq => sq[0]+(9-Number(sq[1]));
export function canonicalUci(move, black) {
  let from=move.from, to=move.to;
  // Accept king-to-rook notation; chess.js normally returns king destination.
  if (move.piece==='k' && from[0]==='e' && to[1]===from[1]) {
    if (to[0]==='h') to='g'+to[1];
    if (to[0]==='a') to='c'+to[1];
  }
  return (black ? mirror(from)+mirror(to) : from+to)+(move.promotion || '');
}
export function legalPolicy(game, vocabulary) {
  const ids = new Map(vocabulary.map((m,i)=>[m,i]));
  return game.moves({verbose:true}).map(move=>{
    const id = ids.get(canonicalUci(move,game.turn()==='b'));
    if (id===undefined) throw new Error('Legal move missing from vocabulary');
    return {move,id};
  });
}
export function choose(legal, logits, temperature=0, random=Math.random) {
  if (!legal.length) return null;
  if (!temperature) return legal.reduce((a,b)=>logits[b.id]>logits[a.id]?b:a).move;
  const max = Math.max(...legal.map(x=>logits[x.id]));
  const weights=legal.map(x=>Math.exp((logits[x.id]-max)/temperature));
  let r=random()*weights.reduce((a,b)=>a+b,0);
  for (let i=0;i<legal.length;i++) if ((r-=weights[i])<=0) return legal[i].move;
  return legal.at(-1).move;
}
