const KEY='pawnstar-game-v1';
export function saveGame(storage,game,human,orientation){
  try{storage.setItem(KEY,JSON.stringify({v:1,moves:game.history(),fen:game.fen(),human,orientation}));}catch{}
}
export function restoreGame(storage,game){
  const defaults={human:'w',orientation:'w'};
  try{
    const state=JSON.parse(storage.getItem(KEY));
    if(state?.v!==1||!Array.isArray(state.moves)||state.moves.length>2000
      ||!['w','b'].includes(state.human)||!['w','b'].includes(state.orientation))return defaults;
    game.reset();
    for(const move of state.moves)game.move(move);
    if(game.fen()!==state.fen)throw new Error('Invalid saved position');
    return {human:state.human,orientation:state.orientation};
  }catch{game.reset();return defaults;}
}
