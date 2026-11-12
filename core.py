"""Canonical orthodox chess policy encoding. Scores are White-relative."""
import hashlib
import chess
import numpy as np

def vocabulary():
    moves=set()
    for a in chess.SQUARES:
        for b in chess.SQUARES:
            x,y=abs(chess.square_file(a)-chess.square_file(b)),abs(chess.square_rank(a)-chess.square_rank(b))
            if a!=b and (x==0 or y==0 or x==y or sorted((x,y))==[1,2]):
                moves.add(chess.square_name(a)+chess.square_name(b))
    for rank,target in [(6,7),(1,0)]:
        for f in range(8):
            for g in range(max(0,f-1),min(8,f+2)):
                for p in "qrbn":
                    moves.add(chess.square_name(chess.square(f,rank))+chess.square_name(chess.square(g,target))+p)
    return sorted(moves)

VOCAB=vocabulary()
MOVE_ID={m:i for i,m in enumerate(VOCAB)}

def best_eval(record):
    """Preserve original PV order, never sort flattened PVs by score."""
    ev=max(enumerate(record["evals"]),key=lambda z:(z[1]["depth"],z[1].get("knodes",0),-z[0]))[1]
    return ev, ev["pvs"][0]

def encode(fen,line,cp=None,mate=None):
    board=chess.Board(fen)
    if not board.is_valid():
        raise ValueError("invalid orthodox board")
    move=board.parse_uci(line.split()[0]) # accepts king-to-rook castling, normalizes
    if move not in board.legal_moves:
        raise ValueError("illegal target")
    black=board.turn==chess.BLACK
    if black:
        move=chess.Move(chess.square_mirror(move.from_square),chess.square_mirror(move.to_square),move.promotion)
        board=board.mirror()
    tokens=np.zeros(128,dtype=np.uint8)
    # 0 padding; 1 empty; 2..7 white P..K; 8..13 black P..K
    for sq in chess.SQUARES:
        p=board.piece_at(sq)
        tokens[sq]=1 if p is None else 1+p.piece_type+(0 if p.color else 6)
    for i,right in enumerate((board.has_kingside_castling_rights(chess.WHITE),board.has_queenside_castling_rights(chess.WHITE),board.has_kingside_castling_rights(chess.BLACK),board.has_queenside_castling_rights(chess.BLACK))):
        tokens[64+i]=14+int(right)
    tokens[68]=16 if board.ep_square is None else 17+chess.square_file(board.ep_square)
    legal=np.zeros(246,dtype=np.uint8)
    ids=[MOVE_ID[m.uci()] for m in board.legal_moves]
    for i in ids:
        legal[i//8]|=1<<(i%8)
    key=board.fen(en_passant="fen").rsplit(" ",2)[0]
    h=int.from_bytes(hashlib.blake2b(key.encode(),digest_size=8).digest(),"little")
    sign=-1 if black else 1
    return tokens,MOVE_ID[move.uci()],legal,h, cp*sign if cp is not None else None,mate*sign if mate is not None else None
