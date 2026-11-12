import chess
import numpy as np
from core import *
def test_vocab():
    assert len(VOCAB)==1968
    assert len(set(VOCAB))==1968
def test_initial():
    x=encode(chess.STARTING_FEN,"e2e4",19)
    assert x[0].shape==(128,) and x[0][69:].sum()==0
    assert x[2].sum()>0 and x[4]==19
def test_black():
    b=chess.Board();b.push_uci("e2e4")
    x=encode(b.fen(),"e7e5",30,-3)
    assert VOCAB[x[1]]=="e2e4" and x[4]==-30 and x[5]==3
def test_castles():
    for turn,move in [("w","e1h1"),("b","e8h8")]:
        x=encode(f"r3k2r/8/8/8/8/8/8/R3K2R {turn} KQkq - 0 1",move)
        assert VOCAB[x[1]]=="e1g1"
def test_queenside():
    x=encode("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1","e8a8")
    assert VOCAB[x[1]]=="e1c1"
def test_ep():
    x=encode("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1","e5d6")
    assert x[0][68]==20
def test_promotion():
    x=encode("4k3/P7/8/8/8/8/8/4K3 w - - 0 1","a7a8n")
    assert VOCAB[x[1]]=="a7a8n"
def test_select():
    ev,pv=best_eval({"evals":[{"depth":30,"pvs":[{"cp":1,"line":"a"}]},{"depth":31,"pvs":[{"cp":-50,"line":"b"},{"cp":999,"line":"c"}]}]})
    assert pv["line"]=="b"
def test_mask():
    x=encode(chess.STARTING_FEN,"g1f3")
    mask=np.unpackbits(x[2],bitorder="little")[:1968]
    assert mask.sum()==20 and mask[x[1]]
def test_mirror_key():
    b=chess.Board()
    x=encode(b.fen(),"e2e4")
    y=encode(b.mirror().fen(),"e7e5")
    assert x[3]==y[3] and np.array_equal(x[0],y[0])
