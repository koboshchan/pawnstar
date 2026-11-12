"""Cross-language regression corpus. Builds the Rust release binary once."""
import json, pathlib, random, subprocess, sys
import chess
import numpy as np
import pytest
from core import encode, best_eval, VOCAB

ROOT=pathlib.Path(__file__).resolve().parents[1]
@pytest.fixture(scope="module")
def binary():
    subprocess.run(["cargo","build","--release","--locked","--manifest-path",str(ROOT/"rust/Cargo.toml")],check=True)
    return ROOT/"rust/target/release/chess-preprocess"

def rec(fen,line,cp=19,mate=None):
    pv={"line":line}
    if cp is not None: pv["cp"]=cp
    if mate is not None: pv["mate"]=mate
    return {"fen":fen,"evals":[{"depth":30,"knodes":99,"pvs":[pv]}]}

def test_cross_language(binary,tmp_path):
    rng=random.Random(1968)
    corpus=[]
    b=chess.Board()
    for i in range(2500):
        moves=list(b.legal_moves)
        if not moves or i%250==0: b=chess.Board(); moves=list(b.legal_moves)
        m=rng.choice(moves)
        corpus.append(rec(b.fen(en_passant="fen"),b.uci(m,chess960=True),rng.choice([0,0.0,-0.0,19,-17,None]),rng.choice([None,0,3,-3])))
        b.push(m)
    for turn,rank in [("w","1"),("b","8")]:
        for rook in "ah":
            corpus.append(rec(f"r3k2r/8/8/8/8/8/8/R3K2R {turn} KQkq - 0 1",f"e{rank}{rook}{rank}"))
    for fen,line in [
        ("4k3/P7/8/8/8/8/8/4K3 w - - 0 1","a7a8n"),
        ("4k3/8/8/8/8/8/p7/4K3 b - - 0 1","a2a1q"),
        ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1","e5d6"),
        ("4k3/8/8/8/3Pp3/8/8/4K3 b - d3 0 1","e4d3"),
        # Python allows extra promoted material beyond shakmaty's check.
        ("4k3/8/8/8/8/8/QQQ5/4K3 w - - 0 1","a2a3"),
        # Python permits this double knight check.
        ("4k3/8/8/8/8/3n1n2/8/4K3 w - - 0 1","e1d1"),
        ("4k3/8/8/8/8/8/8/4K3 w K - 0 1","e1d1"),
        ("4k3/8/8/8/8/8/8/4K3 w - e6 0 1","e1d1"),
        ("4k3/8/8/8/8/8/8/4K3 w - - 0 1","a1a2"),
    ]: corpus.append(rec(fen,line))
    r=rec(chess.STARTING_FEN,"d2d4")
    r["evals"]=[
        {"depth":29,"knodes":999999,"pvs":[{"line":"g1f3","cp":1}]},
        {"depth":30,"knodes":10,"pvs":[{"line":"c2c4","cp":2}]},
        {"depth":30,"knodes":20,"pvs":[{"line":"e2e4","cp":3},{"line":"d2d4","cp":999}]},
        {"depth":30,"knodes":20,"pvs":[{"line":"g2g4","cp":4}]},
    ]
    corpus.append(r)
    # Never fall back to a lower depth when selected PV is empty/illegal.
    for pvs in [[],[{"line":"e2e5","cp":99}]]:
        corpus.append({"fen":chess.STARTING_FEN,"evals":[{"depth":1,"pvs":[{"line":"e2e4","cp":1}]},{"depth":2,"pvs":pvs}]})
    expected=[]
    for r in corpus:
        try:
            _,pv=best_eval(r)
            expected.append(encode(r["fen"],pv["line"],pv.get("cp"),pv.get("mate")))
        except (ValueError,KeyError,IndexError): pass
    source=tmp_path/"corpus.jsonl"
    source.write_text("".join(json.dumps(r)+"\n" for r in corpus))
    out=tmp_path/"rust"
    subprocess.run([str(binary),"--input",str(source),"--output",str(out),"--vocab",str(ROOT/"moves.json"),"--masks","--batch-size","111","--shard-size","700"],check=True)
    arrays={}
    for f in sorted(out.glob("shard-*.npz")):
        with np.load(f) as d:
            for name in d.files: arrays.setdefault(name,[]).append(d[name])
    arrays={name:np.concatenate(v)for name,v in arrays.items()}
    assert len(expected)==len(arrays["labels"])
    for i,x in enumerate(expected):
        for name,value,dtype in [
            ("tokens",x[0],"u1"),("labels",x[1],"<u2"),("legal",x[2],"u1"),
            ("keys",x[3],"<u8"),("cp",np.nan if x[4] is None else x[4],"<f4"),
            ("mate",0 if x[5] is None else x[5],"<i2"),
        ]:
            assert arrays[name][i].tobytes()==np.array(value,dtype=dtype).tobytes(),(i,name)
    nomasks=tmp_path/"nomasks"
    subprocess.run([str(binary),"--input",str(source),"--output",str(nomasks),"--vocab",str(ROOT/"moves.json"),"--threads","1"],check=True)
    with np.load(nomasks/"shard-000000.npz") as d:
        assert "legal" not in d.files
        for name in d.files: assert d[name].tobytes()==arrays[name].tobytes()
