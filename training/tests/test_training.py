import json
from dataclasses import asdict
import random
import subprocess
import sys
import chess
import numpy as np
import pytest
import torch
from preprocess.core import encode, VOCAB
from training.data import Shards, value_targets
from training.model import Config, PolicyValue, CONFIGS
from training.play import encode_board, legal_ids, Player
from training.train import objective

torch.set_num_threads(2)

def test_encoding_parity():
    board = chess.Board()
    rng = random.Random(12)
    boards = [
        chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"),
        chess.Board("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1"),
        chess.Board("4k3/P7/8/8/8/8/7p/4K3 w - - 0 1"),
        chess.Board("4k3/8/8/8/8/8/7p/4K3 b - - 0 1"),
        chess.Board("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1"),
        chess.Board("4k3/8/8/8/3Pp3/8/8/4K3 b - d3 0 1"),
    ]
    for _ in range(100):
        if board.is_game_over():
            board.reset()
        boards.append(board.copy())
        board.push(rng.choice(list(board.legal_moves)))
    for b in boards:
        tokens = encode_board(b)
        for move, label in legal_ids(b):
            reference, target, legal, *_ = encode(b.fen(en_passant="fen"), move.uci())
            assert np.array_equal(tokens, reference[:69])
            assert label == target
            assert legal[label//8] & (1 << (label%8))

def test_value_targets():
    target, valid = value_targets([0,400,-400,np.nan,np.nan,np.nan], [0,0,0,2,-2,0])
    assert target[0] == .5
    assert target[1] > .5 > target[2]
    assert list(target[3:5]) == [1,0]
    assert list(valid) == [1,1,1,1,1,0]

def make_shards(tmp_path):
    board = chess.Board()
    rows=[]
    for _ in range(24):
        move = list(board.legal_moves)[0]
        tok, label, *_ = encode(board.fen(en_passant="fen"), move.uci())
        rows.append((tok[:69],label))
        board.push(move)
        if board.is_game_over():
            board.reset()
    for i in range(3):
        np.savez(tmp_path/f"shard-{i:04d}.npz",
                 tokens=np.array([r[0] for r in rows],dtype=np.uint8),
                 labels=np.array([r[1] for r in rows],dtype=np.uint16),
                 cp=np.zeros(24,dtype=np.float32),mate=np.zeros(24,dtype=np.int16),
                 split=np.array([0]*16+[1]*4+[2]*4,dtype=np.uint8))

def test_loader_split_and_workers(tmp_path):
    make_shards(tmp_path)
    from torch.utils.data import DataLoader
    for split, expected in [(0,48),(1,12),(2,12)]:
        for workers in (0,2):
            batches=list(DataLoader(Shards(tmp_path,split,7), batch_size=None,num_workers=workers))
            assert sum(len(b[0]) for b in batches)==expected
            assert all(b[0].shape[1]==69 for b in batches)

def test_loss_drop_and_player(tmp_path):
    torch.manual_seed(4)
    make_shards(tmp_path)
    batch=next(iter(Shards(tmp_path,0,16)))
    model=PolicyValue(Config(64,2,4))
    optimizer=torch.optim.AdamW(model.parameters(),lr=.003)
    losses=[]
    for _ in range(25):
        optimizer.zero_grad()
        loss,*_=objective(model(batch[0]),batch,.1)
        losses.append(float(loss.detach()))
        loss.backward()
        optimizer.step()
    assert losses[-1] < losses[0]*.7
    path=tmp_path/"model.pt"
    torch.save(dict(config=asdict(model.config),model=model.state_dict()),path)
    player=Player(path)
    for fen in [chess.STARTING_FEN,"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1"]:
        b=chess.Board(fen)
        assert player.move(b) in b.legal_moves
    result=subprocess.run([sys.executable,"-m","training.play",str(path)],
        input="uci\nisready\nposition startpos moves e2e4\ngo\nquit\n",
        text=True,capture_output=True,check=True)
    assert "uciok" in result.stdout and "readyok" in result.stdout
    b=chess.Board()
    b.push_uci("e2e4")
    assert chess.Move.from_uci(result.stdout.split("bestmove ")[1].split()[0]) in b.legal_moves
    print(f"CPU tiny overfit policy+value loss {losses[0]:.4f} -> {losses[-1]:.4f}")

def test_parameter_sizes():
    for name, bounds in [("small",(10e6,12e6)),("medium",(49e6,52e6)),("large",(150e6,160e6))]:
        with torch.device("meta"):
            m=PolicyValue(CONFIGS[name])
        assert bounds[0] < sum(p.numel() for p in m.parameters()) < bounds[1]

def test_cli_resume_and_retention(tmp_path):
    data=tmp_path/"data"
    data.mkdir()
    make_shards(data)
    out=tmp_path/"run"
    resume=tmp_path/"resume"
    command=[sys.executable,"-m","training.train","--data",str(data),
             "--device","cpu","--config","small","--batch-size","16",
             "--workers","2","--steps","3","--warmup","0",
             "--save-every","1","--eval-every","3","--eval-batches","2",
             "--log-every","1"]
    import os
    env=dict(os.environ,OMP_NUM_THREADS="2",MKL_NUM_THREADS="2")
    subprocess.run(command+["--output",str(out)],check=True,capture_output=True,env=env)
    assert [p.name for p in sorted(out.glob("step-*.pt"))] == ["step-000000002.pt","step-000000003.pt"]
    subprocess.run(command+["--output",str(resume),"--resume",str(out/"step-000000002.pt")],
                   check=True,capture_output=True,env=env)
    a=torch.load(out/"step-000000003.pt",weights_only=True)
    b=torch.load(resume/"step-000000003.pt",weights_only=True)
    assert all(torch.equal(a["model"][k],b["model"][k]) for k in a["model"])
    assert a["consumed"] == b["consumed"]
    result=subprocess.run(command+["--output",str(resume),"--resume",str(out/"step-000000002.pt"),
                                  "--batch-size","8"],capture_output=True,env=env,text=True)
    assert result.returncode and "Resume requires unchanged --batch-size" in result.stderr

def test_remote_retention_only_after_success(tmp_path, monkeypatch):
    from argparse import Namespace
    from training.train import save_checkpoint
    model=PolicyValue(Config(32,1,4))
    optimizer=torch.optim.AdamW(model.parameters())
    args=Namespace(output=str(tmp_path),hf_run="test-run")
    calls=[]
    monkeypatch.setattr(subprocess,"run",lambda cmd,**kw: calls.append(cmd))
    history=[]
    for step in range(1,4):
        save_checkpoint(model,optimizer,args,step,step,history)
    assert len(history)==2
    assert len(list(tmp_path.glob("*.pt")))==2
    assert calls[-1] == ["hf","buckets","rm",
        "hf://buckets/koboshchan/checkpoints/pawnstar/runs/test-run/step-000000001.pt","-y"]
    def fail(*args,**kwargs):
        raise subprocess.CalledProcessError(1,args[0])
    monkeypatch.setattr(subprocess,"run",fail)
    with pytest.raises(subprocess.CalledProcessError):
        save_checkpoint(model,optimizer,args,4,4,history)
    assert (tmp_path/"step-000000002.pt").exists()
    assert (tmp_path/"step-000000004.pt").exists()
