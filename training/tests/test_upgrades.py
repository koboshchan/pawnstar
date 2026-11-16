import json
import os
from pathlib import Path
import random
import subprocess
import sys
import chess
import numpy as np
import pytest
import torch
from preprocess.core import encode
from training.data import Shards
from training.legal import legal_masks, mask_from_tokens
from training.model import Config, PolicyValue
from training.tests.test_training import make_shards
from training.train import objective, save_checkpoint


def test_masks_match_reference_special_and_random():
    boards = [chess.Board(f) for f in (
        chess.STARTING_FEN,
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
        "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
        "4k3/8/8/8/3Pp3/8/8/4K3 b - d3 0 1",
        "4k3/P7/8/8/8/8/7p/4K3 w - - 0 1",
        "4k3/8/8/8/8/8/7p/4K3 b - - 0 1",
    )]
    rng = random.Random(71)
    board = chess.Board()
    for _ in range(200):
        if board.is_game_over():
            board.reset()
        boards.append(board.copy())
        board.push(rng.choice(list(board.legal_moves)))
    for board in boards:
        move = next(iter(board.legal_moves))
        tok, label, packed, *_ = encode(board.fen(en_passant="fen"), move.uci())
        actual = legal_masks(tok[None, :69], np.array([label]))
        expected = legal_masks(tok[None, :69], np.array([label]), packed[None])
        assert np.array_equal(actual, expected)
    with pytest.raises(ValueError):
        mask_from_tokens(np.zeros(69, dtype=np.uint8))


def test_loader_stored_and_computed_masks(tmp_path):
    make_shards(tmp_path)
    generated = list(Shards(tmp_path, 1, 7, masked_policy=True))
    for path in tmp_path.glob("*.npz"):
        with np.load(path) as z:
            arrays = dict(z)
        masks = np.stack([mask_from_tokens(row) for row in arrays["tokens"]])
        arrays["legal"] = np.packbits(masks, axis=1, bitorder="little")
        np.savez(path, **arrays)
    stored = list(Shards(tmp_path, 1, 7, masked_policy=True))
    assert all(torch.equal(a, b) for x, y in zip(generated, stored) for a, b in zip(x, y))
    # Bad masks must fail, never silently unmask the target.
    with pytest.raises(ValueError):
        legal_masks(np.zeros((1,69)), np.array([0]), np.zeros((1,246),dtype=np.uint8))


def test_masked_gradient_and_loss():
    logits = torch.tensor([[2., 1., 100.]], requires_grad=True)
    value = torch.tensor([0.], requires_grad=True)
    batch = (torch.zeros(1,69), torch.tensor([0]), torch.tensor([.5]),
             torch.ones(1), torch.tensor([[True, True, False]]))
    loss, policy, _ = objective((logits,value), batch, .5)
    assert torch.allclose(policy, torch.nn.functional.cross_entropy(logits[:,:2], batch[1]))
    loss.backward()
    assert logits.grad[0,2] == 0
    with pytest.raises(ValueError):
        objective((logits,value), batch[:4]+(torch.tensor([[False,True,False]]),), .5)


def test_fork_and_ema_resume(tmp_path):
    data = tmp_path/"data"
    data.mkdir()
    make_shards(data)
    out, resumed, forked = (tmp_path/n for n in ("run","resumed","forked"))
    cmd = [sys.executable,"-m","training.train","--data",str(data),"--device","cpu",
           "--config","small","--batch-size","16","--workers","2","--steps","3",
           "--warmup","0","--save-every","1","--eval-every","3","--eval-batches","1",
           "--log-every","1","--ema-decay",".9"]
    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    subprocess.run(cmd+["--output",str(out)], check=True,capture_output=True,env=env)
    path = out/"step-000000002.pt"
    subprocess.run(cmd+["--resume",str(path),"--output",str(resumed)],
                   check=True,capture_output=True,env=env)
    a = torch.load(out/"step-000000003.pt",weights_only=True)
    b = torch.load(resumed/"step-000000003.pt",weights_only=True)
    for key in ("model","ema"):
        assert all(torch.equal(a[key][k],b[key][k]) for k in a[key])
    assert any(not torch.equal(a["model"][k],a["ema"][k]) for k in a["model"])
    subprocess.run(cmd+["--fork-from",str(path),"--skip-batches","4","--output",str(forked)],
                   check=True,capture_output=True,env=env)
    f = torch.load(forked/"step-000000003.pt",weights_only=True)
    assert f["args"]["skip_batches"] == 4
    assert f["consumed"] == 3
    assert min(int(v["step"]) for v in f["optimizer"]["state"].values()) == 5
    failed = subprocess.run(cmd+["--fork-from",str(path),"--skip-batches","1",
                                  "--output",str(tmp_path/"bad")],capture_output=True,env=env,text=True)
    assert failed.returncode and "worker-aligned" in failed.stderr
