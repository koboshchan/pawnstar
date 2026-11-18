from dataclasses import asdict
import ast
from pathlib import Path
import chess
import numpy as np
import pytest
import torch
from training.model import Config, PolicyValue
from training.play import encode_board, legal_ids, Player
from training.rl.search import Node, backup, terminal_value, search, Network
from training.rl.selfplay import selfplay, targets, save_replay, load_replay
from training.rl.train import batch, objective, load_model
from training.rl.evaluate import evaluate, paired_openings, pure_moves

torch.set_num_threads(2)


def uniform(boards):
    return [(legal_ids(b), np.ones(b.legal_moves.count())/b.legal_moves.count(), .5)
            for b in boards]


def tiny():
    return PolicyValue(Config(width=16, layers=1, heads=2))


def test_backup_probability_perspective():
    root, child, leaf = Node(), Node(), Node()
    backup([root, child, leaf], .9)
    assert [n.visits for n in (root, child, leaf)] == [1, 1, 1]
    assert [n.total for n in (root, child, leaf)] == pytest.approx([.9, .1, .9])


def test_terminal_values():
    b = chess.Board()
    for uci in ("f2f3", "e7e5", "g2g4", "d8h4"):
        b.push_uci(uci)
    assert terminal_value(b) == 0
    assert terminal_value(chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")) == .5


def test_search_prefers_mate_and_preserves_board():
    b = chess.Board("7k/8/5KQ1/8/8/8/8/8 w - - 0 1")
    before = b.fen()
    legal, pi = search([b], uniform, simulations=120, noise=0)[0]
    b2 = b.copy()
    b2.push(legal[pi.argmax()][0])
    assert b2.is_checkmate()
    assert b.fen() == before
    assert pi.sum() == pytest.approx(1)
    assert all(m in b.legal_moves for m, _ in legal)


@pytest.mark.parametrize("fen", [
    chess.STARTING_FEN,
    "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
    "7k/8/8/8/3pP3/8/8/K7 b - e3 0 1",
    "7k/P7/8/8/8/8/7p/K7 w - - 0 1"])
def test_canonical_search_actions(fen):
    board = chess.Board(fen)
    (legal, pi), = search([board], uniform, simulations=3, noise=0)
    assert set(i for _, i in legal) == set(i for _, i in legal_ids(board))
    assert pi.sum() == pytest.approx(1)


def test_search_bounds_and_terminal():
    with pytest.raises(ValueError):
        search([chess.Board()], uniform, simulations=0)
    with pytest.raises(ValueError):
        search([chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")], uniform)


def test_outcome_targets_and_truncation():
    tokens = encode_board(chess.Board())
    h = [(tokens, [1], [1.], chess.WHITE), (tokens, [1], [1.], chess.BLACK)]
    win = targets(h, chess.Outcome(chess.Termination.CHECKMATE, chess.WHITE))
    assert [r[3:] for r in win] == [(1., 1.), (0., 1.)]
    assert [r[3:] for r in targets(h, None)] == [(.5, 0.), (.5, 0.)]
    draw = targets(h, chess.Outcome(chess.Termination.STALEMATE, None))
    assert all(r[3:] == (.5, 1.) for r in draw)


def test_repetition_preserved():
    b = chess.Board()
    for uci in ("g1f3", "g8f6", "f3g1", "f6g8")*2:
        b.push_uci(uci)
    assert terminal_value(b) == .5


def test_bounded_selfplay_reproducible_roundtrip(tmp_path):
    rows, result = selfplay(uniform, games=3, parallel=2, simulations=2, max_plies=4, seed=3)
    other, _ = selfplay(uniform, games=3, parallel=2, simulations=2, max_plies=4, seed=3)
    assert len(rows) == 12
    assert result == {"truncated": 3}
    assert all(r[4] == 0 for r in rows)
    assert all(np.array_equal(a[0], b[0]) for a, b in zip(rows, other))
    save_replay(tmp_path/"replay.npz", rows)
    restored = load_replay(tmp_path/"replay.npz")
    assert len(restored) == len(rows)
    assert np.array_equal(restored[0][0], rows[0][0])


def test_loss_anchor_and_value_mask_gradients():
    rows, _ = selfplay(uniform, games=1, simulations=2, max_plies=2)
    data = batch(rows, [0, 1], "cpu")
    logits = torch.randn(2, 1968, requires_grad=True)
    value = torch.zeros(2, requires_grad=True)
    loss, report = objective((logits, value), logits.detach().clone(), data)
    loss.backward()
    assert report["kl"] == pytest.approx(0, abs=1e-6)
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad[~data[1]]) == 0
    assert torch.count_nonzero(value.grad) == 0
    _, shifted = objective((logits, value), torch.zeros_like(logits), data)
    assert shifted["kl"] > 0


def test_real_model_step_and_frozen_anchor():
    torch.manual_seed(4)
    model = tiny()
    anchor = tiny().eval().requires_grad_(False)
    before = {k: v.clone() for k, v in anchor.state_dict().items()}
    rows, _ = selfplay(uniform, games=1, simulations=2, max_plies=2)
    rows = [(*r[:3], 1., 1.) for r in rows]
    data = batch(rows, [0, 1], "cpu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    with torch.no_grad():
        anchor_logits, _ = anchor(data[0])
    loss, _ = objective(model(data[0]), anchor_logits, data)
    loss.backward()
    assert model.value.weight.grad.abs().sum() > 0
    optimizer.step()
    assert all(torch.equal(v, before[k]) for k, v in anchor.state_dict().items())
    assert all(p.grad is None for p in anchor.parameters())


def test_ema_load_and_pure_export_compatibility(tmp_path):
    model = tiny()
    ema = {k: torch.zeros_like(v) for k, v in model.state_dict().items()}
    path = tmp_path/"ema.pt"
    torch.save(dict(config=asdict(model.config), model=model.state_dict(), ema=ema), path)
    restored = load_model(path, require_ema=True)
    assert all(torch.count_nonzero(v) == 0 for v in restored.state_dict().values())
    board = chess.Board()
    assert Player(path).move(board) in board.legal_moves
    assert Network(restored)([board])[0][1].sum() == pytest.approx(1)
    torch.save(dict(config=asdict(model.config), model=model.state_dict()), path)
    with pytest.raises(ValueError):
        load_model(path, require_ema=True)


def test_paired_eval_pure_and_cap():
    openings = paired_openings(3, 7)
    assert len(openings) == 3
    assert [b.fen() for b in openings] == [b.fen() for b in paired_openings(3, 7)]
    calls = []
    def recorded(boards):
        calls.append(len(boards))
        return uniform(boards)
    report = evaluate(recorded, recorded, pairs=3, max_plies=2, seed=7)
    assert report["games"] == 6
    assert report["truncated"] == 6
    assert report["score"] is None
    assert sum(calls) == 12
    assert all(pair == [None, None] for pair in report["pair_scores"])


def test_serving_has_no_rl_imports():
    root = Path(__file__).parents[3]
    for directory in ("engine", "lichess"):
        for path in (root/directory).rglob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    assert not (node.module or "").startswith("training.rl")
                if isinstance(node, ast.Import):
                    assert all(not alias.name.startswith("training.rl") for alias in node.names)
