"""Bounded, batched self-play and sparse replay shards."""
from collections import Counter
import chess
import numpy as np
from training.play import encode_board
from training.rl.search import search


def targets(history, outcome):
    # Do not label a move-cap truncation as a draw or use network bootstraps.
    return [(tokens, ids, pi,
             .5 if outcome is None or outcome.winner is None
             else float(turn == outcome.winner), float(outcome is not None))
            for tokens, ids, pi, turn in history]


def selfplay(infer, games=32, parallel=32, simulations=32, max_plies=300,
             seed=42, temperature_plies=24, noise=.25):
    if min(games, parallel, max_plies, simulations) < 1:
        raise ValueError("Self-play bounds must be positive")
    rng = np.random.default_rng(seed)
    records, results = [], Counter()
    for offset in range(0, games, parallel):
        boards = [chess.Board() for _ in range(min(parallel, games-offset))]
        histories = [[] for _ in boards]
        for ply in range(max_plies):
            active = [i for i, b in enumerate(boards)
                      if b.outcome(claim_draw=True) is None]
            if not active:
                break
            policies = search([boards[i] for i in active], infer, simulations,
                              rng=rng, noise=noise)
            for i, (legal, pi) in zip(active, policies):
                histories[i].append((encode_board(boards[i]),
                                     [j for _, j in legal], pi, boards[i].turn))
                choice = rng.choice(len(pi), p=pi) if ply < temperature_plies else pi.argmax()
                boards[i].push(legal[choice][0])
        for board, history in zip(boards, histories):
            outcome = board.outcome(claim_draw=True)
            results["truncated" if outcome is None else outcome.termination.name] += 1
            records.extend(targets(history, outcome))
        print(f"selfplay games={offset+len(boards)}/{games} positions={len(records)} "
              f"results={dict(results)}", flush=True)
    return records, dict(results)


def save_replay(path, records):
    if not records:
        raise ValueError("Empty replay")
    # Legal IDs and visit counts use CSR to avoid storing 1968 floats per position.
    np.savez_compressed(path,
        tokens=np.stack([r[0] for r in records]).astype(np.uint8),
        offsets=np.cumsum([0]+[len(r[1]) for r in records]).astype(np.int64),
        ids=np.concatenate([r[1] for r in records]).astype(np.int16),
        pi=np.concatenate([r[2] for r in records]).astype(np.float32),
        value=np.array([r[3] for r in records], np.float32),
        valid=np.array([r[4] for r in records], np.float32))


def load_replay(path):
    with np.load(path, allow_pickle=False) as data:
        records = []
        for i, token in enumerate(data["tokens"]):
            a, b = data["offsets"][i:i+2]
            ids, pi = data["ids"][a:b], data["pi"][a:b]
            if (len(token) != 69 or not len(ids) or len(set(ids.tolist())) != len(ids)
                    or np.any(ids < 0) or np.any(ids >= 1968)
                    or np.any(pi < 0) or not np.isfinite(pi).all()
                    or not np.isclose(pi.sum(), 1)
                    or data["valid"][i] not in (0, 1)
                    or not 0 <= data["value"][i] <= 1):
                raise ValueError("Invalid replay")
            from training.legal import mask_from_tokens
            if not mask_from_tokens(token)[ids].all():
                raise ValueError("Replay contains illegal actions")
            records.append((token.copy(), ids.copy(), pi.copy(),
                            float(data["value"][i]), float(data["valid"][i])))
    return records
