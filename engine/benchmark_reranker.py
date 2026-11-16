"""Reproducible CPU fixed-ply arena. No online play, clocks, or GPU."""
import argparse
import hashlib
import json
import platform
import random
import sys
import time
from pathlib import Path

import chess
import chess.engine
import chess.pgn
import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pawnstar_uci import Player

OPENINGS = [
    "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6",
    "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4",
    "d2d4 d7d5 c2c4 e7e6 b1c3 g8f6",
    "d2d4 g8f6 c2c4 g7g6 b1c3 f8g7",
    "e2e4 e7e6 d2d4 d7d5 b1c3 g8f6",
    "e2e4 c7c6 d2d4 d7d5 b1c3 d5e4",
    "g1f3 d7d5 g2g3 g8f6 f1g2 e7e6",
    "c2c4 e7e5 b1c3 g8f6 g2g3 d7d5",
    "d2d4 d7d5 c2c4 c7c6 g1f3 g8f6",
    "e2e4 e7e5 f1c4 g8f6 d2d3 b8c6",
]


def session_for(model, threads):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(model, opts, providers=["CPUExecutionProvider"])


def opening(pair):
    # Paired, deterministic legal 10-ply openings, independently generated of players.
    rng = random.Random(106575 + pair)
    b = chess.Board()
    for m in OPENINGS[pair % len(OPENINGS)].split():
        b.push_uci(m)
    for _ in range(4):
        b.push(rng.choice(list(b.legal_moves)))
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/medium.onnx")
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--max-plies", type=int, default=120)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--stockfish", default="/usr/games/stockfish")
    ap.add_argument("--stockfish-games", type=int, default=8)
    ap.add_argument("--out", default="artifacts/reranker-arena")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    session = session_for(args.model, args.threads)
    players = {True: Player(session), False: Player(session, rerank=False)}
    timings = {}
    boards = [opening(i) for i in range(12)]
    for mode in [False, True]:
        players[mode].move(boards[0])  # warm up
        times = []
        for b in boards:
            start = time.perf_counter()
            players[mode].move(b)
            times.append((time.perf_counter() - start) * 1000)
        timings[str(mode)] = dict(mean_ms=float(np.mean(times)),
                                 median_ms=float(np.median(times)),
                                 p95_ms=float(np.percentile(times, 95)))
    manifest = dict(model_sha256=hashlib.sha256(Path(args.model).read_bytes()).hexdigest(),
                    python=platform.python_version(), ort=ort.__version__,
                    numpy=np.__version__, chess=chess.__version__,
                    threads=args.threads, max_plies=args.max_plies, latency=timings)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest), flush=True)
    records = []
    pgn_file = (out / "games.pgn").open("w")
    def play(game_id, sf=None, skill=None):
        b = opening(game_id // 2)
        root = b.copy()
        rerank_color = chess.WHITE if game_id % 2 == 0 else chess.BLACK
        moves = []
        for _ in range(args.max_plies):
            if b.is_game_over(claim_draw=True):
                break
            if b.turn == rerank_color:
                move = players[True].move(b)
            elif sf is not None:
                move = sf.play(b, chess.engine.Limit(nodes=1000)).move
            else:
                move = players[False].move(b)
            if move not in b.legal_moves:
                raise AssertionError(f"illegal move {move}")
            b.push(move)
            moves.append(move)
        outcome = b.outcome(claim_draw=True)
        capped = outcome is None
        result = outcome.result() if outcome else "1/2-1/2"
        points = .5 if capped or outcome.winner is None else float(outcome.winner == rerank_color)
        game = chess.pgn.Game()
        game.setup(root)
        game.headers.update(White="reranker" if rerank_color else ("argmax" if sf is None else f"sf{skill}"),
                            Black="reranker" if not rerank_color else ("argmax" if sf is None else f"sf{skill}"),
                            Result=result)
        node = game
        for m in moves:
            node = node.add_variation(m)
        print(game, file=pgn_file, end="\n\n", flush=True)
        record = dict(id=game_id, opponent="argmax" if sf is None else f"sf{skill}",
                      reranker_color="white" if rerank_color else "black",
                      result=result, points=points, plies=len(moves), capped=capped,
                      termination="move_cap" if capped else outcome.termination.name)
        records.append(record)
        with (out / "games.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
    for i in range(args.games):
        play(i)
    if args.stockfish_games and Path(args.stockfish).exists():
        with chess.engine.SimpleEngine.popen_uci(args.stockfish) as sf:
            sf.configure({"Threads": 1, "Hash": 16})
            for skill in [0, 5]:
                sf.configure({"Skill Level": skill})
                for i in range(args.stockfish_games):
                    play(i, sf, skill)
    summary = {}
    for opponent in sorted({r["opponent"] for r in records}):
        subset = [r for r in records if r["opponent"] == opponent]
        summary[opponent] = dict(games=len(subset), points=sum(r["points"] for r in subset),
                                wins=sum(r["points"] == 1 for r in subset),
                                draws=sum(r["points"] == .5 for r in subset),
                                losses=sum(r["points"] == 0 for r in subset),
                                capped=sum(r["capped"] for r in subset))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
