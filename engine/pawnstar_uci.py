#!/usr/bin/env python3
"""Pawnstar ONNX UCI engine, orthodox chess only."""
import os
import sys
import time
from pathlib import Path

import chess
import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preprocess.core import MOVE_ID


def tokens_for(board):
    canonical = board.mirror() if board.turn == chess.BLACK else board
    t = np.ones((1, 69), dtype=np.int64)
    for sq, piece in canonical.piece_map().items():
        t[0, sq] = 1 + piece.piece_type + (0 if piece.color else 6)
    for i, right in enumerate((
        canonical.has_kingside_castling_rights(chess.WHITE),
        canonical.has_queenside_castling_rights(chess.WHITE),
        canonical.has_kingside_castling_rights(chess.BLACK),
        canonical.has_queenside_castling_rights(chess.BLACK),
    )):
        t[0, 64+i] = 14 + int(right)
    t[0, 68] = 16 if canonical.ep_square is None else 17 + chess.square_file(canonical.ep_square)
    return t


def policy_id(move, black):
    if black:
        move = chess.Move(chess.square_mirror(move.from_square),
                          chess.square_mirror(move.to_square), move.promotion)
    return MOVE_ID[move.uci()]


def parse_position(command):
    parts = command.split()
    if len(parts) < 2:
        raise ValueError("position missing board")
    if parts[1] == "startpos":
        board, pos = chess.Board(), 2
    elif parts[1] == "fen":
        board, pos = chess.Board(" ".join(parts[2:8])), 8
    else:
        raise ValueError("unknown position")
    if pos < len(parts):
        if parts[pos] != "moves":
            raise ValueError("expected moves")
        for move in parts[pos+1:]:
            board.push_uci(move)
    return board


def main():
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = int(os.environ.get("PAWNSTAR_THREADS", "2"))
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    model = os.environ.get("PAWNSTAR_MODEL", str(Path(__file__).resolve().parents[1] / "models" / "medium.onnx"))
    start = time.perf_counter()
    session = ort.InferenceSession(model, sess_options=opts, providers=["CPUExecutionProvider"])
    print(f"model_init_ms={(time.perf_counter()-start)*1000:.1f} providers={session.get_providers()}", file=sys.stderr, flush=True)
    board = chess.Board()
    for line in sys.stdin:
        command = line.strip()
        try:
            if command == "uci":
                print("id name Pawnstar medium int8\nid author koboshchan\nuciok", flush=True)
            elif command == "isready":
                print("readyok", flush=True)
            elif command == "ucinewgame":
                board = chess.Board()
            elif command.startswith("position "):
                board = parse_position(command)
            elif command.startswith("go"):
                start = time.perf_counter()
                legal = list(board.legal_moves)
                if not legal:
                    print("bestmove 0000", flush=True)
                    continue
                logits = session.run(["policy"], {"tokens": tokens_for(board)})[0].reshape(-1)
                move = max(legal, key=lambda m: logits[policy_id(m, board.turn == chess.BLACK)])
                elapsed = (time.perf_counter() - start)*1000
                print(f"move_ms={elapsed:.1f}", file=sys.stderr, flush=True)
                print("bestmove " + move.uci(), flush=True)
            elif command == "quit":
                return
        except Exception as exc:
            print(f"engine error: {exc}", file=sys.stderr, flush=True)
            raise


if __name__ == "__main__":
    main()
