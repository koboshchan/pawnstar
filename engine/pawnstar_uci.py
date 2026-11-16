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


def probability(logit):
    """Value head emits logits for the canonical side to move, not White."""
    return float(1 / (1 + np.exp(-np.clip(logit, -40, 40))))


def value_for(logit, board, color):
    p = probability(logit)
    return p if board.turn == color else 1 - p


def ranked_moves(board, logits):
    legal = list(board.legal_moves)
    scores = np.array([logits[policy_id(m, board.turn == chess.BLACK)]
                       for m in legal], dtype=np.float64)
    if not np.all(np.isfinite(scores)):
        raise ValueError("non-finite legal policy")
    weights = np.exp(scores - scores.max())
    priors = weights / weights.sum()
    order = np.argsort(-scores, kind="stable")
    return [(legal[i], float(priors[i])) for i in order]


def terminal_value(board, color, draw_value):
    # Preserve move stacks when copying boards, otherwise repetition disappears.
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return None
    return draw_value if outcome.winner is None else float(outcome.winner == color)


class Player:
    def __init__(self, session, rerank=True, alpha=0.05, contempt=0.02):
        self.session = session
        self.rerank = rerank
        self.alpha = alpha
        self.contempt = contempt

    def evaluate(self, boards, value=True):
        names = ["policy", "value"] if value else ["policy"]
        outputs = self.session.run(names, {"tokens": np.concatenate(
            [tokens_for(b) for b in boards], axis=0)})
        policy = np.asarray(outputs[0])
        if policy.shape != (len(boards), len(MOVE_ID)):
            raise ValueError("unexpected policy shape")
        if not value:
            return policy, None
        values = np.asarray(outputs[1]).reshape(-1)
        if values.shape != (len(boards),) or not np.all(np.isfinite(values)):
            raise ValueError("invalid value head")
        return policy, values

    def move(self, board):
        if not any(board.legal_moves):
            return None
        # Mate scan covers ALL legal root moves, not just the policy top five.
        if self.rerank:
            for move in board.legal_moves:
                child = board.copy()
                child.push(move)
                if child.is_checkmate():
                    return move
        # Policy-only remains usable if the value head/batched search fails.
        try:
            policy, values = self.evaluate([board], value=self.rerank)
        except Exception:
            if not self.rerank:
                raise
            policy, _ = self.evaluate([board], value=False)
            values = None
        ranked = ranked_moves(board, policy[0])
        fallback = ranked[0][0]
        if not self.rerank or values is None:
            return fallback
        try:
            return self.search(board, ranked, values[0])
        except Exception as exc:
            print(f"reranker fallback: {exc}", file=sys.stderr, flush=True)
            return fallback

    def search(self, board, ranked, root_value):
        color = board.turn
        root_p = value_for(root_value, board, color)
        draw_value = 0.5 - (self.contempt if root_p >= 0.5 else -self.contempt)
        candidates = ranked[:5]
        scores = {}
        children, child_indices = [], []
        for i, (move, _) in enumerate(candidates):
            child = board.copy()
            child.push(move)
            score = terminal_value(child, color, draw_value)
            if score is not None:
                scores[i] = score
            else:
                children.append(child)
                child_indices.append(i)

        leaves, owners = [], []
        if children:
            policies, _ = self.evaluate(children, value=False)
            for i, child, policy in zip(child_indices, children, policies):
                replies = ranked_moves(child, policy)
                # A low-policy mate must never escape the two-reply shortlist.
                mated = False
                for reply, _ in replies:
                    leaf = child.copy()
                    leaf.push(reply)
                    if leaf.is_checkmate():
                        scores[i] = 0.0
                        mated = True
                        break
                if mated:
                    continue
                scores[i] = 1.0
                for reply, _ in replies[:2]:
                    leaf = child.copy()
                    leaf.push(reply)
                    score = terminal_value(leaf, color, draw_value)
                    if score is None:
                        leaves.append(leaf)
                        owners.append(i)
                    else:
                        scores[i] = min(scores[i], score)
        if leaves:
            _, values = self.evaluate(leaves)
            for i, leaf, logit in zip(owners, leaves, values):
                scores[i] = min(scores[i], value_for(logit, leaf, color))
        best = max(range(len(candidates)),
                   key=lambda i: scores[i] + self.alpha * candidates[i][1])
        return candidates[best][0]


def main():
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = int(os.environ.get("PAWNSTAR_THREADS", "2"))
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    model = os.environ.get("PAWNSTAR_MODEL", str(Path(__file__).resolve().parents[1] / "models" / "medium.onnx"))
    start = time.perf_counter()
    session = ort.InferenceSession(model, sess_options=opts, providers=["CPUExecutionProvider"])
    print(f"model_init_ms={(time.perf_counter()-start)*1000:.1f} providers={session.get_providers()}", file=sys.stderr, flush=True)
    uci(session)


def uci(session):
    board = chess.Board()
    player = Player(session)
    for line in sys.stdin:
        command = line.strip()
        try:
            if command == "uci":
                print("id name Pawnstar medium int8\nid author koboshchan\noption name ValueReranker type check default true\nuciok", flush=True)
            elif command.startswith("setoption name ValueReranker value "):
                player.rerank = command.split()[-1].lower() == "true"
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
                move = player.move(board)
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
