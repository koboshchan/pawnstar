"""Paired-opening, color-swapped PURE legal-policy argmax evaluation."""
import chess
import numpy as np


def pure_moves(boards, infer):
    # Ignore value estimates. No search, reranking, sampling, or MCTS.
    return [legal[int(prior.argmax())][0] for legal, prior, _ in infer(boards)]


def paired_openings(pairs, seed=42, opening_plies=8):
    rng = np.random.default_rng(seed)
    result = []
    for _ in range(pairs):
        board = chess.Board()
        for _ in range(opening_plies):
            if board.outcome(claim_draw=True) is not None:
                break
            moves = list(board.legal_moves)
            board.push(moves[int(rng.integers(len(moves)))])
        result.append(board)
    return result


def evaluate(candidate, previous, pairs=16, max_plies=300, seed=42):
    if pairs < 1 or max_plies < 1:
        raise ValueError("Evaluation bounds must be positive")
    boards, colors = [], []
    for board in paired_openings(pairs, seed):
        boards.extend([board.copy(stack=True), board.copy(stack=True)])
        colors.extend([chess.WHITE, chess.BLACK])
    for _ in range(max_plies):
        moves = []
        for infer, is_candidate in ((candidate, True), (previous, False)):
            active = [i for i, b in enumerate(boards)
                      if b.outcome(claim_draw=True) is None
                      and (b.turn == colors[i]) == is_candidate]
            moves.extend(zip(active, pure_moves([boards[i] for i in active], infer)))
        if not moves:
            break
        for i, move in moves:
            boards[i].push(move)
    scores, truncated = [], 0
    for b, color in zip(boards, colors):
        outcome = b.outcome(claim_draw=True)
        if outcome is None:
            truncated += 1
            scores.append(None)
        else:
            scores.append(.5 if outcome.winner is None else float(outcome.winner == color))
    complete = [s for s in scores if s is not None]
    return dict(games=len(boards), wins=complete.count(1.), draws=complete.count(.5),
                losses=complete.count(0.), truncated=truncated,
                score=sum(complete)/len(complete) if complete else None,
                pair_scores=[scores[i:i+2] for i in range(0, len(scores), 2)],
                mode="pure-legal-argmax")
