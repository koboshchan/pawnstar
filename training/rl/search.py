"""Small PUCT reference search with batched leaves across independent roots.

All values are side-to-move win probabilities, including draws at 0.5.
Uses python-chess for all legality, repetition, castling and promotion rules.
"""
from dataclasses import dataclass, field
import math
import chess
import numpy as np
import torch
from training.play import encode_board, legal_ids


def terminal_value(board):
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return None
    return .5 if outcome.winner is None else float(outcome.winner == board.turn)


@dataclass
class Node:
    prior: float = 1.
    visits: int = 0
    total: float = 0.
    children: dict = field(default_factory=dict)

    def select(self, c_puct):
        # A child's mean is in the opponent's perspective.
        scale = math.sqrt(max(1, self.visits))
        return max(self.children.items(), key=lambda item:
                   (1-item[1].total/item[1].visits if item[1].visits else .5)
                   + c_puct*item[1].prior*scale/(1+item[1].visits))


def backup(path, value):
    for node in reversed(path):
        node.visits += 1
        node.total += value
        value = 1-value


class Network:
    def __init__(self, model, device="cpu", batch_size=128):
        self.model = model
        self.device = torch.device(device)
        self.batch_size = batch_size

    @torch.inference_mode()
    def __call__(self, boards):
        if not boards:
            return []
        self.model.eval()
        result = []
        for start in range(0, len(boards), self.batch_size):
            chunk = boards[start:start+self.batch_size]
            tokens = torch.as_tensor(np.stack([encode_board(b) for b in chunk]),
                                     dtype=torch.long, device=self.device)
            with torch.autocast(self.device.type, dtype=torch.bfloat16,
                                enabled=self.device.type == "cuda"):
                logits, values = self.model(tokens)
            values = values.float().sigmoid().cpu().numpy()
            for board, row, value in zip(chunk, logits.float(), values):
                legal = legal_ids(board)
                ids = [i for _, i in legal]
                prior = row[ids].softmax(0).cpu().numpy().astype(np.float64)
                prior /= prior.sum()
                result.append((legal, prior, float(value)))
        return result


def expand(node, prediction):
    legal, prior, value = prediction
    node.children = {move: Node(float(p)) for (move, _), p in zip(legal, prior)}
    return value


def search(boards, infer, simulations=32, c_puct=1.5, rng=None, noise=.25):
    """One leaf per root per simulation; no search is used in evaluation."""
    if simulations < 1 or c_puct <= 0 or not 0 <= noise <= 1:
        raise ValueError("Invalid search budget or exploration")
    if any(terminal_value(b) is not None for b in boards):
        raise ValueError("Search roots must be nonterminal")
    roots = [Node() for _ in boards]
    for root, pred in zip(roots, infer(boards)):
        expand(root, pred)
        if noise:
            if rng is None:
                raise ValueError("Root noise needs an explicit RNG")
            eta = rng.dirichlet(np.full(len(root.children), .3))
            for child, e in zip(root.children.values(), eta):
                child.prior = (1-noise)*child.prior + noise*float(e)
    for _ in range(simulations):
        pending = []
        for original, root in zip(boards, roots):
            board, node, path = original.copy(stack=True), root, [root]
            value = terminal_value(board)
            while value is None and node.children:
                move, node = node.select(c_puct)
                board.push(move)
                path.append(node)
                value = terminal_value(board)
            if value is not None:
                backup(path, value)
            else:
                pending.append((board, node, path))
        for (_, node, path), pred in zip(pending, infer([p[0] for p in pending])):
            backup(path, expand(node, pred))
    result = []
    for board, root in zip(boards, roots):
        legal = legal_ids(board)
        counts = np.array([root.children[m].visits for m, _ in legal], dtype=np.float64)
        counts /= counts.sum()
        result.append((legal, counts))
    return result
