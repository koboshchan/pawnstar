"""Deterministic policy/value tests, no model download required."""
import sys
import io
import unittest
from pathlib import Path
from unittest.mock import patch
import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pawnstar_uci import (Player, tokens_for, policy_id, value_for,
                          terminal_value, parse_position, uci)
from preprocess.core import MOVE_ID, encode


class FakeSession:
    def __init__(self, fn):
        self.fn = fn
        self.calls = []

    def run(self, names, inputs):
        tokens = inputs["tokens"]
        self.calls.append((tuple(names), len(tokens)))
        pairs = [self.fn(row) for row in tokens]
        policy = np.stack([p for p, _ in pairs])
        value = np.array([v for _, v in pairs])
        return [policy if name == "policy" else value for name in names]


def logits(board, choices=()):
    p = np.full(len(MOVE_ID), -30.0)
    for j, uci in enumerate(choices):
        p[policy_id(chess.Move.from_uci(uci), board.turn == chess.BLACK)] = 5 - j
    return p


class RerankerTests(unittest.TestCase):
    def test_training_perspective_and_mirror(self):
        for board in (chess.Board(), parse_position("position startpos moves e2e4")):
            move = next(iter(board.legal_moves))
            reference = encode(board.fen(en_passant="fen"), move.uci(), cp=400)
            np.testing.assert_array_equal(tokens_for(board)[0], reference[0][:69])
            self.assertEqual(reference[4], 400 if board.turn else -400)
            self.assertAlmostEqual(value_for(np.log(3), board, board.turn), .75)
            self.assertAlmostEqual(value_for(np.log(3), board, not board.turn), .25)

    def test_mate_outside_top_five_both_colors(self):
        board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
        for b in (board, board.mirror()):
            session = FakeSession(lambda _: (np.zeros(len(MOVE_ID)), 0))
            chosen = Player(session).move(b)
            b.push(chosen)
            self.assertTrue(b.is_checkmate())
            self.assertEqual(session.calls, [])

    def test_batched_worst_case_not_average_and_black_sign(self):
        for board in (chess.Board(), chess.Board().mirror()):
            # Canonical tokens and priors must behave identically for Black.
            root_moves = [m for m in board.legal_moves][:5]
            root_policy = logits(board, [m.uci() for m in root_moves])
            table = {tokens_for(board).tobytes(): (root_policy, 0.)}
            for i, move in enumerate(root_moves):
                child = board.copy()
                child.push(move)
                replies = list(child.legal_moves)[:2]
                table[tokens_for(child).tobytes()] = (
                    logits(child, [r.uci() for r in replies]), 0.)
                # Candidate zero has a tempting average, but disastrous worst leaf.
                vals = ([.99, .1] if i == 0 else [.7, .7] if i == 1 else [.3, .3])
                for reply, v in zip(replies, vals):
                    leaf = child.copy()
                    leaf.push(reply)
                    table[tokens_for(leaf).tobytes()] = (
                        np.zeros(len(MOVE_ID)), np.log(v / (1-v)))
            session = FakeSession(lambda row: table[row[None].tobytes()])
            original = board.fen()
            self.assertEqual(Player(session).move(board), root_moves[1])
            self.assertEqual(board.fen(), original)
            self.assertEqual([n for _, n in session.calls], [1, 5, 10])
            self.assertEqual(Player(session, rerank=False).move(board), root_moves[0])

    def test_draw_and_repetition_history(self):
        b = parse_position("position startpos moves g1f3 g8f6 f3g1 f6g8 g1f3 g8f6 f3g1 f6g8")
        self.assertEqual(terminal_value(b.copy(), b.turn, .48), .48)
        self.assertEqual(terminal_value(b.copy(), b.turn, .52), .52)
        stalemate = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
        self.assertEqual(terminal_value(stalemate, chess.WHITE, .48), .48)

    def test_contempt_flips_when_losing(self):
        board = chess.Board()
        ranked = [(m, .2) for m in list(board.legal_moves)[:5]]
        player = Player(None)
        seen = []
        def terminal(b, color, draw):
            seen.append(draw)
            return draw
        with patch("pawnstar_uci.terminal_value", terminal):
            player.search(board, ranked, 2.)
            self.assertTrue(all(v == .48 for v in seen))
            seen.clear()
            player.search(board, ranked, -2.)
            self.assertTrue(all(v == .52 for v in seen))

    def test_opponent_mate_outside_top_two(self):
        # White's f2f3 permits Qh4 mate, even if Black policy ranks it last.
        board = parse_position("position startpos moves g2g4 e7e5")
        moves = [chess.Move.from_uci(x) for x in ["f2f3", "b1c3", "d2d3", "e2e3", "h2h3"]]
        root_policy = logits(board, [m.uci() for m in moves])
        table = {tokens_for(board).tobytes(): (root_policy, 0)}
        for m in moves:
            child = board.copy()
            child.push(m)
            table[tokens_for(child).tobytes()] = (
                logits(child, [r.uci() for r in list(child.legal_moves)[:2]]), 0)
        session = FakeSession(lambda row: table.get(row[None].tobytes(),
                            (np.zeros(len(MOVE_ID)), 2.)))
        self.assertNotEqual(Player(session).move(board), moves[0])

    def test_uci_option_switches_modes(self):
        b = chess.Board()
        session = FakeSession(lambda _: (logits(b, ["e2e4"]), 0))
        commands = "uci\nisready\nsetoption name ValueReranker value false\nposition startpos\ngo\nquit\n"
        output = io.StringIO()
        with patch("sys.stdin", io.StringIO(commands)), patch("sys.stdout", output):
            uci(session)
        text = output.getvalue()
        self.assertIn("option name ValueReranker type check default true", text)
        self.assertIn("readyok", text)
        self.assertIn("bestmove e2e4", text)
        self.assertEqual(session.calls, [(("policy",), 1)])

    def test_batch_exception_falls_back(self):
        b = chess.Board()
        session = FakeSession(lambda _: (logits(b, ["e2e4"]), 0))
        player = Player(session)
        with patch.object(player, "search", side_effect=RuntimeError("batch unavailable")):
            self.assertEqual(player.move(b).uci(), "e2e4")

    def test_fallback_and_no_legal(self):
        b = chess.Board()
        p = logits(b, ["e2e4"])
        session = FakeSession(lambda _: (p, float("nan")))
        self.assertEqual(Player(session).move(b).uci(), "e2e4")
        self.assertIsNone(Player(session).move(chess.Board(
            "7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")))


if __name__ == "__main__":
    unittest.main()
