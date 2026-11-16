import json
import unittest
from pathlib import Path
import chess
from pawnstar_uci import tokens_for, policy_id, parse_position

ROOT = Path(__file__).resolve().parents[1]

class Tests(unittest.TestCase):
    def test_website_parity(self):
        fixtures = json.loads((ROOT / "web/test/parity.json").read_text())
        for f in fixtures:
            board = chess.Board(f["fen"])
            self.assertEqual(tokens_for(board)[0].tolist(), f["tokens"], f["fen"])
            legal = {m.uci(): policy_id(m, board.turn == chess.BLACK) for m in board.legal_moves}
            self.assertEqual(legal, {x["uci"]: x["id"] for x in f["legal"]}, f["fen"])
            best = max(legal, key=lambda m: f["logits"][legal[m]])
            self.assertEqual(best, f["argmax"], f["fen"])

    def test_position(self):
        expected = chess.Board()
        for move in ("e2e4", "e7e5", "g1f3"):
            expected.push_uci(move)
        self.assertEqual(parse_position("position startpos moves e2e4 e7e5 g1f3").fen(), expected.fen())
        self.assertEqual(parse_position("position fen " + expected.fen(en_passant="fen")).fen(), expected.fen())

if __name__ == "__main__":
    unittest.main()
