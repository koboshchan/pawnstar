"""Searchless legal-move inference and minimal UCI protocol."""
import argparse
import sys
import chess
import numpy as np
import torch
from preprocess.core import encode, MOVE_ID
from training.model import PolicyValue, Config

def encode_board(board):
    moves = list(board.legal_moves)
    if not moves:
        raise ValueError("No legal moves")
    # Reuse the reference encoder, including literal EP and black mirroring.
    tokens, _, _, _, _, _ = encode(board.fen(en_passant="fen"), moves[0].uci())
    return tokens[:69]

def legal_ids(board):
    return [(move, MOVE_ID[(chess.Move(chess.square_mirror(move.from_square),
            chess.square_mirror(move.to_square), move.promotion) if
            board.turn == chess.BLACK else move).uci()])
            for move in board.legal_moves]

class Player:
    def __init__(self, checkpoint, device="cpu"):
        self.device = torch.device(device)
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.model = PolicyValue(Config(**ckpt["config"])).to(self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

    @torch.inference_mode()
    def move(self, board):
        legal = legal_ids(board)
        if not legal:
            return None
        tokens = torch.from_numpy(encode_board(board).astype(np.int64))[None].to(self.device)
        logits, _ = self.model(tokens)
        ids = torch.tensor([i for _, i in legal], device=self.device)
        return legal[logits[0, ids].argmax().item()][0]

def uci(player):
    board = chess.Board()
    for line in sys.stdin:
        parts = line.strip().split()
        if not parts:
            continue
        command = parts[0]
        if command == "uci":
            print("id name pawnstar\nid author koboshchan\nuciok", flush=True)
        elif command == "isready":
            print("readyok", flush=True)
        elif command == "ucinewgame":
            board = chess.Board()
        elif command == "position":
            try:
                m = parts.index("moves") if "moves" in parts else len(parts)
                board = chess.Board() if parts[1] == "startpos" else chess.Board(" ".join(parts[2:m]))
                for move in parts[m+1:]:
                    board.push_uci(move)
            except (ValueError, IndexError) as exc:
                print(f"info string invalid position {exc}", flush=True)
        elif command == "go":
            move = player.move(board)
            print(f"bestmove {move.uci() if move else '0000'}", flush=True)
        elif command == "quit":
            break

def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--device", default="cpu")
    p.add_argument("--fen")
    args = p.parse_args()
    player = Player(args.checkpoint, args.device)
    if args.fen:
        move = player.move(chess.Board(args.fen))
        print(move.uci() if move else "0000")
    else:
        uci(player)

if __name__ == "__main__":
    main()
