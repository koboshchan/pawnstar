"""Legal masks for canonical white-to-move 69-token positions."""
import chess
import numpy as np
from preprocess.core import MOVE_ID


def mask_from_tokens(tokens):
    if len(tokens) < 69 or np.any((tokens[:64] < 1) | (tokens[:64] > 13)):
        raise ValueError("Invalid board tokens")
    if np.any((tokens[64:68] < 14) | (tokens[64:68] > 15)) or not 16 <= tokens[68] <= 24:
        raise ValueError("Invalid state tokens")
    board = chess.Board(None)
    for square, token in enumerate(tokens[:64]):
        token = int(token)
        if token != 1:
            board.set_piece_at(square, chess.Piece((token-2) % 6+1, token < 8))
    board.turn = chess.WHITE
    for token, square in zip(tokens[64:68], (chess.H1, chess.A1, chess.H8, chess.A8)):
        if token == 15:
            board.castling_rights |= chess.BB_SQUARES[square]
    board.ep_square = None if tokens[68] == 16 else chess.square(int(tokens[68])-17, 5)
    if not board.is_valid():
        raise ValueError("Invalid canonical board")
    mask = np.zeros(1968, dtype=np.bool_)
    for move in board.legal_moves:
        mask[MOVE_ID[move.uci()]] = True
    return mask


def legal_masks(tokens, labels, packed=None):
    if packed is not None:
        packed = np.asarray(packed)
        if packed.dtype != np.uint8 or packed.shape != (len(tokens), 246):
            raise ValueError("Invalid packed legal masks")
        masks = np.unpackbits(packed, axis=1, bitorder="little").astype(np.bool_)
    else:
        masks = np.stack([mask_from_tokens(row) for row in tokens])
    if not masks[np.arange(len(labels)), labels].all():
        raise ValueError("Policy target missing from legal mask")
    return masks
