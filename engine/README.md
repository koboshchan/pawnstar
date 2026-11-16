# Pawnstar UCI

A CPU-only orthodox UCI engine for the int8 ONNX model used by chess.kjt.lol.
It scores legal moves with the policy head and plays the highest-logit move.
It does not search and never uses a GPU. Requires Python, python-chess, NumPy,
and ONNX Runtime. Set PAWNSTAR_MODEL to the model path and optionally
PAWNSTAR_THREADS (default 2). The model is not included in this repository.

Run `python engine/pawnstar_uci.py` through any UCI-compatible GUI or bot
client. Model startup and each move's inference time are written to stderr.
For tests, run `python -m unittest discover -s engine -p 'test_*.py'`.
The fixtures compare 90 positions, legal move IDs and argmax against the
website's Python-generated parity data.
