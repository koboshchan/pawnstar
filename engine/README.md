# Pawnstar UCI

A CPU-only orthodox UCI engine for the int8 ONNX model used by chess.kjt.lol.
Requires Python, python-chess, NumPy, and ONNX Runtime. Set PAWNSTAR_MODEL to
the model path and optionally PAWNSTAR_THREADS (default 2). The model is not
included in this repository.

Run `python engine/pawnstar_uci.py` through any UCI-compatible GUI or bot
client. Model startup and each move's inference time are written to stderr.

## Optional two-ply value reranker

The UCI check option `ValueReranker` defaults to true. To restore the original
legal-policy argmax, send `setoption name ValueReranker value false`.

The reranker takes the top five root moves by softmax over legal policy logits,
then the opponent's top two legal-policy replies per candidate. All legal moves
are checked for immediate mates at both levels, including moves outside those
shortlists. Children and leaves are evaluated in separate ONNX batches, so a
normal search makes three inference calls with batch sizes 1, 5, and up to 10.

Each root candidate receives its worst leaf value plus 0.05 times its root
legal-policy prior. The value head emits a logit, not a probability.
`preprocess/core.py:encode` flips White-relative cp and mate signs when
mirroring Black positions; `training/data.py:value_targets` uses sigmoid(cp/400).
Therefore sigmoid(value) is side-to-move-relative, and a two-ply leaf already
has our perspective for either root color. Exact mates override predicted
values. Draws, including claimable repetition and fifty-move draws, receive
0.48 when our root prediction is at least 0.5 and 0.52 when losing. Move history
is retained in search copies. Value/batch inference failures fall back to the
root policy argmax.

This is a narrow fixed-depth reranker, not a full chess search. It can miss
tactics beyond two plies or outside the policy shortlists. There is no
retraining or checkpoint change. Full ONNX graph optimization is enabled for
CPU inference.

## Tests and arena

Run `python -m unittest discover -s engine -p 'test_*.py'`.
Tests include the original 90-position website parity fixture, value sign and
Black mirroring, minimax rather than average leaf values, batch sizes, mate
detection outside shortlists, repetition history, contempt, UCI options, and
fallbacks. No model download is required for these tests.

Run `python engine/benchmark_reranker.py --model models/medium.onnx` for the
CPU arena. Defaults are 100 games against policy argmax, alternating colors
on 50 paired reproducible ten-ply openings, a 120-ply cap after the opening,
and eight additional games at each of Stockfish skill 0 and 5 (1000 nodes per
move). This is a fixed-move comparison, not a clock-enforced 1+0 match.
Capped games are separately labelled and counted as draws, not adjudicated
wins. PGNs, per-game results, environment/model fingerprint, latency statistics
and summary are saved in `artifacts/reranker-arena/`. Both Pawnstar modes use
the same model, provider, graph optimization and thread count.
