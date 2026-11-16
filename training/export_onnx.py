"""Export browser-compatible attention and benchmark weight-only quantization.

Checkpoints are pickle containers: only export your own trusted checkpoints.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import onnx
from onnx import numpy_helper, helper
import onnxruntime as ort
import torch
from torch import nn
from torch.nn import functional as F
from training.model import PolicyValue, Config
from training.data import Shards
from training.play import encode_board, legal_ids
import chess

class Portable(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, tokens):
        m = self.model
        x = m.token(tokens.long()) + m.position
        for b in m.blocks:
            z = b.norm1(x)
            q, k, v = F.linear(z, b.self_attn.in_proj_weight, b.self_attn.in_proj_bias).chunk(3, -1)
            heads = b.self_attn.num_heads
            width = m.config.width // heads
            q, k, v = [t.reshape(-1, 69, heads, width).transpose(1, 2) for t in (q, k, v)]
            a = torch.softmax((q @ k.transpose(-1, -2)) / width**.5, -1)
            z = (a @ v).transpose(1, 2).reshape(-1, 69, m.config.width)
            x = x + b.self_attn.out_proj(z)
            z = b.norm2(x)
            x = x + b.linear2(F.gelu(b.linear1(z)))
        x = m.norm(x).mean(1)
        return m.policy(x), m.value(x).squeeze(-1)

def int8_weights(src, dst):
    """Per-output-channel symmetric INT8 weights, float activations.

    Standard DequantizeLinear + MatMul, no MatMulNBits or activation quantization.
    Embeddings and normalization stay FP32.
    """
    graph = onnx.load(src)
    weights = {w.name: w for w in graph.graph.initializer}
    added, nodes, removed = [], [], set()
    for node in graph.graph.node:
        if node.op_type == 'MatMul' and node.input[1] in weights:
            name = node.input[1]
            w = numpy_helper.to_array(weights[name])
            if w.ndim == 2:
                scale = np.maximum(np.max(np.abs(w), axis=0)/127, 1e-12).astype(np.float32)
                quant = np.clip(np.rint(w/scale), -127, 127).astype(np.int8)
                if name not in removed:
                    added += [numpy_helper.from_array(quant, name+'_q'),
                              numpy_helper.from_array(scale, name+'_scale'),
                              numpy_helper.from_array(np.zeros(w.shape[1], np.int8), name+'_zero')]
                    nodes.append(helper.make_node('DequantizeLinear',
                        [name+'_q', name+'_scale', name+'_zero'], [name+'_dq'], axis=1))
                    removed.add(name)
                node.input[1] = name+'_dq'
        nodes.append(node)
    keep = [w for w in graph.graph.initializer if w.name not in removed]
    del graph.graph.initializer[:]
    graph.graph.initializer.extend(keep+added)
    del graph.graph.node[:]
    graph.graph.node.extend(nodes)
    onnx.checker.check_model(graph)
    onnx.save(graph, dst)

def session(path):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    return ort.InferenceSession(str(path), sess_options=opts, providers=['CPUExecutionProvider'])

def main():
    p = argparse.ArgumentParser()
    p.add_argument('checkpoint', type=Path)
    p.add_argument('--output', type=Path, default=Path('artifacts/export'))
    p.add_argument('--data', type=Path, required=True, help='NPZ shards, split=1 only')
    p.add_argument('--samples', type=int, default=1024)
    a = p.parse_args()
    torch.set_num_threads(2)
    a.output.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    model = PolicyValue(Config(**ckpt['config'])).eval()
    model.load_state_dict(ckpt['model'])
    portable = Portable(model).eval()
    rows, labels = [], []
    for x, y, _, _ in Shards(a.data, 1, 32):
        rows.extend(x.numpy()); labels.extend(y.numpy())
        if len(rows) >= a.samples:
            break
    x = np.asarray(rows[:a.samples], dtype=np.int64)
    y = np.asarray(labels[:a.samples], dtype=np.int64)
    if not len(x):
        raise ValueError('No validation positions')
    with torch.no_grad():
        expected = model(torch.from_numpy(x[:4]))[0]
        actual = portable(torch.from_numpy(x[:4]))[0]
        torch.testing.assert_close(expected, actual, atol=2e-5, rtol=2e-5)
    fp = a.output/'pawnstar-fp32.onnx'
    with torch.no_grad():
        torch.onnx.export(portable, (torch.from_numpy(x[:1]),), str(fp),
            input_names=['tokens'], output_names=['policy', 'value'], opset_version=18,
            dynamic_axes={'tokens': {0:'batch'}, 'policy': {0:'batch'}, 'value': {0:'batch'}},
            dynamo=False)
    int8_weights(str(fp), str(a.output/'pawnstar-int8.onnx'))
    results = {}
    baseline = None
    for kind in ('fp32', 'int8'):
        path = a.output/f'pawnstar-{kind}.onnx'
        sess = session(path)
        pred = np.concatenate([sess.run(['policy'], {'tokens':x[i:i+16]})[0]
                               for i in range(0, len(x), 16)])
        if baseline is None:
            baseline = pred.argmax(1)
            with torch.no_grad():
                torch_logits = portable(torch.from_numpy(x[:4]))[0].numpy()
            np.testing.assert_allclose(pred[:4], torch_logits, atol=3e-5, rtol=3e-5)
        maximum = pred.max(1)
        loss = np.mean(np.log(np.exp(pred-maximum[:,None]).sum(1)) + maximum-pred[np.arange(len(y)),y])
        results[kind] = dict(bytes=path.stat().st_size, agreement=float(np.mean(pred.argmax(1)==baseline)),
                            val_policy=float(loss), samples=len(x))
    default = 'int8'
    report = dict(checkpoint=str(a.checkpoint), default=default, results=results)
    (a.output/'report.json').write_text(json.dumps(report, indent=2))
    (a.output/'config.json').write_text(json.dumps({'model':f'pawnstar-{default}.onnx'}))
    # Mixed turns, castling, literal uncapturable EP, EP capture, underpromotion.
    boards = [chess.Board(), chess.Board('r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1'),
              chess.Board('r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1'),
              chess.Board('4k3/P7/8/8/8/8/7p/4K3 w - - 0 1'),
              chess.Board('4k3/P7/8/8/8/8/7p/4K3 b - - 0 1'),
              chess.Board('4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1')]
    b = chess.Board()
    for uci in ('e2e4','c7c5','g1f3','d7d5'):
        b.push_uci(uci); boards.append(b.copy())
    rng = np.random.default_rng(42)
    b = chess.Board()
    for i in range(80):
        if b.is_game_over(): b = chess.Board()
        b.push(list(b.legal_moves)[int(rng.integers(b.legal_moves.count()))])
        boards.append(b.copy())
    fixtures = []
    for b in boards:
        tokens = encode_board(b)
        with torch.no_grad(): logits = model(torch.from_numpy(tokens.astype(np.int64))[None])[0][0].numpy()
        legal = legal_ids(b)
        fixtures.append(dict(fen=b.fen(en_passant='fen'), tokens=tokens.tolist(), logits=logits.tolist(),
            legal=[dict(uci=m.uci(), id=i) for m,i in legal],
            argmax=max(legal, key=lambda pair: logits[pair[1]])[0].uci()))
    (a.output/'parity.json').write_text(json.dumps(fixtures))
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
