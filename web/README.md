# Pawnstar in the browser

Static Vite application with chess.js rules and ONNX Runtime Web. Inference stays on the device, in a worker. WebGPU is attempted first, then WASM on unsupported browsers or operator errors. Both paths use INT8 weights, never FP16. The quantized graph stores per-output-channel signed INT8 weights and standard DequantizeLinear + MatMul nodes; activations, embeddings and normalization remain FP32. This avoids MatMulNBits and dynamic activation quantization.

## Export your checkpoint

Run from the repository root using Python with torch, chess, numpy, onnx and onnxruntime installed.

```sh
python -m training.export_onnx artifacts/small-1k.pt --data artifacts/val --samples 1024
mkdir -p web/public/models
cp artifacts/export/pawnstar-int8.onnx web/public/models/
cp artifacts/export/parity.json web/test/parity.json
cd web
npm ci
npm test
npm run build
npm run dev -- --host 0.0.0.0 --port 18273
```

Download the development checkpoint with the authenticated Hugging Face bucket CLI from `hf://buckets/koboshchan/checkpoints/pawnstar/runs/small-1k/step-000001000.pt`. The export accepts any compatible small/medium/large checkpoint via its first argument. Validation data must be preprocessed NPZ shards, with split=1 positions. Report JSON contains unmasked vocabulary top-1 agreement, cross-entropy policy loss and exact model bytes. No checkpoint or model binaries are checked into git.

The Node tests check all 90 Python token/legal-ID fixtures and policy move selection. When an exported FP32 ONNX exists, an additional Node WASM test compares complete logits and legal argmax against Python on ten fixtures, tolerance 1e-4.

## Live medium checkpoint

Deployed medium-v1 step 106575 on November 14, 2026. On 512 split=1 positions, FP32 is 202,291,023 bytes with policy loss 1.55546653. Per-channel INT8 is 55,130,763 bytes with policy loss 1.55562055 and 98.4375% unmasked top-1 agreement (504/512). The 98% deployment gate passed. All 90 encoding/legal-ID fixtures and ten full-logit Node/WASM parity positions pass against this checkpoint.

The default URL is versioned as models/pawnstar-medium-106575-7a3587112b5c-int8.onnx, SHA256 7a3587112b5ce8f3559f74140aa0627d85afe184c71cba24cd426647836a888b. The old small model remains available at models/pawnstar-small-1k-int8.onnx and its original URL for rollback. No model binaries are committed.

## Measured development checkpoint

512 held-out positions from validation shard000000; not a full validation-set score.

| Format | Bytes | FP32 top-1 agreement | Policy loss |
| --- | ---: | ---: | ---: |
| FP32 | 45,800,714 | 100% | 5.86309719 |
| INT8 | 14,058,554 | 99.0234% | 5.86309242 |

INT8 is the only deployed format. The value bar is a sigmoid of the learned raw value logit, made white-relative. It is not a centipawn evaluation or a calibrated strength claim. The original 1,000-step development network is weak. Both networks use direct policy moves without search.

## Hosting

Deploy the contents of `dist/` to any HTTPS static host. Runtime assets are copied locally by prebuild, so no third-party CDN is required. Serve .wasm as application/wasm and .mjs as application/javascript. Serve index.html with no-cache, hashed assets as immutable, and model files with a short cache or versioned name. HTTPS is necessary for WebGPU outside localhost. WASM is single-threaded and needs no cross-origin-isolation headers.

The live site is https://chess.kjt.lol, served by the desktop's existing oneserver nginx from E:/oneserver/public/chess. Its model is same-origin at the versioned medium URL above, with CORS enabled. No inference backend, Docker restart, or DNS record change was needed. New route, backups and certificate renewal files are in E:/pawnstar-web. The existing proxy was syntax-tested and gracefully reloaded only.

The versioned medium network loads automatically on page open. A subtle loading indicator is shown and moves are disabled until ready. Temperature is fixed internally at 0.2, with no model or temperature settings exposed. The model URL is fixed in source, not accepted from query parameters.

## Interface

Click or drag legal moves, choose promotion, flip the board, select your side and start a new game. The move list and terminal states use chess.js. Temperature zero uses legal masked argmax; nonzero samples only legal moves. New game invalidates old in-flight results. Appearance follows the device's light/dark setting.
