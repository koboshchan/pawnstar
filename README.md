# Chess policy preprocessing

Run with Python 3 and numpy, python-chess, requests, zstandard. This implementation reads the ORIGINAL Lichess JSONL export, not the flattened Hugging Face parquet. The flattened data lacks an explicit PV index, so it cannot reliably recover pvs[0] by score sorting. Source https://database.lichess.org/lichess_db_eval.jsonl.zst is mutable; download and sha256-pin the export before a reproducible full build.

Select maximum-depth evaluation, with maximum knodes then first occurrence as deterministic ties. Preserve its first principal variation. Invalid orthodox boards, empty PVs and illegal first moves are rejected, not replaced by weaker evaluations.

Source cp and mate are White-relative, documented at https://lichess.org/api/openapi.yaml. Output scores are mover-relative. Standard chess only. python-chess normalizes king-to-rook castling before black rank mirroring and color swapping. Output move vocabulary is 1968 sorted UCI strings including both promotion directions.

Each shard has uint8 tokens [N,128], 69 active positions, uint16 policy labels, 246-byte little-bit-order packed legal masks, canonical-position uint64 hashes, float32 cp with NaN for absent, int16 mate with zero for absent, and uint8 split (0 train,1 validation,2 test). Piece tokens 1 empty,2..7 white P..K,8..13 black P..K; 14/15 castling false/true; 16 no EP,17..24 EP files. Padding token 0. Squares a1..h8. Castling white king/queen then black king/queen. Only black-to-move positions are mirrored.

Canonical-position hashes assign 98/1/1 splits; duplicate positions cannot cross splits, but this streaming version does not globally deduplicate repeated original records. FEN lacks move counters/history so repetition and fifty-move state are unavailable.

Sample command

    .venv/bin/python preprocess.py --input https://database.lichess.org/lichess_db_eval.jsonl.zst --output sample-original-1m --limit 1000000

Full command, NOT executed

    sha256sum lichess_db_eval.jsonl.zst
    .venv/bin/python preprocess.py --input lichess_db_eval.jsonl.zst --output full --revision SHA256_FROM_ABOVE
    .venv/bin/python verify.py full

Tests

    PYTHONPATH=. .venv/bin/pytest -q

## Rust streaming preprocessor

The Rust executable uses shakmaty, rayon, serde_json and zstd. Ordered parallel
batches preserve input order and Python shard boundaries. It writes uncompressed
NPZ containers containing byte-identical NPY members. ZIP metadata and manifest
performance/source fields are intentionally not byte-identical.

    cargo build --release --locked --manifest-path rust/Cargo.toml
    rust/target/release/chess-preprocess --input original-1m.jsonl.zst --output rust-sample --threads 8 --no-masks
    python compare_outputs.py sample-original-1m rust-sample
    python verify.py rust-sample

Masks are omitted by default; --no-masks makes that explicit. Use --masks to
include the exact Python legal mask. Training must reconstruct legal masks from
tokens using core.decode_board and the same moves.json vocabulary when omitted.
The labels are still checked for legality during preprocessing. Input supports
local .zst, local JSONL, or uncompressed JSONL on stdin (--input -). Download HTTPS
sources before running. Batches default to 20,000 records; output shards default
to 10,000 accepted rows. Both are configurable. Output directories must be fresh.

Actual original-export 1,000,000-row parity test accepted 994,251 and rejected
5,749 invalid boards. All six unmasked NPY members match Python byte-for-byte
across 100 shards, including float NaN and signed-zero bits. The optional seventh
legal-mask member also matches across the entire million-row sample. A separate
seeded regression corpus tests legal playouts, depth/knodes/earliest ties, first
PV, both castling representations, underpromotion, literal EP, mate/CP signs,
signed zero, Python validity differences and invalid selected PV rejection.

Measured local release throughput, including zstd parsing and NPZ writing, was
295,982 input rows/s without masks (3.379 s), 204,316 with masks (4.894 s). These
are sandbox measurements with eight rayon threads on a 3-vCPU-exposed AMD host,
not a Colab eight-core benchmark. The cached million-line compressed sample has
SHA256 1ddc9c2f61de02dea2423110c899dddb08b941d794c8245fe8977b8bab1d1a1a.

The unmasked sample occupies 144,315,773 bytes versus 388,923,725 with masks.
Unmasked rows use 145 bytes, not 69: the on-disk context retains all 128 tokens.
At 417,862,375 input rows this is at most about 60.6 GB of array data, plus small
shard overhead. The current compressed source is about 22.5 GB. A 225 GB Colab
disk comfortably fits both without expanding the JSONL; batches stay bounded
well below 50 GB RAM. No full dataset run has been performed.

## Colab CPU High-RAM

Copy this repository to /content/chess-transformer, select CPU High-RAM, then run
this notebook cell. No repository remote is configured, so this does not assume
a published GitHub URL.

    !bash /content/chess-transformer/colab_preprocess.sh

The script installs Rust, builds release with Cargo.lock, downloads the compressed
export, records SHA256/HTTP provenance, checks disk space, processes on eight
threads with no masks, and verifies shards. It never uploads automatically.
CHESS_EXPORT, CHESS_OUTPUT, CHESS_ROWS and CHESS_SOURCE_URL are optional environment
overrides. CHESS_ROWS=0 means the complete export. For a quick 1M run set it to
1000000, but the script still downloads the complete compressed export.

After explicit approval, install huggingface_hub>=2.4.0, authenticate through
hf auth login or Colab Secrets/HF_TOKEN, and run the separate uploader with a
real existing bucket and dedicated prefix. koboshchan is the namespace, not a
bucket name. The uploader additionally asks for UPLOAD, writes SHA256SUMS, and
syncs without deleting any remote files.

    python -m pip install 'huggingface_hub>=2.4.0'
    python upload_shards.py /content/chess-policy-v1 hf://buckets/koboshchan/BUCKET/chess-policy-v1

The Rust crate is GPL-3.0-or-later because shakmaty is GPL-3.0-or-later. Lichess
export data is CC0. No Colab runtime was allocated and no Hugging Face upload
was performed while implementing this port.
