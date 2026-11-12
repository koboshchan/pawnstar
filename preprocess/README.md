# Chess policy preprocessing

Run with Python 3 and numpy, python-chess, requests, zstandard. Python reads the original Lichess JSONL export. Rust supports both JSONL and ordered Hugging Face parquet. The flattened data lacks an explicit PV index: preserve file order, never sort PVs by score. Source https://database.lichess.org/lichess_db_eval.jsonl.zst is mutable; download and sha256-pin the export before a reproducible full build.

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

    cd preprocess && PYTHONPATH=. ../.venv/bin/pytest -q

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
tokens and the same moves.json vocabulary when omitted. A training-time decoder
is not included in this preprocessing-only repository.
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


## Ordered Hugging Face parquet

From a Colab checkout, with CPU High-RAM selected:

    !git clone https://github.com/koboshchan/pawnstar
    !bash /content/pawnstar/preprocess/colab_preprocess_hf.sh

The script reuses existing files in `/content/lichess-hf/data/`, downloads only
missing shards with `hf download` and parallel Xet transfers, then verifies every
shard's size and SHA256 against the pinned Hub commit. It does not silently
replace local mismatched files. HF Hub v1+ uses Xet rather than the deprecated
hf_transfer switch. Validation reads existing files but does not redownload them.
It requires roughly 70 GB free for the unmasked output after input download.
Output defaults to `/content/chess-policy-hf-v1`; nothing is uploaded.

Default snapshot: `Lichess/chess-position-evaluations` at
`8c5285259fdbf3218bf2582de9775dd1378f520b`.
`CHESS_HF_INPUT`, `CHESS_HF_REVISION`, `CHESS_OUTPUT`, and `CHESS_ROWS` override
these defaults. A revision override must be a full commit SHA. The manifest
records `source_revision`, `input_format`, and ordered `source_files`; the Colab
script also saves `source.json` with authoritative per-file SHA256 checksums.
The raw Rust CLI requires a revision but trusts the caller's local files; use
`prepare_hf.py` to verify the bytes before a reproducible full build.

Manual command from `preprocess/` after compiling:

    rust/target/release/chess-preprocess --input /content/lichess-hf --input-format hf-parquet --revision 8c5285259fdbf3218bf2582de9775dd1378f520b --output /content/chess-policy-hf-v1 --vocab moves.json --threads 8 --no-masks

Input accepts a single parquet file, a directory of parquet files, or an HF
snapshot directory with a `data/` subdirectory. Shards are read in lexicographic
filename order. Rows belonging to one FEN must be contiguous in that order.
A FEN can span row groups, batches, and shard files. Only a strictly better
(depth, knodes) replaces the selected row, so equal evaluations preserve their
first PV in file order. The producer keeps one selected PV and a bounded queue;
it never groups the billion-row dataset in RAM. Repeated noncontiguous FENs
remain separate records, just as repeated JSONL records do. `--limit` counts
contiguous FEN groups, not flattened PV rows. Malformed parquet aborts rather
than silently rejecting source data.

### Validation performed

The original 1M-position sample and pinned HF revision have identical FEN order.
All 1,706,302 shared (depth, knodes) evaluations preserve their complete PV order
with zero discrepancies. But the source snapshots differ: 5,745 original eval
keys are absent from HF and 2,889 selected PVs/scores differ. These are snapshot
changes, not evidence of scrambled PV order. Comparing output against the newer
original export is intentionally not a parity requirement.

Same-snapshot test used the first 1,000,000 HF positions, 5,189,871 flattened
rows. They were re-sharded at arbitrary 500,003-row boundaries, including splits
inside FEN groups. HF parquet and reconstructed JSONL both accepted 994,251
positions and rejected 5,749 invalid boards. All six unmasked NPY members are
byte-identical across 100 output shards. Repeating with masks produced identical
seventh legal-mask members too. ZIP container metadata and manifest timing/source
fields differ by design. Local unmasked HF mode took 12.45 seconds on the same
sandbox, not a Colab benchmark. Full-dataset preprocessing has not been run.

Reproduce same-snapshot parity from `preprocess/` (install pyarrow and zstandard):

    python reconstruct_hf_sample.py /content/lichess-hf /content/hf-sample.jsonl.zst --limit 1000000
    rust/target/release/chess-preprocess --input /content/lichess-hf --input-format hf-parquet --revision 8c5285259fdbf3218bf2582de9775dd1378f520b --output /content/hf-parquet-check --vocab moves.json --limit 1000000
    rust/target/release/chess-preprocess --input /content/hf-sample.jsonl.zst --output /content/hf-jsonl-check --vocab moves.json
    python compare_outputs.py /content/hf-jsonl-check /content/hf-parquet-check

The regression suite additionally covers promotion, mover-relative zero and mate,
(depth,knodes) ties, first-PV preservation, cross-file groups, limits, optional
masks, illegal selected PV rejection, missing revisions, and corrupt input.
