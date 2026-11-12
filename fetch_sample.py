"""Cache exactly N original export lines in a standalone zstd frame."""
import argparse, hashlib, io, pathlib, requests, zstandard
p=argparse.ArgumentParser()
p.add_argument("--rows",type=int,default=1000000)
p.add_argument("--output",default="original-1m.jsonl.zst")
p.add_argument("--url",default="https://database.lichess.org/lichess_db_eval.jsonl.zst")
a=p.parse_args()
out=pathlib.Path(a.output)
with requests.get(a.url,stream=True,timeout=(30,180)) as r:
    r.raise_for_status()
    with zstandard.ZstdDecompressor().stream_reader(r.raw) as dec, io.BufferedReader(dec) as reader, out.open("xb") as f:
        with zstandard.ZstdCompressor(level=1).stream_writer(f,closefd=False) as w:
            for i in range(a.rows):
                line=reader.readline()
                if not line: raise RuntimeError(f"early EOF at {i}")
                w.write(line)
print(out,hashlib.sha256(out.read_bytes()).hexdigest())
