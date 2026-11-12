"""Explicitly approved bucket upload only. Separate from preprocessing."""
import argparse,pathlib,subprocess,json,hashlib
p=argparse.ArgumentParser()
p.add_argument("output",type=pathlib.Path)
p.add_argument("destination",help="hf://buckets/koboshchan/BUCKET/PREFIX")
a=p.parse_args()
if not a.destination.startswith("hf://buckets/koboshchan/") or len(a.destination.split("/"))<7:
    p.error("Use a named bucket AND a dedicated prefix under koboshchan")
m=json.loads((a.output/"manifest.json").read_text())
if len(list(a.output.glob("shard-*.npz")))!=m["shards"]:
    p.error("Incomplete shard set")
print(f"Upload {m['accepted']:,} rows from {a.output} to {a.destination}")
if input("Type UPLOAD to approve this upload: ").strip()!="UPLOAD":
    raise SystemExit("Not uploaded.")
# Requires a saved HF login or HF_TOKEN from Colab Secrets.
subprocess.run(["hf","auth","whoami"],check=True)
with (a.output/"SHA256SUMS").open("w") as checks:
    for f in sorted(a.output.iterdir()):
        if not f.is_file() or f.name=="SHA256SUMS": continue
        h=hashlib.sha256()
        with f.open("rb") as stream:
            for block in iter(lambda:stream.read(16*1024**2),b""): h.update(block)
        checks.write(h.hexdigest()+"  "+f.name+"\n")
subprocess.run(["hf","buckets","sync",str(a.output),a.destination,"--no-delete"],check=True)
print("Upload finished. SHA256SUMS is included for download verification.")
