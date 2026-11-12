"""Colab bounded-memory export downloader. Invoked by colab_preprocess.sh."""
import hashlib,json,os,pathlib,shutil,requests
src=os.environ["CHESS_SOURCE_URL"]; dest=pathlib.Path(os.environ["CHESS_EXPORT"])
out=pathlib.Path(os.environ["CHESS_OUTPUT"])
if out.exists(): raise SystemExit(f"Output already exists: {out}. Choose a fresh CHESS_OUTPUT.")
with requests.head(src,allow_redirects=True,timeout=60) as r:
    r.raise_for_status()
    size=int(r.headers.get("Content-Length",0))
    headers={k:r.headers[k] for k in ("ETag","Last-Modified","Content-Length") if k in r.headers}
limit=int(os.environ["CHESS_ROWS"])
positions=limit or 417862375
output_bytes=positions*145 + ((positions+99999)//100000)*4096
need=(0 if dest.exists() else size)+output_bytes+10*1024**3
free=shutil.disk_usage(dest.parent).free
print(f"Estimated required free {need/1e9:.1f} GB, available {free/1e9:.1f} GB",flush=True)
if free<need: raise SystemExit("Insufficient disk. Output estimate assumes the current 417,862,375-position export.")
if not dest.exists():
    tmp=dest.with_suffix(dest.suffix+".partial")
    with requests.get(src,stream=True,timeout=(30,300)) as r:
        r.raise_for_status()
        if headers.get("ETag") and r.headers.get("ETag")!=headers["ETag"]:
            raise SystemExit("Export changed between HEAD and GET. Retry with a pinned source.")
        downloaded=0; report=0
        with tmp.open("wb") as f:
            for block in r.iter_content(8*1024**2):
                f.write(block);downloaded+=len(block)
                if downloaded-report>=1024**3:
                    print(f"Downloaded {downloaded/1e9:.1f} / {size/1e9:.1f} GB",flush=True);report=downloaded
        if size and downloaded!=size: raise SystemExit("Download length mismatch")
    tmp.rename(dest)
h=hashlib.sha256()
with dest.open("rb") as f:
    for block in iter(lambda:f.read(16*1024**2),b""): h.update(block)
digest=h.hexdigest()
sidecar=dest.with_suffix(dest.suffix+".sha256")
if sidecar.exists() and sidecar.read_text().split()[0]!=digest:
    raise SystemExit("Existing export SHA256 changed")
sidecar.write_text(digest+"  "+dest.name+"\n")
dest.with_suffix(dest.suffix+".source.json").write_text(json.dumps(dict(url=src,headers=headers,sha256=digest),indent=2))
print("SHA256",digest,flush=True)
