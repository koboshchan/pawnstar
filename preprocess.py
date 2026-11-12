"""Stream original JSONL to bounded-memory NumPy shards; no flattened-PV guessing."""
import argparse,json,time,pathlib
import numpy as np
from core import encode,best_eval,VOCAB
def run(a):
    import zstandard as zstd
    import requests,io
    out=pathlib.Path(a.output);out.mkdir(parents=True,exist_ok=False)
    source=requests.get(a.input,stream=True,timeout=(30,180)) if a.input.startswith("https://") else None
    if source: source.raise_for_status()
    raw=source.raw if source else open(a.input,"rb")
    reader=io.TextIOWrapper(zstd.ZstdDecompressor().stream_reader(raw)) if a.input.endswith(".zst") else io.TextIOWrapper(raw)
    counts=dict(rows=0,accepted=0,rejected=0,train=0,val=0,test=0)
    buf=[];shards=0;start=time.time();errors={}
    def flush():
        nonlocal shards
        if not buf:return
        np.savez(out/f"shard-{shards:06d}.npz",tokens=np.stack([x[0] for x in buf]),labels=np.array([x[1] for x in buf],dtype=np.uint16),legal=np.stack([x[2] for x in buf]),keys=np.array([x[3] for x in buf],dtype=np.uint64),cp=np.array([x[4] if x[4] is not None else np.nan for x in buf],dtype=np.float32),mate=np.array([x[5] if x[5] is not None else 0 for x in buf],dtype=np.int16),split=np.array([0 if x[3]%1000<980 else 1 if x[3]%1000<990 else 2 for x in buf],dtype=np.uint8))
        shards+=1;buf.clear()
    try:
        for line in reader:
            if a.limit and counts["rows"]>=a.limit:break
            counts["rows"]+=1
            try:
                r=json.loads(line);ev,pv=best_eval(r)
                x=encode(r["fen"],pv["line"],pv.get("cp"),pv.get("mate"))
                buf.append(x);counts["accepted"]+=1
                k=x[3]%1000;counts["train" if k<980 else "val" if k<990 else "test"]+=1
            except (ValueError,KeyError,IndexError) as e:
                counts["rejected"]+=1;name=type(e).__name__+":"+str(e);errors[name]=errors.get(name,0)+1
            if len(buf)>=a.shard_size:flush()
            if counts["rows"]%100000==0:print(json.dumps(counts),flush=True)
        flush()
    finally:
        reader.close()
        if source:source.close()
    manifest=dict(**counts,errors=errors,seconds=time.time()-start,shards=shards,source=a.input,source_revision=a.revision,selection="max(depth,knodes,earliest eval); pvs[0]",score_pov="source White; output side to move",token_length=128,active_tokens=69,vocab_size=len(VOCAB),split="canonical FEN blake2b64 modulo1000: 980/10/10",schema_version=1)
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2))
    (out/"moves.json").write_text(json.dumps(VOCAB))
    print(json.dumps(manifest),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True);p.add_argument("--output",required=True)
    p.add_argument("--limit",type=int,default=0);p.add_argument("--shard-size",type=int,default=10000)
    p.add_argument("--revision",default="unversioned")
    run(p.parse_args())
