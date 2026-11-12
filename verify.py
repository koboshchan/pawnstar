import json,pathlib,sys
import numpy as np
p=pathlib.Path(sys.argv[1]);m=json.loads((p/"manifest.json").read_text())
n=0
for f in sorted(p.glob("shard-*.npz")):
    with np.load(f) as d:
        size=len(d["labels"]);n+=size
        assert d["tokens"].shape==(size,128)
        assert not d["tokens"][:,69:].any()
        assert (d["labels"]<1968).all()
        masks=np.unpackbits(d["legal"],axis=1,bitorder="little")
        assert masks[np.arange(size),d["labels"]].all()
        keys=d["keys"]%1000
        expected=np.where(keys<980,0,np.where(keys<990,1,2))
        assert np.array_equal(d["split"],expected)
assert n==m["accepted"]
assert m["rows"]==m["accepted"]+m["rejected"]
print(json.dumps({"verified":n,"shards":m["shards"],"bytes":sum(f.stat().st_size for f in p.glob("*"))}))
