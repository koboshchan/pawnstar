"""Compare array bytes, not ZIP container metadata. Legal masks optional."""
import argparse,json,pathlib,zipfile
import numpy as np
p=argparse.ArgumentParser()
p.add_argument("python",type=pathlib.Path)
p.add_argument("rust",type=pathlib.Path)
p.add_argument("--masks",action="store_true")
a=p.parse_args()
fields=["tokens","labels","keys","cp","mate","split"]+(["legal"] if a.masks else [])
left=sorted(a.python.glob("shard-*.npz")); right=sorted(a.rust.glob("shard-*.npz"))
assert len(left)==len(right)
rows=0
for x,y in zip(left,right):
    with np.load(x) as px,np.load(y) as rx,zipfile.ZipFile(x) as pz,zipfile.ZipFile(y) as rz:
        for name in fields:
            assert px[name].dtype==rx[name].dtype,(x,name,"dtype")
            assert px[name].shape==rx[name].shape,(x,name,"shape")
            assert px[name].tobytes()==rx[name].tobytes(),(x,name,"array bytes")
            assert pz.read(name+".npy")==rz.read(name+".npy"),(x,name,"NPY bytes")
        rows+=len(px["labels"])
for name in ["rows","accepted","rejected","train","val","test","shards"]:
    assert json.loads((a.python/"manifest.json").read_text())[name]==json.loads((a.rust/"manifest.json").read_text())[name],name
assert json.loads((a.python/"moves.json").read_text())==json.loads((a.rust/"moves.json").read_text())
print(json.dumps(dict(verified_rows=rows,shards=len(left),fields=fields,npy_bytes_identical=True)))
