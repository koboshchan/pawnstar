import torch
from torch.utils.data import DataLoader
from training.data import Shards
from training.tests.test_training import make_shards
def test_initial_skip_workers(tmp_path):
    make_shards(tmp_path)
    import shutil
    for i in range(3,8):
        shutil.copyfile(tmp_path/"shard-0000.npz", tmp_path/f"shard-{i:04d}.npz")
    for workers in (0,2,4):
        kw=dict(batch_size=None,num_workers=workers)
        from itertools import islice
        full=list(islice(DataLoader(Shards(tmp_path,0,7,repeat=True),**kw),20))
        skip=4
        actual=list(islice(DataLoader(Shards(tmp_path,0,7,repeat=True,skip_batches=skip),**kw),8))
        for a,b in zip(actual,full[skip:skip+8]):
            assert all(torch.equal(x,y) for x,y in zip(a,b))
