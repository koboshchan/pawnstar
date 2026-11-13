"""Bounded-memory shuffled shards, partitioned between workers before loading."""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

def value_targets(cp, mate):
    # Win probability, not [-1,1]. Mate signs map to exact 0/1 endpoints.
    cp = np.asarray(cp, dtype=np.float32)
    mate = np.asarray(mate)
    valid = np.isfinite(cp) | (mate != 0)
    value = 1 / (1 + np.exp(-np.clip(np.nan_to_num(cp) / 400, -40, 40)))
    value = np.where(mate != 0, (mate > 0).astype(np.float32), value)
    return value.astype(np.float32), valid.astype(np.float32)

class Shards(IterableDataset):
    def __init__(self, root, split, batch_size, seed=42, repeat=False):
        self.paths = sorted(Path(root).rglob("*.npz"))
        if not self.paths:
            raise ValueError(f"No NPZ shards in {root}")
        self.split, self.batch_size = split, batch_size
        self.seed, self.repeat = seed, repeat

    def __iter__(self):
        worker = get_worker_info()
        wid, workers = (worker.id, worker.num_workers) if worker else (0, 1)
        epoch = 0
        while True:
            rng = np.random.default_rng(self.seed + epoch)
            paths = np.array(self.paths, dtype=object)
            if self.split == 0:
                rng.shuffle(paths)
            found = 0
            for path in paths[wid::workers]:
                with np.load(path, allow_pickle=False) as shard:
                    indices = np.flatnonzero(shard["split"] == self.split)
                    if not len(indices):
                        continue
                    tokens = shard["tokens"]
                    if tokens.shape[1] not in (69, 128):
                        raise ValueError(f"Unsupported token width in {path}")
                    if tokens.shape[1] == 128 and np.any(tokens[:, 69:]):
                        raise ValueError(f"Nonzero padding in {path}")
                    labels = shard["labels"]
                    if np.any(labels[indices] >= 1968):
                        raise ValueError(f"Invalid policy label in {path}")
                    cp, mate = shard["cp"], shard["mate"]
                    if self.split == 0:
                        rng.shuffle(indices)
                    for start in range(0, len(indices), self.batch_size):
                        ix = indices[start:start+self.batch_size]
                        target, valid = value_targets(cp[ix], mate[ix])
                        found += len(ix)
                        yield (torch.from_numpy(tokens[ix, :69].astype(np.int64)),
                               torch.from_numpy(labels[ix].astype(np.int64)),
                               torch.from_numpy(target), torch.from_numpy(valid))
            if not self.repeat or not found:
                return
            epoch += 1
