import torch
import numpy as np
from torch.utils.data import Dataset

class SeismicDataset(Dataset):
    def __init__(self, seismic: torch.Tensor, label: torch.Tensor, patch_size: int = 64, step: int = 32):
        self.seismic = seismic
        self.label = label
        self.ps = patch_size
        self.step = step
        coords, has_pos = [], []
        _, Z, Y, X = seismic.shape
        for z in range(0, Z - self.ps + 1, self.step):
            for y in range(0, Y - self.ps + 1, self.step):
                for x in range(0, X - self.ps + 1, self.step):
                    patch_lbl = label[z:z+self.ps, y:y+self.ps, x:x+self.ps]
                    if (patch_lbl > 0).any():  # only keep patches with annotated foreground
                        coords.append((z, y, x))
                        has_pos.append(True)
        self.patches = coords
        self.has_pos = np.array(has_pos, dtype=bool)

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx: int):
        z, y, x = self.patches[idx]
        data = self.seismic[:, z:z+self.ps, y:y+self.ps, x:x+self.ps]
        lbl = self.label[z:z+self.ps, y:y+self.ps, x:x+self.ps]
        mask = lbl > 0
        return data, lbl.long(), mask

class InferDataset(Dataset):
    def __init__(self, seismic: np.ndarray, patch_size=(64,64,64), step=32):
        if seismic.ndim == 3:
            seismic = seismic[np.newaxis, ...]
        self.seismic = seismic
        self.patch_size = patch_size
        self.step = step
        self.coords = []
        _, Z, Y, X = seismic.shape
        # Correct sliding window over full volume dimensions
        for z in range(0, Z - patch_size[0] + 1, step):
            for y in range(0, Y - patch_size[1] + 1, step):
                for x in range(0, X - patch_size[2] + 1, step):
                    self.coords.append((z, y, x))

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        z, y, x = self.coords[idx]
        ps0, ps1, ps2 = self.patch_size
        patch = self.seismic[:, z:z+ps0, y:y+ps1, x:x+ps2]
        return torch.from_numpy(patch).float(), torch.tensor([z,y,x],dtype=torch.long)


def stratified_split(has_pos: np.ndarray, train_ratio: float = 0.8, seed: int = 42):
    idx = np.arange(len(has_pos))
    pos_idx = idx[has_pos]
    rng = np.random.RandomState(seed)
    rng.shuffle(pos_idx)
    n_pos = int(len(pos_idx) * train_ratio)
    train_idx = pos_idx[:n_pos]
    val_idx = pos_idx[n_pos:]
    return train_idx.tolist(), val_idx.tolist()