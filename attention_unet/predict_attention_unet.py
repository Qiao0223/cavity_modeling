import os
import torch
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, Subset
from monai.networks.nets import AttentionUnet
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

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
                    coords.append((z, y, x))
                    patch_lbl = label[z:z+self.ps, y:y+self.ps, x:x+self.ps]
                    has_pos.append((patch_lbl > 0).any().item())
        self.patches = coords
        self.has_pos = np.array(has_pos, dtype=bool)

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx: int):
        z, y, x = self.patches[idx]
        data = self.seismic[:, z:z+self.ps, y:y+self.ps, x:x+self.ps]
        lbl = self.label[z:z+self.ps, y:y+self.ps, x:x+self.ps]
        mask = lbl >= 0
        return data, lbl.long(), mask


def stratified_split(has_pos: np.ndarray, train_ratio: float = 0.8, seed: int = 42):
    idx = np.arange(len(has_pos))
    pos_idx, neg_idx = idx[has_pos], idx[~has_pos]
    rng = np.random.RandomState(seed)
    rng.shuffle(pos_idx); rng.shuffle(neg_idx)
    n_pos = int(len(pos_idx) * train_ratio)
    n_neg = int(len(neg_idx) * train_ratio)
    train_idx = np.concatenate((pos_idx[:n_pos], neg_idx[:n_neg]))
    val_idx = np.concatenate((pos_idx[n_pos:], neg_idx[n_neg:]))
    rng.shuffle(train_idx); rng.shuffle(val_idx)
    return train_idx.tolist(), val_idx.tolist()


def masked_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss = F.cross_entropy(pred, target, reduction="none")
    m = mask.float()
    tot = m.sum()
    return torch.tensor(0.0, device=pred.device) if tot == 0 else (loss * m).sum() / tot

class SeismicSegModel(pl.LightningModule):
    def __init__(self, in_channels: int, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = AttentionUnet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=2,
            channels=(32, 64, 128, 256),
            strides=(2, 2, 2, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        data, lbl, mask = batch
        loss = masked_loss(self(data), lbl, mask)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data, lbl, mask = batch
        loss = masked_loss(self(data), lbl, mask)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


class SeismicInferModel(SeismicSegModel):
    def predict_step(self, batch, batch_idx):
        data, coords = batch
        logits = self(data)
        probs = torch.softmax(logits, dim=1)
        return {"probs": probs.cpu(), "coords": coords.cpu()}


if __name__ == "__main__":
    data_dir = "/home/zzz/cavity_modeling/data/train"
    seismic_np = np.load(os.path.join(data_dir, "seismic_normalized.npy"))
    label_np = np.load(os.path.join(data_dir, "label.npy"))

    seismic = torch.from_numpy(seismic_np).float()
    label = torch.from_numpy(label_np).long()

    full_ds = SeismicDataset(seismic, label, patch_size=64, step=32)
    train_idx, val_idx = stratified_split(full_ds.has_pos, train_ratio=0.8)
    train_ds, val_ds = Subset(full_ds, train_idx), Subset(full_ds, val_idx)

    train_dl = DataLoader(train_ds, batch_size=9, shuffle=True, num_workers=2, prefetch_factor=2, persistent_workers=True, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=9, shuffle=False, num_workers=2, prefetch_factor=2, persistent_workers=True, pin_memory=True)

    logger = TensorBoardLogger("attention_unet", name="log")
    early_stop = EarlyStopping(monitor="val_loss", patience=5, mode="min")
    checkpoint = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, filename="best-model")

    trainer = pl.Trainer(
        max_epochs=100,
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        logger=logger,
        callbacks=[early_stop, checkpoint]
    )

    trainer.fit(
        SeismicSegModel(in_channels=seismic.shape[0]),
        train_dataloaders=train_dl,
        val_dataloaders=val_dl
    )

    print(f"Best model saved at: {checkpoint.best_model_path}")
    print(f"Best val_loss: {checkpoint.best_model_score}")
