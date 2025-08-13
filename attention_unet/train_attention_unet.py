import torch
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from monai.networks.nets import AttentionUnet

# ========== 1. 自定义 Dataset ========== #
class SeismicDataset(Dataset):
    def __init__(self, seismic, label, patch_size=64, step=32):
        self.seismic = seismic  # shape=(C,Z,Y,X)
        self.label = label      # shape=(Z,Y,X)
        self.ps = patch_size
        self.step = step
        self.patches = self._make_patches()

    def _make_patches(self):
        C,Z,Y,X = self.seismic.shape
        patches = []
        for z in range(0, Z-self.ps, self.step):
            for y in range(0, Y-self.ps, self.step):
                for x in range(0, X-self.ps, self.step):
                    patches.append((z,y,x))
        return patches

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        z,y,x = self.patches[idx]
        data = self.seismic[:, z:z+self.ps, y:y+self.ps, x:x+self.ps]
        lbl = self.label[z:z+self.ps, y:y+self.ps, x:x+self.ps]
        mask = (lbl >= 0)  # 稀疏标签掩膜
        return torch.FloatTensor(data), torch.LongTensor(lbl), torch.BoolTensor(mask)


# ========== 2. 定义 Masked Loss ========== #
def masked_loss(pred, target, mask):
    loss = F.cross_entropy(pred, target, reduction="none")
    return (loss * mask.float()).sum() / mask.float().sum()


# ========== 3. Lightning 模型 ========== #
class SeismicSegModel(pl.LightningModule):
    def __init__(self, in_ch):
        super().__init__()
        self.model = AttentionUnet(
            spatial_dims=3,
            in_channels=in_ch,
            out_channels=2,
            channels=(32, 64, 128, 256),
            strides=(2, 2, 2, 2)
        )

    def training_step(self, batch, batch_idx):
        data, lbl, mask = batch
        pred = self.model(data)
        loss = masked_loss(pred, lbl, mask)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


# ========== 4. 训练入口 ========== #
if __name__ == "__main__":
    seismic = np.load("data/seismic.npy")  # shape=(C,Z,Y,X)
    label = np.load("data/label.npy")      # shape=(Z,Y,X)

    dataset = SeismicDataset(seismic, label, patch_size=64, step=32)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    model = SeismicSegModel(in_ch=seismic.shape[0])

    trainer = pl.Trainer(max_epochs=20, accelerator="gpu", devices=1)  # 如果有GPU
    trainer.fit(model, loader)

    # 训练完保存模型
    torch.save(model.model.state_dict(), "attention_unet3d.pth")