# ================= train.py =================
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from attention_unet.dataset import SeismicDataset, stratified_split
from attention_unet.model import SeismicSegModel

# 数据目录
data_dir = "/home/zzz/cavity_modeling/data/train"

# 读取标签并检查
label_np = np.load(os.path.join(data_dir, "label.npy"))
label_np = np.nan_to_num(label_np, nan=0)
if not (label_np > 0).any():
    raise ValueError("No foreground in labels. Training aborted.")

# 读取地震数据
seismic_np = np.load(os.path.join(data_dir, "seismic_normalized.npy"))

# 转为张量
seismic = torch.from_numpy(seismic_np).float()
label = torch.from_numpy(label_np).long()

# 数据集与采样
full_ds = SeismicDataset(seismic, label, patch_size=64, step=32)
train_idx, val_idx = stratified_split(full_ds.has_pos, train_ratio=0.8)
train_ds, val_ds = Subset(full_ds, train_idx), Subset(full_ds, val_idx)

train_dl = DataLoader(train_ds, batch_size=9, shuffle=True, num_workers=2, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size=9, shuffle=False, num_workers=2, pin_memory=True)

# 日志与回调
logger = TensorBoardLogger("attention_unet", name="log")
early_stop = EarlyStopping(monitor="val_loss", patience=5, mode="min")
checkpoint = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, filename="best-model")

# 训练
trainer = Trainer(
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
