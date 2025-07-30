# ================= predict.py =================
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from attention_unet.dataset import InferDataset
from attention_unet.model import SeismicInferModel

CKPT_PATH = "/home/zzz/cavity_modeling/attention_unet/log/version_2/checkpoints/best-model.ckpt"
SEISMIC_PATH = "/home/zzz/cavity_modeling/data/train/seismic_normalized.npy"
SAVE_PATH = "/home/zzz/cavity_modeling/data/train/prediction.npy"

PATCH_SIZE = (64, 64, 64)
STEP = 32

seismic_np = np.load(SEISMIC_PATH)
infer_ds = InferDataset(seismic_np, patch_size=PATCH_SIZE, step=STEP)
infer_dl = DataLoader(
    infer_ds, 
    batch_size=9, 
    shuffle=False, 
    num_workers=2, 
    pin_memory=True,
    prefetch_factor=2
    )

model = SeismicInferModel.load_from_checkpoint(CKPT_PATH)
trainer = Trainer(accelerator="gpu", devices=1, precision="16-mixed")
outputs = trainer.predict(model, dataloaders=infer_dl)

_, Z, Y, X = (1,) + seismic_np.shape if seismic_np.ndim == 3 else seismic_np.shape
num_classes = 2
prob_map = np.zeros((num_classes, Z, Y, X), np.float32)
count_map = np.zeros((Z, Y, X), np.int32)

for batch in outputs:
    probs, coords = batch["probs"].numpy(), batch["coords"].numpy()
    for i in range(probs.shape[0]):
        z, y, x = coords[i]
        prob_map[:, z:z+PATCH_SIZE[0], y:y+PATCH_SIZE[1], x:x+PATCH_SIZE[2]] += probs[i]
        count_map[z:z+PATCH_SIZE[0], y:y+PATCH_SIZE[1], x:x+PATCH_SIZE[2]] += 1

avg_probs = prob_map / np.maximum(count_map, 1)[None]
preds = np.argmax(avg_probs, axis=0).astype(np.uint8)
np.save(SAVE_PATH, preds)
print(f"Saved predicted segmentation to: {SAVE_PATH}")
