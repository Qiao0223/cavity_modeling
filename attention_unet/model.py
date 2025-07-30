# ================= model.py =================
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from monai.networks.nets import AttentionUnet
from monai.losses import DiceLoss

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
        self.dice_fn = DiceLoss(softmax=True, to_onehot_y=True, include_background=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        data, lbl, mask = batch
        loss = self.masked_loss(self(data), lbl, mask)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data, lbl, mask = batch
        loss = self.masked_loss(self(data), lbl, mask)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def masked_loss(self, pred, target, mask):
        # CrossEntropy with reduced influence from unlabeled (mask=0) regions
        ce = F.cross_entropy(pred, target, reduction='none')
        weights = torch.where(mask, torch.tensor(1.0, device=pred.device), torch.tensor(0.1, device=pred.device))
        ce = (ce * weights).sum() / (weights.sum() + 1e-6)
        # Dice loss on foreground only (needs channel dimension)
        target_unsq = target.unsqueeze(1)
        dice = self.dice_fn(pred, target_unsq)
        return ce + dice

class SeismicInferModel(SeismicSegModel):
    def predict_step(self, batch, batch_idx):
        data, coords = batch
        logits = self(data)
        probs = torch.softmax(logits, dim=1)
        return {"probs": probs.cpu(), "coords": coords.cpu()}