import os
import torch
import numpy as np
from monai.inferers import sliding_window_inference
from train_attention_unet import SeismicSegModel  # 根据你实际模块路径调整

def load_model_from_checkpoint(
    checkpoint_path: str,
    in_channels: int,
    device: torch.device
) -> torch.nn.Module:
    """
    使用 Lightning 提供的接口加载 SeismicSegModel，并提取其中的 UNet 子模型。
    """
    lightning_model = SeismicSegModel.load_from_checkpoint(
        checkpoint_path,
        in_channels=in_channels,
        map_location=device
    )
    unet = lightning_model.model.to(device)
    unet.eval()
    return unet

def predict_full_volume(
    model: torch.nn.Module,
    seismic_np: np.ndarray,
    patch_size: tuple = (64, 64, 64),
    overlap: float = 0.25,
    device: torch.device = torch.device('cuda')
) -> np.ndarray:
    """
    对整个地震数据体积进行滑窗推断，返回分割结果数组 (Z,Y,X)。
    """
    # 确保 (C, Z, Y, X)
    if seismic_np.ndim == 3:
        seismic_np = seismic_np[np.newaxis, ...]
    # 构造 (1, C, Z, Y, X)
    input_tensor = torch.from_numpy(seismic_np).unsqueeze(0).to(device)
    with torch.no_grad():
        output = sliding_window_inference(
            inputs=input_tensor,
            roi_size=patch_size,
            sw_batch_size=4,
            predictor=model,
            overlap=overlap,
        )
    # 输出 (1, 2, Z, Y, X) -> argmax -> (Z,Y,X)
    preds = torch.argmax(output, dim=1).cpu().numpy()[0]
    return preds

if __name__ == "__main__":
    ckpt      = "/home/zzz/cavity_modeling/attention_unet/log/version_0/checkpoints/best-model.ckpt"
    data_path = "/home/zzz/cavity_modeling/data/train/seismic_normalized.npy"
    save_path = "/home/zzz/cavity_modeling/prediction.npy"
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seismic_np = np.load(data_path)
    in_channels = seismic_np.ndim == 4 and seismic_np.shape[0] or 1

    # **1.** 正确调用加载函数
    model = load_model_from_checkpoint(ckpt, in_channels, device)

    # **2.** 显式传 device
    preds = predict_full_volume(model, seismic_np, device=device)

    np.save(save_path, preds)
    print(f"Saved predicted segmentation to: {save_path}")
