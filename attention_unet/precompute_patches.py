import numpy as np
import torch
from torch.utils.data import TensorDataset
from numpy.lib.stride_tricks import sliding_window_view


def precompute_and_save_vectorized(
    seismic_path: str,
    label_path: str,
    patch_size: int,
    step: int,
    output_path: str,
):
    """
    使用 numpy 的 sliding_window_view 高效地将 3D 数据切成 patch 并保存为 TensorDataset。
    边缘不足部分使用 0 填充，并同时生成 mask (label >= 0)。
    """
    # 载入数据
    seismic = np.load(seismic_path)  # shape: (C, Z, Y, X)
    label = np.load(label_path)      # shape: (Z, Y, X)
    C, Z, Y, X = seismic.shape

    # 计算 padding 大小
    def calc_pad(length):
        if length < patch_size:
            return patch_size - length
        rem = (length - patch_size) % step
        return (step - rem) if rem != 0 else 0

    pad_z = calc_pad(Z)
    pad_y = calc_pad(Y)
    pad_x = calc_pad(X)

    # 0 填充
    seismic_padded = np.pad(
        seismic,
        ((0, 0), (0, pad_z), (0, pad_y), (0, pad_x)),
        mode="constant",
        constant_values=0,
    )
    label_padded = np.pad(
        label,
        ((0, pad_z), (0, pad_y), (0, pad_x)),
        mode="constant",
        constant_values=0,
    )

    Zp, Yp, Xp = Z + pad_z, Y + pad_y, X + pad_x

    # 使用 sliding_window_view 提取所有可能的 patch 窗口
    windows = sliding_window_view(
        seismic_padded,
        window_shape=(patch_size, patch_size, patch_size),
        axis=(1, 2, 3)
    )
    # windows.shape = (C, Zp-ps+1, Yp-ps+1, Xp-ps+1, ps, ps, ps)

    # 按 step 抽取
    windows = windows[:, ::step, ::step, ::step, ...]

    # 重排维度到 (N, C, ps, ps, ps)
    nZ, nY, nX = windows.shape[1:4]
    data = windows.transpose(1, 2, 3, 0, 4, 5, 6)
    data = data.reshape(-1, C, patch_size, patch_size, patch_size)

    # 同样切 label
    label_windows = sliding_window_view(
        label_padded,
        window_shape=(patch_size, patch_size, patch_size)
    )
    label_windows = label_windows[::step, ::step, ::step]
    labels = label_windows.reshape(-1, patch_size, patch_size, patch_size)

    # 生成 mask (label >= 0)
    masks = (labels >= 0)

    # 转为 torch Tensor 并保存
    data_tensor  = torch.from_numpy(data).contiguous().float()
    label_tensor = torch.from_numpy(labels).contiguous().long()
    mask_tensor  = torch.from_numpy(masks).contiguous().bool()

    dataset = TensorDataset(data_tensor, label_tensor, mask_tensor)
    torch.save(dataset, output_path)

    total = data_tensor.size(0)
    print(f"Saved {total} patches → {output_path}")
    print(f"Original shape: (C={C}, Z={Z},Y={Y},X={X}), padded to (Z={Zp},Y={Yp},X={Xp})")


if __name__ == '__main__':
    # 参数设置
    precompute_and_save_vectorized(
        seismic_path='/home/zzz/cavity_modeling/data/train/seismic_normalized.npy',
        label_path='/home/zzz/cavity_modeling/data/train/label.npy',
        patch_size=64,
        step=32,
        output_path='dataset.pt'
    )
