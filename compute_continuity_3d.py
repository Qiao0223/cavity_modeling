#!/usr/bin/env python3
"""
compute_continuity_3d.py

计算三维地震数据的连续度热度体，输出与输入同样形状的 npy 文件。
参数直接在 main() 中设定。
"""
import numpy as np
from skimage.filters import threshold_otsu
from skimage.morphology import closing, opening, footprint_rectangle
from scipy.ndimage import grey_dilation, grey_erosion, convolve, gaussian_filter
from tqdm import tqdm


def compute_continuity_3d(
    seismic: np.ndarray,
    threshold_factor: float = 1.0,
    spatial_close: tuple = (30, 30),
    spatial_open: tuple = (30, 30),
    temporal_dilate: int = 3,
    temporal_erode: int = 3,
    continuity_window: tuple = (50, 50),
    temporal_sigma: float = 1.0
) -> np.ndarray:
    """
    计算三维地震体的连续度热度图（归一化到 [0,1]），并显示进度。

    返回：归一化到 [0,1] 的连续度体，形状 (I, X, T)。
    """
    if seismic.ndim != 3:
        raise ValueError(f"Expect 3D array, got {seismic.ndim}D.")

    # 1. 绝对值并归一化到 [0,1]
    amp = np.abs(seismic.astype(np.float32))
    max_amp = float(np.max(amp)) if amp.size > 0 else 1.0
    max_amp = max_amp or 1.0
    normed = amp / max_amp

    # 2. 全局 Otsu 阈值分割
    try:
        th0 = threshold_otsu(normed.flatten())
    except ValueError:
        th0 = 0.5
    th = th0 * threshold_factor
    bin3d = (normed > th).astype(np.uint8)

    # 3. 混合形态学处理
    selem_close = footprint_rectangle(spatial_close)
    selem_open = footprint_rectangle(spatial_open)
    I, X, T = bin3d.shape
    proc = np.empty_like(bin3d)

    # 3.2 空间闭运算逐切片
    for k in tqdm(range(T), desc="Spatial closing"):
        proc[:, :, k] = closing(bin3d[:, :, k], footprint=selem_close)
    # 3.3 时间轴膨胀
    proc = grey_dilation(proc, size=(1, 1, temporal_dilate))
    # 3.4 空间开运算逐切片
    for k in tqdm(range(T), desc="Spatial opening"):
        proc[:, :, k] = opening(proc[:, :, k], footprint=selem_open)
    # 3.5 时间轴腐蚀
    proc = grey_erosion(proc, size=(1, 1, temporal_erode))

    # 4. 空间滑窗卷积计算连续度
    Wi, Wx = continuity_window
    kernel = np.ones((Wi, Wx, 1), dtype=np.float32)
    origin = (-(Wi // 2), -(Wx // 2), 0)
    continuity = convolve(proc.astype(np.float32), kernel,
                         mode='constant', cval=0.0, origin=origin)

    # 5. 归一化并沿时间轴高斯平滑
    continuity_norm = continuity / float(kernel.sum())
    continuity_smooth = gaussian_filter(
        continuity_norm, sigma=(0, 0, temporal_sigma))

    return continuity_smooth

def main():
    # ---- 用户参数区 ----
    input_npy = 'numpy/YingXi_crop.npy'
    output_npy = 'continuity3d.npy'
    threshold_factor = 1.0
    spatial_close = (30, 30)
    spatial_open = (30, 30)
    temporal_dilate = 3
    temporal_erode = 3
    continuity_window = (50, 50)
    temporal_sigma = 1.0

    seismic3d = np.load(input_npy)
    print("Start computing continuity 3D volume...")
    cont3d = compute_continuity_3d(
        seismic3d,
        threshold_factor=threshold_factor,
        spatial_close=spatial_close,
        spatial_open=spatial_open,
        temporal_dilate=temporal_dilate,
        temporal_erode=temporal_erode,
        continuity_window=continuity_window,
        temporal_sigma=temporal_sigma
    )
    np.save(output_npy, cont3d)
    print(f"Done! Saved continuity volume to {output_npy}")


if __name__ == '__main__':
    main()
