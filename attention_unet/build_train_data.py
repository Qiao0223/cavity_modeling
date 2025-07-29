import numpy as np
import glob
import os
from tqdm import tqdm

def remove_invalid_values(arr: np.ndarray) -> np.ndarray:
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

def normalize_array(arr: np.ndarray, mode="none"):
    arr = remove_invalid_values(arr).astype(np.float32)

    if mode == "none":
        return arr

    elif mode == "minmax":
        min_v, max_v = arr.min(), arr.max()
        arr = (arr - min_v) / (max_v - min_v + 1e-8)

    elif mode == "clip_minmax":
        low, high = np.percentile(arr, [1, 99])
        arr = np.clip(arr, low, high)
        arr = (arr - low) / (high - low + 1e-8)

    elif mode == "zscore":
        arr = (arr - arr.mean()) / (arr.std() + 1e-8)

    elif mode == "zscore_clip":
        arr = (arr - arr.mean()) / (arr.std() + 1e-8)
        arr = np.clip(arr, -3, 3)

    elif mode == "robust":
        median = np.median(arr)
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        arr = (arr - median) / (iqr + 1e-8)

    elif mode == "robust_minmax":
        median = np.median(arr)
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        arr = (arr - median) / (iqr + 1e-8)
        min_v, max_v = arr.min(), arr.max()
        arr = (arr - min_v) / (max_v - min_v + 1e-8)

    elif mode == "log_minmax":
        arr = np.log1p(np.abs(arr)) * np.sign(arr)
        min_v, max_v = arr.min(), arr.max()
        arr = (arr - min_v) / (max_v - min_v + 1e-8)

    else:
        raise ValueError(f"未知归一化模式: {mode}")

    return arr.astype(np.float32)  # 确保结果是 float32

def normalize_seismic(seismic: np.ndarray, mode="none", per_channel=True) -> np.ndarray:
    if mode == "none":
        return seismic.astype(np.float32)

    if per_channel:
        out = []
        for c in tqdm(range(seismic.shape[0]), desc="归一化中", unit="通道"):
            out.append(normalize_array(seismic[c], mode))
        return np.stack(out, axis=0).astype(np.float32)
    else:
        return normalize_array(seismic, mode).astype(np.float32)

def load_seismic_from_npy(folder: str, normalize_mode="none", per_channel=True) -> np.ndarray:
    files = sorted(glob.glob(os.path.join(folder, "*.npy")))
    if not files:
        raise FileNotFoundError(f"在 {folder} 中没有找到 .npy 文件")

    arrays = []
    print(f"共找到 {len(files)} 个文件，开始读取...")
    for f in tqdm(files, desc="读取文件", unit="文件"):
        try:
            arr = np.load(f, allow_pickle=True).astype(np.float32)  # 读取时就转 float32
            arr = remove_invalid_values(arr)

            if arr.ndim != 3:
                print(f"跳过文件 {f}，形状不是 (Z,Y,X)")
                continue

            arrays.append(arr)

        except Exception as e:
            print(f"文件 {f} 读取失败: {e}")
            continue

    if not arrays:
        raise ValueError("没有有效文件可用")

    shapes = [a.shape for a in arrays]
    common_shape = shapes[0]
    valid_arrays = [a for a in arrays if a.shape == common_shape]

    if len(valid_arrays) < len(arrays):
        print(f"有 {len(arrays)-len(valid_arrays)} 个文件形状不一致，已跳过")

    seismic = np.stack(valid_arrays, axis=0).astype(np.float32)  # 堆叠时也保持 float32
    seismic = normalize_seismic(seismic, mode=normalize_mode, per_channel=per_channel)
    return seismic.astype(np.float32)  # 返回最终 float32

if __name__ == "__main__":
    folder = r"/home/zzz/cavity_modeling/data/temp"

    seismic = load_seismic_from_npy(
        folder,
        normalize_mode="robust_minmax",  # 先鲁棒归一化，再 Min-Max
        per_channel=True
    )

    print("最终形状:", seismic.shape, "dtype:", seismic.dtype)  # 检查类型
    np.save("/home/zzz/cavity_modeling/data/train/seismic_normalized.npy", seismic.astype(np.float32))
