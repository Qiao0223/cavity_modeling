import numpy as np
import glob
import os

def load_seismic_from_npy(folder: str) -> np.ndarray:
    """
    从指定文件夹读取多个 .npy 文件并堆叠成 (C,Z,Y,X)
    """
    files = sorted(glob.glob(os.path.join(folder, "*.npy")))
    if not files:
        raise FileNotFoundError(f"在 {folder} 中没有找到 .npy 文件")

    arrays = []
    for f in files:
        arr = np.load(f, allow_pickle=True)  # 每个文件应是 (Z,Y,X)
        arrays.append(arr)

    # 检查形状一致性
    shapes = [arr.shape for arr in arrays]
    if not all(s == shapes[0] for s in shapes):
        raise ValueError(f"文件形状不一致: {shapes}")

    seismic = np.stack(arrays, axis=0)  # (C,Z,Y,X)
    return seismic

if __name__ == "__main__":
    # 使用示例
    seismic = load_seismic_from_npy(r"..\data\temp")
    print("最终形状:", seismic.shape)  # (C,Z,Y,X)
    np.save("../data/train/seismic.npy", seismic)
