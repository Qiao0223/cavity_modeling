import numpy as np
import torch

if __name__ == '__main__':
    print("PyTorch 版本:", torch.__version__)
    print("CUDA 是否可用:", torch.cuda.is_available())
    print("CUDA 版本:", torch.version.cuda)
    print("可用 GPU 数量:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("当前 GPU:", torch.cuda.get_device_name(0))

    arr = np.load(r"C:\Work\sunjie\Python\cavity_modeling\data\train\label.npy")

    # 先把 NaN 当成 0 处理
    valid_mask = ~np.isnan(arr)  # True 表示不是 NaN
    non_zero_mask = arr != 0  # True 表示不是 0

    has_valid_value = np.any(valid_mask & non_zero_mask)
    print("是否有非 0 且非 NaN 的元素：", has_valid_value)

