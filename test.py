import numpy as np
import torch

if __name__ == '__main__':
    print("PyTorch 版本:", torch.__version__)
    print("CUDA 是否可用:", torch.cuda.is_available())
    print("CUDA 版本:", torch.version.cuda)
    print("可用 GPU 数量:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("当前 GPU:", torch.cuda.get_device_name(0))

    # npz = np.load(r"C:\Work\sunjie\Python\cavity_modeling\data\train\label.npy")
    # print(npz.shape)

    arr = np.load(r"/home/zzz/cavity_modeling/data/train/label.npy")  # 读取 npy 文件

    # 方法 1：直接判断
    is_all_zero = np.all(arr == 0)
    print(is_all_zero)

