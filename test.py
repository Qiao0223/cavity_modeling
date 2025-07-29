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

import pytorch_lightning as pl
print(pl.__version__) 

