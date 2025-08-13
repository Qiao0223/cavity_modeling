import numpy as np
import numba
import os


def load_data(filename="seismic_data.npy"):
    """ 读取地震数据体 """
    return np.load(filename).astype(np.float32)


def compute_gradients(data):
    """ 计算三维梯度 """
    return np.gradient(data)


def compute_structure_tensor(I_x, I_y, I_z):
    """ 计算结构张量的元素 """
    S_xx = I_x * I_x
    S_yy = I_y * I_y
    S_zz = I_z * I_z
    S_xy = I_x * I_y
    S_xz = I_x * I_z
    S_yz = I_y * I_z
    return S_xx, S_yy, S_zz, S_xy, S_xz, S_yz


@numba.njit(parallel=True)
def compute_eigenvalues(S_xx, S_yy, S_zz, S_xy, S_xz, S_yz, lambda1, lambda2, lambda3):
    """ 并行计算每个体素的特征值 """
    X, Y, Z = S_xx.shape
    for x in numba.prange(X):  # 并行计算
        for y in range(Y):
            for z in range(Z):
                # 组装 3×3 结构张量
                S = np.array([
                    [S_xx[x, y, z], S_xy[x, y, z], S_xz[x, y, z]],
                    [S_xy[x, y, z], S_yy[x, y, z], S_yz[x, y, z]],
                    [S_xz[x, y, z], S_yz[x, y, z], S_zz[x, y, z]]
                ], dtype=np.float32)

                # 计算特征值
                eigenvalues = np.linalg.eigh(S)[0]  # 仅计算特征值

                # 存入数组
                lambda1[x, y, z] = eigenvalues[0]
                lambda2[x, y, z] = eigenvalues[1]
                lambda3[x, y, z] = eigenvalues[2]

if __name__ == "__main__":
    npz = np.load(r'..\data\input_npy\fuyuan3_crop.npz', allow_pickle=True)
    data = npz['data']

    print("计算梯度...")
    I_x, I_y, I_z = compute_gradients(data)

    print("计算结构张量...")
    S_xx, S_yy, S_zz, S_xy, S_xz, S_yz = compute_structure_tensor(I_x, I_y, I_z)

    # 获取数据体形状
    X, Y, Z = data.shape

    print(f"数据体大小: {X}x{Y}x{Z}, 开始分配 NumPy 数组存储特征值...")

    # **改进：使用 NumPy 数组而不是 memmap**
    lambda1 = np.zeros((X, Y, Z), dtype=np.float32)
    lambda2 = np.zeros((X, Y, Z), dtype=np.float32)
    lambda3 = np.zeros((X, Y, Z), dtype=np.float32)

    print("开始并行计算特征值...")
    compute_eigenvalues(S_xx, S_yy, S_zz, S_xy, S_xz, S_yz, lambda1, lambda2, lambda3)

    # **改进：用 np.save() 正确保存 `.npy`**
    np.save(r"../output_npy/lambda1.npy", lambda1)
    np.save(r"../output_npy/lambda2.npy", lambda2)
    np.save(r"../output_npy/lambda3.npy", lambda3)

    print("计算完成，特征值已保存为 lambda1.npy, lambda2.npy, lambda3.npy")
