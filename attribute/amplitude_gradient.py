import numpy as np

def amplitude_gradient(seismic_data: np.ndarray,
                               dx: float = 1.0,
                               dy: float = 1.0,
                               dt: float = 1.0) -> np.ndarray:
    """
    计算三维地震数据的振幅梯度幅值。

    参数:
        seismic_data: 3D numpy 数组，形状为 (inline, xline, time)
        dx: inline 方向采样间隔
        dy: xline 方向采样间隔
        dt: time 方向采样间隔

    返回:
        G: 梯度幅值体
    """
    if seismic_data.ndim != 3:
        raise ValueError("输入数据必须是三维数组 (inline, xline, time)")

    Gx = np.gradient(seismic_data, dx, axis=0)
    Gy = np.gradient(seismic_data, dy, axis=1)
    Gt = np.gradient(seismic_data, dt, axis=2)

    G = np.sqrt(Gx**2 + Gy**2 + Gt**2)
    return G

if __name__ == "__main__":
    npz = np.load(r'..\input_npy\yingxi_crop.npz', allow_pickle=True)
    data = npz['data']
    dx, dy, dt = 1.0, 1.0, 1.0  # 根据实际采样间隔修改
    G = amplitude_gradient(data, dx, dy, dt)
    np.save("amplitude_gradient.npy", G)
