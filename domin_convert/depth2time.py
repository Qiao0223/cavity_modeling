import numpy as np
import numpy as np
from scipy.interpolate import interp1d
from concurrent.futures import ThreadPoolExecutor

def fill_velocity_zeros(velocity: np.ndarray, min_velocity: float) -> np.ndarray:
    """
    插值填充速度数组中的零值，并对过小速度按下限裁剪。
    """
    v = velocity.copy().astype(float)
    mask = v > 0
    if not mask.any():
        # 整条 trace 速度全为 0，返回下限值数组
        return np.full_like(v, min_velocity)
    idx = np.arange(len(v))
    # 边界插值
    if not mask[0]:
        first = np.where(mask)[0][0]
        v[:first] = v[first]
    if not mask[-1]:
        last = np.where(mask)[0][-1]
        v[last+1:] = v[last]
    # 中间插值
    v[~mask] = np.interp(idx[~mask], idx[mask], v[mask])
    # 下限裁剪
    return np.clip(v, min_velocity, None)


def process_trace(trace: np.ndarray,
                  vel_trace: np.ndarray,
                  K: int,
                  dz: float,
                  t_regular: np.ndarray,
                  v_thresh: float,
                  min_velocity: float) -> np.ndarray:
    """
    单道处理：若速度低于阈值则输出全零，否则填充速度零值、计算双程走时并插值。
    """
    if np.nanmin(vel_trace) < v_thresh:
        # 跳过该 trace，输出全零
        return np.zeros_like(t_regular)
    # 填充零值并裁剪下限速度
    v_clean = fill_velocity_zeros(vel_trace, min_velocity)
    # 等距深度双程走时
    t_seg = 2 * dz / v_clean
    t_at_depth = np.linspace(0.0, np.sum(t_seg), K)
    # 插值
    return interp1d(t_at_depth, trace,
                    kind='linear',
                    bounds_error=False,
                    fill_value=0.0)(t_regular)


def depth3d_to_time(data_3d: np.ndarray,
                    velocity_array: np.ndarray,
                    dz: float,
                    dt: float,
                    v_thresh: float = 500.0,
                    min_velocity: float = 1500.0,
                    n_workers: int = 1,
                    max_samples: int = 1_000_000) -> tuple[np.ndarray, np.ndarray]:
    """
    三维深度域数据转换为时间域数据，等距深度采样。

    参数:
    data_3d : (I, J, K) np.ndarray
        深度域地震数据
    velocity_array : (I, J, K) or (K,) np.ndarray
        速度模型
    dz : float
        深度采样间隔 (米)
    dt : float
        目标时间采样间隔 (秒)
    v_thresh : float
        速度阈值，低于该值的 trace 输出零
    min_velocity : float
        速度插值下限 (m/s)
    n_workers : int
        并行线程数
    max_samples : int
        最大时间采样数

    返回:
    data_time : (I, J, N) np.ndarray
    t_regular : (N,) np.ndarray
    """
    I, J, K = data_3d.shape
    # 计算最大双程走时
    if velocity_array.ndim == 3:
        max_t = max(2.0 * np.sum(dz / fill_velocity_zeros(velocity_array[i, j, :], min_velocity))
                    for i in range(I) for j in range(J)
                    if np.nanmin(velocity_array[i, j, :]) >= v_thresh)
    else:
        max_t = 2.0 * np.sum(dz / fill_velocity_zeros(velocity_array, min_velocity))
    # 样本数及时间轴
    n_samples = int(np.ceil(max_t / dt))
    if n_samples > max_samples:
        dt = max_t / max_samples
        n_samples = max_samples
    t_regular = np.linspace(0.0, max_t, n_samples, endpoint=False)

    # 并行或串行处理
    args = []
    for i in range(I):
        for j in range(J):
            vel = velocity_array[i, j, :] if velocity_array.ndim == 3 else velocity_array
            args.append((data_3d[i, j, :], vel, K, dz, t_regular, v_thresh, min_velocity))
    if n_workers > 1:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(lambda p: process_trace(*p), args))
    else:
        results = [process_trace(*p) for p in args]

    # 重构输出
    data_time = np.zeros((I, J, n_samples), dtype=data_3d.dtype)
    idx = 0
    for i in range(I):
        for j in range(J):
            data_time[i, j, :] = results[idx]
            idx += 1
    return data_time, t_regular

# 示例调用省略，确保主逻辑正确无误


if __name__ == '__main__':
    # 加载深度域数据和速度模型
    sies_npz = np.load(r'C:\Work\sunjie\Python\cavity_modeling\input_npy\yingxi_crop.npz', allow_pickle=True)
    seis = sies_npz['data']  # shape: (I, J, K)
    vel_npz = np.load(r'C:\Work\sunjie\Python\cavity_modeling\input_npy\yingxi_velocity_crop.npz', allow_pickle=True)
    velocity = vel_npz['data']  # shape: (I, J, K) or (K,)

    # 转换参数
    dz = 5.0      # 深度采样间隔 (米)
    dt = 0.01     # 时间采样间隔 (秒) 对应 100Hz
    workers = 10  # 并行线程数

    # 执行转换
    data_time, t_regular = depth3d_to_time(seis, velocity, dz, dt, workers)

    # 保存结果
    np.save(r'C:\Work\sunjie\converted_data_time.npy', data_time)

