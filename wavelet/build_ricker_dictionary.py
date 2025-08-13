import numpy as np
import pywt  # 仍然需要用它来计算中心频率以保持一致性
import matplotlib.pyplot as plt
from sklearn.linear_model import OrthogonalMatchingPursuit
from tqdm import tqdm

def ricker_wavelet_formula(t, f):
    """
    根据数学公式生成Ricker子波。

    Args:
        t (np.ndarray): 时间轴。
        f (float): 子波的中心频率 (Hz)。

    Returns:
        np.ndarray: Ricker子波的波形。
    """
    pi_sq = np.pi ** 2
    f_sq = f ** 2
    t_sq = t ** 2
    # Ricker wavelet formula
    A = (1.0 - 2.0 * pi_sq * f_sq * t_sq)
    B = np.exp(-pi_sq * f_sq * t_sq)
    return A * B


def build_ricker_dictionary(
        dz: float,
        velocity: float,
        n_samples: int,
        freq_min: float,
        freq_max: float,
        num_atoms: int = 50,
        visualize: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    根据频谱分析结果，构建一个用于匹配追踪的Ricker小波字典。
    *** 最终修正版：直接使用Ricker子波的数学公式构建，最稳定可靠 ***

    Args:
        dz, velocity, n_samples, freq_min, freq_max, num_atoms, visualize...

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: ...
    """
    # --- 1. 计算核心参数 ---
    dt = dz / velocity if velocity else None
    if not dt:
        raise ValueError("必须提供速度和深度采样间隔以计算时间采样间隔 dt。")

    # --- 2. 生成频率列表 ---
    target_frequencies = np.linspace(freq_min, freq_max, num=num_atoms)

    # --- 3. 创建字典矩阵 (使用数学公式) ---
    print(f"正在构建字典，包含 {num_atoms} 个原子...")
    dictionary_matrix = np.zeros((n_samples, num_atoms))

    # 创建一个标准的时间轴，以0为中心
    time_axis = (np.arange(n_samples) - n_samples // 2) * dt

    for i, freq in enumerate(target_frequencies):
        # *** 核心修正：直接调用公式生成小波 ***
        atom_current = ricker_wavelet_formula(time_axis, freq)

        # 对原子进行L2范数归一化
        norm = np.linalg.norm(atom_current)
        if norm > 1e-9:
            atom_normalized = atom_current / norm
        else:
            atom_normalized = atom_current

        dictionary_matrix[:, i] = atom_normalized

    print("字典构建完成！")

    # --- 4. 可视化 (保持不变) ---
    if visualize:
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        plt.figure(figsize=(12, 6))
        plt.title(f"小波字典中的部分原子 (共 {num_atoms} 个)")
        indices_to_plot = [0, num_atoms // 4, num_atoms // 2, num_atoms * 3 // 4, num_atoms - 1]

        for idx in indices_to_plot:
            freq = target_frequencies[idx]
            plt.plot(time_axis, dictionary_matrix[:, idx], label=f'Atom for {freq:.1f} Hz')

        plt.xlabel("时间 (s)")
        plt.ylabel("归一化振幅")
        plt.legend()
        plt.grid(True, linestyle='--')
        y_max = np.max(np.abs(dictionary_matrix)) * 1.1
        plt.ylim(-y_max, y_max)
        plt.show()

    # 虽然我们没用scales来生成，但为了接口统一还是返回它
    central_freq_ricker = pywt.central_frequency('mexh', precision=10)
    scales = central_freq_ricker / (target_frequencies * dt)

    return dictionary_matrix, target_frequencies, scales


def apply_wavelet_reconstruction(
        seismic_volume: np.ndarray,
        dictionary_matrix: np.ndarray,
        n_components_to_remove: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """
    对三维地震数据体应用小波重构，移除最强的背景分量。

    Args:
        seismic_volume (np.ndarray):
            输入的三维地震数据体 (n_inlines, n_xlines, n_samples)。
        dictionary_matrix (np.ndarray):
            预先构建好的小波字典 (n_samples, num_atoms)。
        n_components_to_remove (int):
            要从每条道中移除的能量最强的分量数量。
            根据论文，这个值通常为 1。

    Returns:
        tuple[np.ndarray, np.ndarray]:
            一个元组，包含:
            - processed_volume: 处理后的三维数据体。
            - removed_background: 被移除的背景部分的三维数据体（用于QC）。
    """
    # --- 1. 初始化 ---
    n_inlines, n_xlines, n_samples = seismic_volume.shape

    # 创建空的numpy数组来存储结果
    processed_volume = np.zeros_like(seismic_volume, dtype=np.float32)
    removed_background = np.zeros_like(seismic_volume, dtype=np.float32)

    # OMP算法需要数据是列向量，我们先转置字典以提高后续计算效率
    # fit方法期望的字典是 (n_features, n_components)，这里是 (n_samples, n_atoms)
    # y是 (n_samples, )
    # coef_ 将是 (n_components, )

    print("开始对地震数据体进行小波重构...")

    # --- 2. 逐道处理 ---
    # 使用tqdm来显示进度条
    for i in tqdm(range(n_inlines), desc="处理 Inlines"):
        for j in range(n_xlines):
            # 提取单条地震道
            original_trace = seismic_volume[i, j, :]

            # 如果道是平的（全为0），则跳过
            if np.all(original_trace == 0):
                continue

            # 创建OMP模型实例
            # 我们分解出比要移除的分量稍多一点，以确保能捕获到最强的那个
            # n_nonzero_coefs 定义了我们要找多少个“原子”来表示信号
            omp = OrthogonalMatchingPursuit(n_nonzero_coefs=n_components_to_remove)

            # 使用OMP拟合数据
            # reshape(-1, 1) 将原始道变为列向量，虽然OMP的y可以是一维的
            omp.fit(dictionary_matrix, original_trace)

            # --- 3. 重构背景并执行减法 ---
            # omp.coef_ 包含了被选中的原子的系数（振幅）
            # omp.idx_ 包含了被选中的原子在字典中的列索引

            # 重构背景分量
            background_trace = np.dot(dictionary_matrix[:, omp.idx_], omp.coef_)

            # 从原始道中减去背景
            processed_trace = original_trace - background_trace

            # 存储结果
            processed_volume[i, j, :] = processed_trace
            removed_background[i, j, :] = background_trace

    print("小波重构完成！")
    return processed_volume, removed_background


def visualize_reconstruction_result(original, processed, background, slice_index=None):
    """可视化小波重构前后的剖面对比"""
    if slice_index is None:
        slice_index = original.shape[0] // 2  # 默认显示中间的剖面

    fig, axes = plt.subplots(1, 3, figsize=(18, 8), sharex=True, sharey=True)

    vmax = np.percentile(np.abs(original[slice_index, :, :]), 98)

    axes[0].imshow(original[slice_index, :, :].T, cmap='seismic', aspect='auto', vmin=-vmax, vmax=vmax)
    axes[0].set_title("原始剖面")
    axes[0].set_ylabel("深度采样点")
    axes[0].set_xlabel("Crossline")

    axes[1].imshow(background[slice_index, :, :].T, cmap='seismic', aspect='auto', vmin=-vmax, vmax=vmax)
    axes[1].set_title(f"被移除的背景 (第一个分量)")
    axes[1].set_xlabel("Crossline")

    axes[2].imshow(processed[slice_index, :, :].T, cmap='seismic', aspect='auto', vmin=-vmax, vmax=vmax)
    axes[2].set_title("处理后剖面")
    axes[2].set_xlabel("Crossline")

    plt.suptitle(f"小波重构效果对比 (Inline: {slice_index})", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# --- 使用示例 (保持不变) ---
if __name__ == '__main__':



    FREQ_MIN = 24.0
    FREQ_MAX = 54.0
    DZ_METERS = 5.0
    VELOCITY_MS = 6000.0
    N_SAMPLES = 512

    dictionary, freqs, used_scales = build_ricker_dictionary(
        dz=DZ_METERS,
        velocity=VELOCITY_MS,
        n_samples=N_SAMPLES,
        freq_min=FREQ_MIN,
        freq_max=FREQ_MAX,
        num_atoms=60,
        visualize=True
    )

    print(f"\n构建的字典矩阵形状为: {dictionary.shape}")
    print("这个矩阵现在可以作为scikit-learn OMP算法的输入了。")