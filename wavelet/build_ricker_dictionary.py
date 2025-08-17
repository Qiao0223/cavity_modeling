import numpy as np
import pywt
import matplotlib.pyplot as plt
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
    """
    dt = dz / velocity if velocity else None
    if not dt:
        raise ValueError("必须提供速度和深度采样间隔以计算时间采样间隔 dt。")

    target_frequencies = np.linspace(freq_min, freq_max, num=num_atoms)

    print(f"正在构建字典，包含 {num_atoms} 个原子...")
    dictionary_matrix = np.zeros((n_samples, num_atoms))
    time_axis = (np.arange(n_samples) - n_samples // 2) * dt

    for i, freq in enumerate(target_frequencies):
        atom_current = ricker_wavelet_formula(time_axis, freq)
        norm = np.linalg.norm(atom_current)
        if norm > 1e-9:
            atom_normalized = atom_current / norm
        else:
            atom_normalized = atom_current
        dictionary_matrix[:, i] = atom_normalized
    print("字典构建完成！")

    if visualize:
        # --- 设置中文字体 ---
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

    central_freq_ricker = pywt.central_frequency('mexh', precision=10)
    scales = central_freq_ricker / (target_frequencies * dt)

    return dictionary_matrix, target_frequencies, scales


def apply_wavelet_reconstruction_single_thread(
        seismic_volume: np.ndarray,
        base_dictionary: np.ndarray,
        n_components_to_remove: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """
    对三维地震数据体应用小波重构 (单线程版)。
    手动实现匹配追踪的核心逻辑。
    """
    n_inlines, n_xlines, n_samples = seismic_volume.shape
    num_freq_atoms = base_dictionary.shape[1]

    processed_volume = np.zeros_like(seismic_volume, dtype=np.float32)
    removed_background = np.zeros_like(seismic_volume, dtype=np.float32)

    print("开始对地震数据体进行小波重构 (单线程)...")

    # 使用tqdm来显示总进度
    total_traces = n_inlines * n_xlines
    with tqdm(total=total_traces, desc="处理地震道") as pbar:
        for i in range(n_inlines):
            for j in range(n_xlines):
                original_trace = seismic_volume[i, j, :]

                if np.all(original_trace == 0):
                    pbar.update(1)
                    continue

                residual = original_trace.copy()
                background_trace = np.zeros_like(residual)

                # 迭代移除分量
                for _ in range(n_components_to_remove):
                    best_proj_amp = 0
                    best_reconstructed_atom = None

                    # 遍历所有频率的原子
                    for atom_idx in range(num_freq_atoms):
                        atom = base_dictionary[:, atom_idx]

                        # 使用卷积模拟滑窗匹配
                        projections = np.convolve(residual, atom[::-1], mode='same')

                        # 找到最佳匹配位置和振幅
                        best_shift_idx = np.argmax(np.abs(projections))
                        current_proj_amp = projections[best_shift_idx]

                        if np.abs(current_proj_amp) > np.abs(best_proj_amp):
                            best_proj_amp = current_proj_amp
                            # 重构这个最佳匹配的原子
                            shifted_atom = np.roll(atom, best_shift_idx - n_samples // 2)
                            # 因为原子能量为1，所以重构分量就是 投影系数 * 平移后的原子
                            best_reconstructed_atom = best_proj_amp * shifted_atom

                    # 从残差中减去这个本轮找到的最佳分量
                    if best_reconstructed_atom is not None:
                        residual -= best_reconstructed_atom
                        background_trace += best_reconstructed_atom

                processed_volume[i, j, :] = residual
                removed_background[i, j, :] = background_trace
                pbar.update(1)

    print("小波重构完成！")
    return processed_volume, removed_background


def visualize_reconstruction_result(original, processed, background, slice_index=None):
    """可视化小波重构前后的剖面对比"""
    # --- 设置中文字体 ---
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    if slice_index is None:
        slice_index = original.shape[0] // 2

    fig, axes = plt.subplots(1, 3, figsize=(20, 8), sharex=True, sharey=True)

    vmax = np.percentile(np.abs(original[slice_index, :, :]), 98)

    axes[0].imshow(original[slice_index, :, :].T, cmap='seismic', aspect='auto', vmin=-vmax, vmax=vmax)
    axes[0].set_title("原始剖面")
    axes[0].set_ylabel("深度采样点")
    axes[0].set_xlabel("Crossline")

    axes[1].imshow(background[slice_index, :, :].T, cmap='seismic', aspect='auto', vmin=-vmax, vmax=vmax)
    axes[1].set_title(f"被移除的背景")
    axes[1].set_xlabel("Crossline")

    axes[2].imshow(processed[slice_index, :, :].T, cmap='seismic', aspect='auto', vmin=-vmax, vmax=vmax)
    axes[2].set_title("处理后剖面")
    axes[2].set_xlabel("Crossline")

    plt.suptitle(f"小波重构效果对比 (Inline: {slice_index})", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# --- 主程序 ---
if __name__ == '__main__':
    # --- 1. 设置参数 ---
    # 定义数据路径
    SEISMIC_FILE_PATH = r'C:\Work\sunjie\Python\cavity_modeling\data\input_npy\yingxi_crop.npz'

    # 定义处理参数
    FREQ_MIN = 24.0
    FREQ_MAX = 54.0
    DZ_METERS = 5.0
    VELOCITY_MS = 6000.0
    NUM_ATOMS = 30  # 字典中的原子（频率）数量
    COMPONENTS_TO_REMOVE = 10

    # --- 2. 加载数据 ---
    print(f"从 {SEISMIC_FILE_PATH} 加载数据...")
    with np.load(SEISMIC_FILE_PATH, allow_pickle=True) as npz:
        seis = npz['data']
    print(f"数据加载完成，形状为: {seis.shape}")
    N_SAMPLES = seis.shape[2]

    # --- 3. 构建小波字典 ---
    dictionary, freqs, used_scales = build_ricker_dictionary(
        dz=DZ_METERS,
        velocity=VELOCITY_MS,
        n_samples=N_SAMPLES,
        freq_min=FREQ_MIN,
        freq_max=FREQ_MAX,
        num_atoms=NUM_ATOMS,
        visualize=False  # 在正式处理时可以关闭可视化以节省时间
    )

    # --- 4. 执行小波重构 (单线程) ---
    processed_data, background_data = apply_wavelet_reconstruction_single_thread(
        seismic_volume=seis,
        base_dictionary=dictionary,
        n_components_to_remove=COMPONENTS_TO_REMOVE
    )

    # --- 5. 结果分析与可视化 ---
    print(f"\n处理后数据形状: {processed_data.shape}")
    print(f"移除背景形状: {background_data.shape}")

    # 可视化结果
    visualize_reconstruction_result(seis, processed_data, background_data)
    np.save("processed_data.npy", processed_data)
    np.save("background_data.npy", background_data)