import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os


# --- 基础工具函数 ---
def ricker_wavelet_formula(t, f):
    """根据数学公式生成Ricker子波。"""
    pi_sq = np.pi ** 2
    f_sq = f ** 2
    t_sq = t ** 2
    A = (1.0 - 2.0 * pi_sq * f_sq * t_sq)
    B = np.exp(-pi_sq * f_sq * t_sq)
    return A * B


def build_ricker_dictionary(dz, velocity, n_samples, freq_min, freq_max, num_atoms=50):
    """构建一个Ricker小波字典矩阵。"""
    dt = dz / velocity
    target_frequencies = np.linspace(freq_min, freq_max, num=num_atoms)
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
    return dictionary_matrix, target_frequencies


# --- 步骤1：耗时的统计计算函数 ---
def calculate_and_cache_energy_stats(
        seismic_volume, dz, velocity, freq_min, freq_max, num_atoms_in_dict,
        n_components_to_decompose, cache_file_path):
    """
    执行耗时的全局能量统计，并将完整的统计结果（频率和能量）保存到.npz文件。
    """
    n_inlines, n_xlines, n_samples = seismic_volume.shape

    print("正在构建Ricker小波字典用于统计...")
    dictionary, frequencies = build_ricker_dictionary(
        dz, velocity, n_samples, freq_min, freq_max, num_atoms_in_dict
    )

    total_energy_per_frequency = np.zeros(num_atoms_in_dict)

    print("开始执行耗时的全局能量统计 (此过程只需运行一次)...")
    total_traces = n_inlines * n_xlines
    with tqdm(total=total_traces, desc="统计能量") as pbar:
        for i in range(n_inlines):
            for j in range(n_xlines):
                original_trace = seismic_volume[i, j, :]
                if np.all(original_trace == 0):
                    pbar.update(1)
                    continue

                residual = original_trace.copy()

                for _ in range(n_components_to_decompose):
                    best_proj_amp = 0
                    best_atom_idx = -1

                    for atom_idx in range(num_atoms_in_dict):
                        atom = dictionary[:, atom_idx]
                        projections = np.convolve(residual, atom[::-1], mode='same')
                        max_proj_in_trace = np.max(np.abs(projections))

                        if max_proj_in_trace > best_proj_amp:
                            best_proj_amp = max_proj_in_trace
                            best_atom_idx = atom_idx

                    if best_atom_idx != -1:
                        atom = dictionary[:, best_atom_idx]
                        projections = np.convolve(residual, atom[::-1], mode='same')
                        best_shift_idx = np.argmax(np.abs(projections))
                        amplitude = projections[best_shift_idx]
                        reconstructed_component = amplitude * np.roll(atom, best_shift_idx - n_samples // 2)

                        total_energy_per_frequency[best_atom_idx] += amplitude ** 2

                        residual -= reconstructed_component
                pbar.update(1)

    print("\n能量统计完成！")

    try:
        np.savez(cache_file_path, frequencies=frequencies, energy_stats=total_energy_per_frequency)
        print(f"完整的能量统计结果已保存到: {cache_file_path}")
    except Exception as e:
        print(f"警告：保存统计结果失败！错误: {e}")

    return frequencies, total_energy_per_frequency


# --- 步骤2：背景移除函数 ---
def remove_background_with_top_n_components(
        seismic_volume,
        top_n_frequencies,
        top_n_energies,
        dz,
        velocity):
    """
    使用能量排名前N的频率分量，构建复合背景模板并移除。
    """
    n_inlines, n_xlines, n_samples = seismic_volume.shape
    dt = dz / velocity
    time_axis = (np.arange(n_samples) - n_samples // 2) * dt

    print(f"使用能量排名前 {len(top_n_frequencies)} 的频率构建复合背景模板...")

    composite_background_waveform = np.zeros(n_samples)
    total_energy = np.sum(top_n_energies)

    if total_energy > 0:
        for i, freq in enumerate(top_n_frequencies):
            waveform = ricker_wavelet_formula(time_axis, freq)
            weight = top_n_energies[i] / total_energy
            composite_background_waveform += weight * waveform
            print(f"  - 添加频率 {freq:.2f} Hz, 权重 {weight:.2f}")

    norm = np.linalg.norm(composite_background_waveform)
    if norm > 1e-9:
        composite_background_waveform /= norm

    processed_volume = np.zeros_like(seismic_volume, dtype=np.float32)
    removed_background = np.zeros_like(seismic_volume, dtype=np.float32)

    print("开始执行基于复合模板的背景移除...")
    total_traces = n_inlines * n_xlines
    with tqdm(total=total_traces, desc="移除背景") as pbar:
        for i in range(n_inlines):
            for j in range(n_xlines):
                original_trace = seismic_volume[i, j, :]
                if np.all(original_trace == 0):
                    pbar.update(1)
                    continue

                projections = np.convolve(original_trace, composite_background_waveform[::-1], mode='same')
                best_shift_idx = np.argmax(np.abs(projections))
                amplitude_scaler = projections[best_shift_idx]

                shifted_waveform = np.roll(composite_background_waveform, best_shift_idx - n_samples // 2)
                background_on_this_trace = amplitude_scaler * shifted_waveform

                processed_trace = original_trace - background_on_this_trace

                processed_volume[i, j, :] = processed_trace
                removed_background[i, j, :] = background_on_this_trace
                pbar.update(1)

    print("背景移除完成！")
    return processed_volume, removed_background, composite_background_waveform


def visualize_reconstruction_result(original, processed, background, slice_index=None):
    """可视化处理前后的剖面对比"""
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
    axes[1].set_title("被移除的全局背景")
    axes[1].set_xlabel("Crossline")

    axes[2].imshow(processed[slice_index, :, :].T, cmap='seismic', aspect='auto', vmin=-vmax, vmax=vmax)
    axes[2].set_title("处理后剖面")
    axes[2].set_xlabel("Crossline")

    plt.suptitle(f"基于全局分量的背景移除效果 (Inline: {slice_index})", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# --- 主程序 (带有缓存逻辑) ---
if __name__ == '__main__':
    # --- 1. 设置参数 ---
    SEISMIC_FILE_PATH = r'C:\Work\sunjie\Python\cavity_modeling\data\input_npy\yingxi_crop.npz'
    ENERGY_STATS_CACHE_FILE = 'energy_statistics.npz'

    FREQ_MIN, FREQ_MAX = 24.0, 54.0
    DZ_METERS, VELOCITY_MS = 5.0, 6000.0
    NUM_ATOMS_IN_DICT = 30
    COMPONENTS_TO_DECOMPOSE_FOR_STATS = 5

    # *** 关键调试参数：要移除的前N个分量 ***
    N_COMPONENTS_TO_REMOVE = 3

    # --- 2. 加载数据 ---
    print(f"从 {SEISMIC_FILE_PATH} 加载数据...")
    with np.load(SEISMIC_FILE_PATH, allow_pickle=True) as npz:
        seis = npz['data']
    print(f"数据加载完成，形状为: {seis.shape}")

    # --- 3. 检查缓存，决定是计算还是加载完整的统计结果 ---
    freqs, energy_stats = None, None
    if os.path.exists(ENERGY_STATS_CACHE_FILE):
        try:
            print(f"--- 从缓存文件 {ENERGY_STATS_CACHE_FILE} 加载能量统计结果 ---")
            cached_data = np.load(ENERGY_STATS_CACHE_FILE)
            freqs = cached_data['frequencies']
            energy_stats = cached_data['energy_stats']
        except Exception as e:
            print(f"警告：读取缓存文件失败，将重新计算。错误: {e}")
            if os.path.exists(ENERGY_STATS_CACHE_FILE):
                os.remove(ENERGY_STATS_CACHE_FILE)

    if freqs is None or energy_stats is None:
        print("--- 未找到有效缓存，开始执行一次性耗时统计 ---")
        freqs, energy_stats = calculate_and_cache_energy_stats(
            seis, DZ_METERS, VELOCITY_MS, FREQ_MIN, FREQ_MAX,
            NUM_ATOMS_IN_DICT, COMPONENTS_TO_DECOMPOSE_FOR_STATS, ENERGY_STATS_CACHE_FILE
        )

    # --- 4. 从完整的统计结果中，选出前N名 ---
    print(f"--- 将使用能量排名前 {N_COMPONENTS_TO_REMOVE} 的频率进行背景移除 ---")
    sorted_indices = np.argsort(energy_stats)[::-1]
    top_n_indices = sorted_indices[:N_COMPONENTS_TO_REMOVE]
    top_n_freqs = freqs[top_n_indices]
    top_n_energies = energy_stats[top_n_indices]

    # --- 5. 执行快速的背景移除 ---
    processed_data, background_data, global_waveform = remove_background_with_top_n_components(
        seismic_volume=seis,
        top_n_frequencies=top_n_freqs,
        top_n_energies=top_n_energies,
        dz=DZ_METERS,
        velocity=VELOCITY_MS
    )

    # --- 6. 结果分析与可视化 ---
    print(f"\n处理后数据形状: {processed_data.shape}")
    print(f"移除背景形状: {background_data.shape}")

    visualize_reconstruction_result(seis, processed_data, background_data)

    # 可视化能量统计结果 (现在每次都可以显示)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(10, 5))
    plt.title("全局频率能量贡献统计")
    plt.bar(freqs, energy_stats, width=(freqs[1] - freqs[0]) * 0.8)
    plt.xlabel("频率 (Hz)")
    plt.ylabel("累积能量 (振幅平方和)")
    plt.grid(True, axis='y')
    plt.show()

    # 可视化最终使用的全局背景波形
    plt.figure(figsize=(10, 5))
    plt.title("最终使用的全局背景波形模板")
    time_axis = (np.arange(seis.shape[2]) - seis.shape[2] // 2) * (DZ_METERS / VELOCITY_MS)
    plt.plot(time_axis, global_waveform, label='全局背景波形')
    plt.xlabel("时间 (s)")
    plt.ylabel("归一化振幅")
    plt.legend()
    plt.grid(True)
    plt.show()