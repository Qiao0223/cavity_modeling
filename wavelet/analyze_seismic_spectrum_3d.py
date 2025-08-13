import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from scipy.signal import windows

plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False   # 解决保存图像是负号'-'显示为方块的问题

def analyze_seismic_spectrum_3d(
        seismic_volume: np.ndarray,
        dz: float,
        velocity: float = None,
        db_threshold: float = -10.0
) -> dict:
    """
    对三维深度域地震数据体进行频谱分析。

    该函数计算数据体所有道的平均振幅谱，并识别峰值频率和有效频宽。
    核心分析在空间频率（波数, 1/m）域进行，如果提供速度，则同时换算到
    时间频率（Hz）域。

    Args:
        seismic_volume (np.ndarray):
            输入的三维地震数据体，维度应为 (n_inlines, n_xlines, n_samples)。
        dz (float):
            深度采样间隔，单位为米 (m)。
        velocity (float, optional):
            用于时深转换的宏观平均速度，单位为米/秒 (m/s)。
            如果提供，结果将同时以时间频率（Hz）展示。默认为 None。
        db_threshold (float, optional):
            用于定义有效频宽的dB阈值。频宽定义为振幅谱从峰值下降
            该dB值所对应的频率范围。默认为 -10.0 dB。

    Returns:
        dict:
            一个包含频谱分析关键结果的字典，包括：
            - 'wavenumber_axis': 空间频率轴 (1/m)
            - 'avg_spectrum': 平均振幅谱
            - 'peak_wavenumber': 峰值空间频率 (1/m)
            - 'wavenumber_band': 有效空间频宽 (f_min, f_max) (1/m)
            以及（如果提供了速度）:
            - 'temporal_freq_axis': 时间频率轴 (Hz)
            - 'peak_temporal_freq': 峰值时间频率 (Hz)
            - 'temporal_band': 有效时间频宽 (f_min, f_max) (Hz)
    """
    # --- 1. 输入验证和数据准备 ---
    if seismic_volume.ndim != 3:
        raise ValueError("输入 seismic_volume 必须是三维 NumPy 数组。")

    ni, nx, nt = seismic_volume.shape
    # 将三维数据重塑为二维，每一行是一条地震道，便于处理
    traces = seismic_volume.reshape(-1, nt)
    num_traces = traces.shape[0]

    if num_traces == 0:
        raise ValueError("地震数据体中不包含有效的地震道。")

    # --- 2. 计算平均频谱 ---
    # 创建一个汉宁窗，以减少频谱泄漏
    window = windows.hann(nt)

    # 我们只计算一次频率轴
    # rfft 专门用于实数输入，更高效
    wavenumber_axis = rfftfreq(nt, d=dz)

    # 存储每条道的频谱
    all_spectra = np.zeros((num_traces, len(wavenumber_axis)))

    for i in range(num_traces):
        trace = traces[i, :]
        # 应用窗函数并计算FFT
        spectrum = rfft(trace * window)
        all_spectra[i, :] = np.abs(spectrum)

    # 计算所有道的平均频谱
    avg_spectrum = np.mean(all_spectra, axis=0)

    # --- 3. 分析频谱，提取关键指标 ---
    # 找到峰值频率
    peak_index = np.argmax(avg_spectrum)
    peak_wavenumber = wavenumber_axis[peak_index]
    peak_amplitude = avg_spectrum[peak_index]

    # 计算有效频宽
    # 将振幅谱转换为dB，相对于峰值
    with np.errstate(divide='ignore'):  # 忽略 log10(0) 的警告
        db_spectrum = 20 * np.log10(avg_spectrum / peak_amplitude)
    db_spectrum[np.isneginf(db_spectrum)] = -999  # 将-inf替换为极小值

    # 找到所有高于阈值的频率点
    above_threshold_indices = np.where(db_spectrum >= db_threshold)[0]

    if len(above_threshold_indices) > 0:
        min_idx = above_threshold_indices[0]
        max_idx = above_threshold_indices[-1]
        wavenumber_band = (wavenumber_axis[min_idx], wavenumber_axis[max_idx])
    else:
        # 如果没有点高于阈值（不太可能），则频宽为0
        wavenumber_band = (peak_wavenumber, peak_wavenumber)

    # --- 4. 准备返回结果和可视化 ---
    results = {
        'wavenumber_axis': wavenumber_axis,
        'avg_spectrum': avg_spectrum,
        'peak_wavenumber': peak_wavenumber,
        'wavenumber_band': wavenumber_band,
    }

    # 如果提供了速度，计算时间频率域的结果
    if velocity:
        temporal_freq_axis = wavenumber_axis * velocity
        peak_temporal_freq = peak_wavenumber * velocity
        temporal_band = (wavenumber_band[0] * velocity, wavenumber_band[1] * velocity)
        results.update({
            'temporal_freq_axis': temporal_freq_axis,
            'peak_temporal_freq': peak_temporal_freq,
            'temporal_band': temporal_band,
        })

    # --- 5. 可视化 ---
    fig, ax = plt.subplots(figsize=(12, 7))

    # 使用主坐标轴绘制空间频率
    ax.plot(wavenumber_axis, avg_spectrum, color='b', label='平均振幅谱')
    ax.set_xlabel('空间频率 (波数, 1/m)', color='b', fontsize=14)
    ax.set_ylabel('振幅', fontsize=14)
    ax.tick_params(axis='x', labelcolor='b')
    ax.grid(True, linestyle='--', alpha=0.6)

    # 标注峰值和频宽
    ax.axvline(peak_wavenumber, color='r', linestyle='--', label=f'峰值波数: {peak_wavenumber:.4f} 1/m')
    ax.axvspan(wavenumber_band[0], wavenumber_band[1], color='g', alpha=0.3, label=f'{db_threshold}dB 频宽')

    title_text = f"三维地震数据平均频谱分析\n主波数 = {peak_wavenumber:.4f} (1/m)"

    # 如果有时间频率，使用次坐标轴绘制
    if velocity:
        ax2 = ax.twiny()  # 共享Y轴
        ax2.set_xlabel('时间频率 (Hz)', color='k', fontsize=14)
        ax2.set_xlim(wavenumber_axis[0] * velocity, wavenumber_axis[-1] * velocity)
        ax2.tick_params(axis='x', labelcolor='k')
        title_text += f" | 主频 ≈ {peak_temporal_freq:.1f} Hz"

    ax.set_title(title_text, fontsize=16, pad=30 if velocity else 20)
    # 合并图例
    lines, labels = ax.get_legend_handles_labels()
    ax.legend(lines, labels, loc='upper right')

    plt.tight_layout()
    plt.show()

    return results


# --- 使用示例 ---
if __name__ == '__main__':
    npz = np.load(r'C:\Work\sunjie\Python\cavity_modeling\data\input_npy\yingxi_crop.npz', allow_pickle=True)
    seis3d = npz['data']
    DZ_METERS = 5.0
    VELOCITY_MS = 6000.0
    print("正在分析模拟地震数据...")

    # 2. 调用分析函数
    analysis_results = analyze_seismic_spectrum_3d(
        seismic_volume=seis3d,
        dz=DZ_METERS,
        velocity=VELOCITY_MS,
        db_threshold=-6.0  # 使用-6dB频宽进行演示
    )

    # 3. 打印分析结果
    print("\n--- 频谱分析结果 ---")
    print(f"峰值空间频率 (波数): {analysis_results['peak_wavenumber']:.4f} 1/m")
    print(
        f"有效空间频宽 (-6dB): ({analysis_results['wavenumber_band'][0]:.4f}, {analysis_results['wavenumber_band'][1]:.4f}) 1/m")
    if 'peak_temporal_freq' in analysis_results:
        print(f"峰值时间频率: {analysis_results['peak_temporal_freq']:.2f} Hz")
        print(
            f"有效时间频宽 (-6dB): ({analysis_results['temporal_band'][0]:.2f}, {analysis_results['temporal_band'][1]:.2f}) Hz")
