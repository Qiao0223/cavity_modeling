import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

# 全局设置字体，解决中文乱码问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# =============================================================================
# 自定义雷克子波函数
# =============================================================================
def ricker_wavelet(points: int, a: float) -> np.ndarray:
    """
    生成一个雷克子波，也被称为“墨西哥帽小波”。
    """
    vec = np.arange(points) - (points - 1.0) / 2.0
    wsq = a ** 2
    xsq = vec ** 2
    mod = 1 - (xsq / wsq)
    gauss = np.exp(-xsq / (2 * wsq))
    A = 2 / (np.sqrt(3 * a) * (np.pi ** 0.25))
    total = A * mod * gauss
    return total


# =============================================================================
# 核心算法函数
# =============================================================================

def perform_matching_pursuit_2d(
        seismic_profile: np.ndarray,
        horizon_profile: np.ndarray,
        params: dict
) -> tuple[np.ndarray, np.ndarray]:
    """
    对单个2D剖面沿层位执行匹配追踪算法，提取振幅和最佳匹配子波。
    """
    n_traces, n_samples = seismic_profile.shape
    win_half_width = params['time_window_half_width']
    frequencies = params['frequencies_to_search']

    print("\n步骤2: 执行匹配追踪算法...")

    amplitude_1d = np.zeros(n_traces)
    reconstructed_wavelets = np.zeros((n_traces, 2 * win_half_width))

    wavelet_dictionary = {
        freq: ricker_wavelet(2 * win_half_width, 1000 / freq / 3) for freq in frequencies
    }

    with tqdm(total=n_traces, desc="匹配追踪2D剖面") as pbar:
        for j in range(n_traces):
            t0 = horizon_profile[j]
            t_start, t_end = max(0, t0 - win_half_width), min(n_samples, t0 + win_half_width)
            trace_window = seismic_profile[j, t_start:t_end]

            if len(trace_window) < 2 * win_half_width:
                trace_window = np.pad(trace_window, (0, 2 * win_half_width - len(trace_window)), 'constant')

            best_corr, best_amp, best_wavelet = -np.inf, 0, None
            for freq in frequencies:
                candidate = wavelet_dictionary[freq]
                dot_sw = np.dot(trace_window, candidate)
                dot_ww = np.dot(candidate, candidate)
                if dot_ww < 1e-9: continue

                amp = dot_sw / dot_ww
                corr = -np.sum((trace_window - amp * candidate) ** 2)

                if corr > best_corr:
                    best_corr, best_amp, best_wavelet = corr, amp, candidate

            amplitude_1d[j] = best_amp
            if best_wavelet is not None:
                reconstructed_wavelets[j, :] = best_wavelet * best_amp
            pbar.update(1)

    print("匹配追踪完成！")
    return amplitude_1d, reconstructed_wavelets


def separate_strong_reflection_2d(
        seismic_profile: np.ndarray,
        horizon_profile: np.ndarray,
        amplitude_profile: np.ndarray,
        reconstructed_wavelets: np.ndarray,
        params: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算分离系数并从原始剖面中减去校正后的强反射。
    """
    print("\n步骤3: 计算分离系数并执行分离...")

    background_amplitude = gaussian_filter1d(amplitude_profile, sigma=params['smoothing_sigma'])
    lambda_profile = background_amplitude / (amplitude_profile + 1e-9)

    # --- !! 关键优化：对 Lambda 剖面本身进行平滑 !! ---
    lambda_smoothing_sigma = params.get('lambda_smoothing_sigma')
    if lambda_smoothing_sigma and lambda_smoothing_sigma > 0:
        print(f"  - 对 Lambda 剖面进行平滑 (sigma={lambda_smoothing_sigma})...")
        lambda_profile = gaussian_filter1d(lambda_profile, sigma=lambda_smoothing_sigma)

    final_profile = seismic_profile.copy()
    win_half_width = params['time_window_half_width']

    with tqdm(total=seismic_profile.shape[0], desc="执行最终分离") as pbar:
        for j in range(seismic_profile.shape[0]):
            t0 = horizon_profile[j]
            t_start, t_end = max(0, t0 - win_half_width), min(seismic_profile.shape[1], t0 + win_half_width)

            corrected_wavelet = reconstructed_wavelets[j, :] * lambda_profile[j]
            len_to_place = t_end - t_start

            final_profile[j, t_start:t_end] -= corrected_wavelet[:len_to_place]
            pbar.update(1)

    print("强反射分离完成！")
    return final_profile, background_amplitude, lambda_profile


def visualize_separation_results_2d(
        original_slice, final_slice, horizon_profile,
        amp_actual, amp_bg, lambda_vals, slice_info, params
):
    """可视化强反射分离算法的每一步结果。"""
    fig, axes = plt.subplots(2, 2, figsize=(20, 14), gridspec_kw={'height_ratios': [3, 2]})
    fig.suptitle(f"强反射分离结果 - {slice_info}", fontsize=18)

    vmax = np.percentile(np.abs(original_slice), 99)
    xlabel = f"{params['slice_type']} 道号"

    # 1. 原始地震剖面
    axes[0, 0].imshow(original_slice.T, aspect='auto', cmap='seismic', vmin=-vmax, vmax=vmax)
    axes[0, 0].plot(horizon_profile, 'k-', lw=2, label='层位')
    axes[0, 0].set_title("1. 原始地震剖面")
    axes[0, 0].set_ylabel("时间采样点")
    axes[0, 0].legend()

    # 2. 最终分离后剖面
    axes[0, 1].imshow(final_slice.T, aspect='auto', cmap='seismic', vmin=-vmax, vmax=vmax)
    axes[0, 1].plot(horizon_profile, 'k-', lw=2, label='层位')
    axes[0, 1].set_title("2. 分离后剖面 (弱反射突显)")
    axes[0, 1].legend()

    # 3. 振幅曲线对比
    axes[1, 0].plot(amp_actual, label='匹配振幅 (实际)', color='blue', lw=2)
    axes[1, 0].plot(amp_bg, label=f'背景趋势 (Sigma={params["smoothing_sigma"]})', color='red', linestyle='--', lw=2)
    axes[1, 0].set_title("3. 振幅与背景趋势")
    axes[1, 0].set_xlabel(xlabel)
    axes[1, 0].set_ylabel("振幅")
    axes[1, 0].grid(True, linestyle=':', alpha=0.6)
    axes[1, 0].legend()

    # 4. 分离系数 Lambda
    ax = axes[1, 1]
    ax.plot(lambda_vals, label='λ = 背景/实际', color='green', lw=2)
    ax.axhline(1.0, color='gray', linestyle='--')
    ax.set_title("4. 分离系数 (Lambda)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("λ 值")
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_ylim(-1, 3)  # 固定Y轴范围
    ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# =============================================================================
# 主程序
# =============================================================================
if __name__ == '__main__':
    # --- 1. 设置文件路径 ---
    SEISMIC_NPY_PATH = r'C:\Work\sunjie\Python\cavity_modeling\data\input_npy\yingxi_crop.npy'
    HORIZON_NPY_PATH = r'C:\Work\sunjie\Python\cavity_modeling\data\input_npy\yingxi_hor_TO3t.npy'

    # --- 2. 定义处理目标和算法参数 ---
    PROCESSING_AXIS = 1
    SLICE_INDEX = 500

    # 算法核心参数 (可调)
    algorithm_params = {
        'time_window_half_width': 25,
        'frequencies_to_search': np.arange(15, 70, 1),
        'smoothing_sigma': 10.0,
        'lambda_smoothing_sigma': 5.0,  # <-- 新增：Lambda平滑系数 (道数)，从 3.0 或 5.0 开始尝试
    }

    # --- 3. 加载完整的3D数据 ---
    try:
        print("\n步骤1: 加载数据...")
        seismic_3d = np.load(SEISMIC_NPY_PATH)
        horizon_2d = np.load(HORIZON_NPY_PATH)
        print("数据加载完成。")
        print(f"  - 地震数据形状: {seismic_3d.shape}")
        print(f"  - 层位数据形状: {horizon_2d.shape}")
    except Exception as e:
        print(f"数据加载失败，错误信息: {e}")
        exit()

    # --- 4. 根据 PROCESSING_AXIS 提取单个2D剖面 ---
    if PROCESSING_AXIS == 0:
        slice_type = "Inline"
        other_axis_name = "Xline"
        seismic_2d_slice = seismic_3d[SLICE_INDEX, :, :]
        horizon_1d_slice = horizon_2d[SLICE_INDEX, :]
    elif PROCESSING_AXIS == 1:
        slice_type = "Xline"
        other_axis_name = "Inline"
        seismic_2d_slice = seismic_3d[:, SLICE_INDEX, :]
        horizon_1d_slice = horizon_2d[:, SLICE_INDEX]
    else:
        print(f"错误: 无效的 PROCESSING_AXIS: {PROCESSING_AXIS}。请选择 0 (Inline) 或 1 (Xline)。")
        exit()

    algorithm_params['slice_type'] = other_axis_name

    print(f"\n已提取 {slice_type} 剖面 {SLICE_INDEX} (axis={PROCESSING_AXIS}) 进行处理...")

    # --- 5. 执行匹配追踪 ---
    amplitude_1d, wavelets_2d = perform_matching_pursuit_2d(
        seismic_profile=seismic_2d_slice,
        horizon_profile=horizon_1d_slice,
        params=algorithm_params
    )

    # --- 6. 执行强反射分离 ---
    final_2d, background_amp_1d, lambda_1d = separate_strong_reflection_2d(
        seismic_profile=seismic_2d_slice,
        horizon_profile=horizon_1d_slice,
        amplitude_profile=amplitude_1d,
        reconstructed_wavelets=wavelets_2d,
        params=algorithm_params
    )

    # --- 7. 可视化最终结果 ---
    visualize_separation_results_2d(
        original_slice=seismic_2d_slice,
        final_slice=final_2d,
        horizon_profile=horizon_1d_slice,
        amp_actual=amplitude_1d,
        amp_bg=background_amp_1d,
        lambda_vals=lambda_1d,
        slice_info=f"{slice_type} {SLICE_INDEX}",
        params=algorithm_params
    )