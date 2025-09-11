import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
import os

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# =============================================================================
# 自定义雷克子波函数 (保持不变)
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
# 核心算法函数 - 3D版本
# =============================================================================

def perform_matching_pursuit_3d(
        seismic_3d: np.ndarray,
        horizon_2d: np.ndarray,
        params: dict
) -> tuple[np.ndarray, np.ndarray]:
    """
    对整个3D地震体沿层位执行匹配追踪算法，提取振幅和最佳匹配子波。
    """
    n_inlines, n_xlines, n_samples = seismic_3d.shape
    win_half_width = params['time_window_half_width']
    frequencies = params['frequencies_to_search']

    print("\n步骤2: 执行3D匹配追踪算法...")

    amplitude_3d = np.zeros((n_inlines, n_xlines))
    # 对于3D，我们不直接存储reconstructed_wavelets的3D体，因为内存消耗巨大
    # 而是存储最佳匹配的子波索引和振幅，但这里简化为只返回振幅
    # 如果确实需要，可以考虑在需要时重建或存储更紧凑的表示

    wavelet_dictionary = {
        freq: ricker_wavelet(2 * win_half_width, 1000 / freq / 3) for freq in frequencies
    }

    # 使用两个tqdm进度条，一个用于Inline，一个用于Xline
    with tqdm(total=n_inlines * n_xlines, desc="匹配追踪3D体") as pbar_total:
        for i in range(n_inlines):
            for j in range(n_xlines):
                t0 = int(horizon_2d[i, j])  # 层位是2D的
                t_start, t_end = max(0, t0 - win_half_width), min(n_samples, t0 + win_half_width)

                trace_window = seismic_3d[i, j, t_start:t_end].copy()  # 使用copy确保独立操作

                if len(trace_window) < 2 * win_half_width:
                    trace_window = np.pad(trace_window, (0, 2 * win_half_width - len(trace_window)), 'constant')

                best_corr, best_amp = -np.inf, 0

                for freq in frequencies:
                    candidate = wavelet_dictionary[freq]
                    dot_sw = np.dot(trace_window, candidate)
                    dot_ww = np.dot(candidate, candidate)
                    if dot_ww < 1e-9: continue

                    amp = dot_sw / dot_ww
                    corr = -np.sum((trace_window - amp * candidate) ** 2)

                    if corr > best_corr:
                        best_corr, best_amp = corr, amp

                amplitude_3d[i, j] = best_amp
                pbar_total.update(1)

    print("3D匹配追踪完成！")
    return amplitude_3d  # 不再返回reconstructed_wavelets_3d


def separate_strong_reflection_3d(
        seismic_3d: np.ndarray,
        horizon_2d: np.ndarray,
        amplitude_3d: np.ndarray,
        params: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算分离系数并从原始3D体中减去校正后的强反射。
    """
    print("\n步骤3: 计算3D分离系数并执行分离...")

    n_inlines, n_xlines, n_samples = seismic_3d.shape
    win_half_width = params['time_window_half_width']
    frequencies = params['frequencies_to_search']

    # 对2D振幅体进行高斯平滑
    print(f"  - 对振幅体进行2D高斯平滑 (sigma={params['smoothing_sigma']})...")
    background_amplitude_3d = gaussian_filter(amplitude_3d, sigma=params['smoothing_sigma'], mode='nearest')

    lambda_3d = background_amplitude_3d / (amplitude_3d + 1e-9)

    lambda_smoothing_sigma = params.get('lambda_smoothing_sigma')
    if lambda_smoothing_sigma and lambda_smoothing_sigma > 0:
        print(f"  - 对 Lambda 体进行2D平滑 (sigma={lambda_smoothing_sigma})...")
        lambda_3d = gaussian_filter(lambda_3d, sigma=lambda_smoothing_sigma, mode='nearest')

    final_seismic_3d = seismic_3d.copy()

    # 获取压制系数
    suppression_factor = params.get('suppression_factor', 1.0)
    if suppression_factor != 1.0:
        print(f"  - [优化] 应用强反射压制系数 (系数={suppression_factor})...")

    wavelet_dictionary = {
        freq: ricker_wavelet(2 * win_half_width, 1000 / freq / 3) for freq in frequencies
    }

    with tqdm(total=n_inlines * n_xlines, desc="执行最终3D分离") as pbar_total:
        for i in range(n_inlines):
            for j in range(n_xlines):
                t0 = int(horizon_2d[i, j])
                t_start, t_end = max(0, t0 - win_half_width), min(n_samples, t0 + win_half_width)

                trace_window = seismic_3d[i, j, t_start:t_end].copy()

                if len(trace_window) < 2 * win_half_width:
                    trace_window_padded = np.pad(trace_window, (0, 2 * win_half_width - len(trace_window)), 'constant')
                else:
                    trace_window_padded = trace_window

                # 重新进行匹配追踪以获取最佳子波
                best_corr, best_amp, best_wavelet_candidate = -np.inf, 0, None
                for freq in frequencies:
                    candidate = wavelet_dictionary[freq]
                    dot_sw = np.dot(trace_window_padded, candidate)
                    dot_ww = np.dot(candidate, candidate)
                    if dot_ww < 1e-9: continue

                    amp = dot_sw / dot_ww
                    corr = -np.sum((trace_window_padded - amp * candidate) ** 2)

                    if corr > best_corr:
                        best_corr, best_amp, best_wavelet_candidate = corr, amp, candidate

                if best_wavelet_candidate is not None:
                    # 原始估计的强反射 (基于重新匹配的子波和当前位置的振幅)
                    strong_reflection_estimated = best_wavelet_candidate * amplitude_3d[i, j]

                    # 应用分离系数和压制系数
                    corrected_wavelet = strong_reflection_estimated * lambda_3d[i, j] * suppression_factor

                    len_to_place = t_end - t_start
                    final_seismic_3d[i, j, t_start:t_end] -= corrected_wavelet[:len_to_place]
                pbar_total.update(1)

    print("3D强反射分离完成！")
    return final_seismic_3d, background_amplitude_3d, lambda_3d


# 辅助函数：二维高斯平滑（对于2D的振幅图）
def gaussian_filter(data_2d: np.ndarray, sigma: float, mode: str = 'nearest') -> np.ndarray:
    """对二维数组进行高斯平滑"""
    from scipy.ndimage import gaussian_filter as gf_ndimage
    return gf_ndimage(data_2d, sigma=sigma, mode=mode)


# =============================================================================
# 可视化函数 - 适用于3D结果的切片展示
# =============================================================================
def visualize_separation_results_3d_slice(
        original_3d, final_3d, horizon_2d,
        amp_actual_3d, amp_bg_3d, lambda_3d,
        slice_index: int, slice_axis: int, params: dict, save_path: str = None
):
    """
    可视化3D分离算法的某个切片结果。
    slice_axis: 0 (Inline) 或 1 (Xline)
    """
    if slice_axis == 0:  # Inline切片
        original_slice = original_3d[slice_index, :, :]
        final_slice = final_3d[slice_index, :, :]
        horizon_1d = horizon_2d[slice_index, :]
        slice_type_str = f"Inline {slice_index}"
        xlabel = "Xline 道号"
    else:  # Xline切片
        original_slice = original_3d[:, slice_index, :]
        final_slice = final_3d[:, slice_index, :]
        horizon_1d = horizon_2d[:, slice_index]
        slice_type_str = f"Xline {slice_index}"
        xlabel = "Inline 道号"

    amp_actual_slice = amp_actual_3d[slice_index, :] if slice_axis == 0 else amp_actual_3d[:, slice_index]
    amp_bg_slice = amp_bg_3d[slice_index, :] if slice_axis == 0 else amp_bg_3d[:, slice_index]
    lambda_slice = lambda_3d[slice_index, :] if slice_axis == 0 else lambda_3d[:, slice_index]

    fig, axes = plt.subplots(2, 2, figsize=(20, 14), gridspec_kw={'height_ratios': [3, 2]})
    fig.suptitle(f"强反射分离结果 (3D切片) - {slice_type_str}", fontsize=18)

    vmax = np.percentile(np.abs(original_slice), 99)

    # 1. 原始地震剖面
    axes[0, 0].imshow(original_slice.T, aspect='auto', cmap='seismic', vmin=-vmax, vmax=vmax)
    axes[0, 0].plot(horizon_1d, 'k-', lw=2, label='层位')
    axes[0, 0].set_title("1. 原始地震剖面")
    axes[0, 0].set_ylabel("时间采样点")
    axes[0, 0].legend()

    # 2. 最终分离后剖面
    axes[0, 1].imshow(final_slice.T, aspect='auto', cmap='seismic', vmin=-vmax, vmax=vmax)
    axes[0, 1].plot(horizon_1d, 'k-', lw=2, label='层位')
    axes[0, 1].set_title("2. 分离后剖面 (弱反射突显)")
    axes[0, 1].legend()

    # 3. 振幅曲线对比
    axes[1, 0].plot(amp_actual_slice, label='匹配振幅 (实际)', color='blue', lw=2)
    axes[1, 0].plot(amp_bg_slice, label=f'背景趋势 (Sigma={params["smoothing_sigma"]})', color='red', linestyle='--',
                    lw=2)
    axes[1, 0].set_title("3. 振幅与背景趋势")
    axes[1, 0].set_xlabel(xlabel)
    axes[1, 0].set_ylabel("振幅")
    axes[1, 0].grid(True, linestyle=':', alpha=0.6)
    axes[1, 0].legend()

    # 4. 分离系数 Lambda
    ax = axes[1, 1]
    ax.plot(lambda_slice, label='λ = 背景/实际', color='green', lw=2)
    ax.axhline(1.0, color='gray', linestyle='--')
    ax.set_title("4. 分离系数 (Lambda)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("λ 值")
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_ylim(-1, 3)
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
    OUTPUT_DIR = r'C:\Work\sunjie\Python\cavity_modeling\data\output_npy'  # 新增：输出目录
    OUTPUT_FILE_NAME = 'separation.npy'

    os.makedirs(OUTPUT_DIR, exist_ok=True)  # 创建输出目录

    # --- 2. 定义处理目标和算法参数 ---
    # 算法核心参数 (可调)
    algorithm_params = {
        'time_window_half_width': 25,
        'frequencies_to_search': np.arange(15, 70, 5),
        'smoothing_sigma': 30.0,  # 振幅平滑的sigma (现在是2D平滑)
        'lambda_smoothing_sigma': 5.0,  # lambda平滑的sigma (现在是2D平滑)
        'suppression_factor': 1,  # 强制压制系数
    }

    # 可视化参数 (用于展示某个切片)
    VIS_SLICE_AXIS = 0  # 0 for Inline, 1 for Xline
    VIS_SLICE_INDEX = 100  # 要可视化的切片索引

    # --- 3. 加载完整的3D数据 ---
    try:
        print("\n步骤1: 加载3D数据...")
        seismic_3d = np.load(SEISMIC_NPY_PATH)
        horizon_2d = np.load(HORIZON_NPY_PATH)  # 层位现在是2D的 (Inline x Xline)
        print("3D数据加载完成。")
        print(f"  - 地震数据形状: {seismic_3d.shape} (Inline, Xline, Samples)")
        print(f"  - 层位数据形状: {horizon_2d.shape} (Inline, Xline)")
    except Exception as e:
        print(f"3D数据加载失败，错误信息: {e}")
        exit()

    # 检查层位和地震数据的Inline/Xline维度是否匹配
    if seismic_3d.shape[0] != horizon_2d.shape[0] or \
            seismic_3d.shape[1] != horizon_2d.shape[1]:
        print("错误: 3D地震数据和2D层位数据的Inline/Xline维度不匹配！请检查数据。")
        exit()

    print("\n开始3D强反射分离处理...")

    # --- 4. 执行3D匹配追踪 ---
    amplitude_3d = perform_matching_pursuit_3d(
        seismic_3d=seismic_3d,
        horizon_2d=horizon_2d,
        params=algorithm_params
    )
    print(f"匹配振幅体形状: {amplitude_3d.shape}")

    # --- 5. 执行3D强反射分离 ---
    final_seismic_3d, background_amp_3d, lambda_3d = separate_strong_reflection_3d(
        seismic_3d=seismic_3d,
        horizon_2d=horizon_2d,
        amplitude_3d=amplitude_3d,
        params=algorithm_params
    )
    print(f"分离后地震体形状: {final_seismic_3d.shape}")
    print(f"背景振幅体形状: {background_amp_3d.shape}")
    print(f"Lambda体形状: {lambda_3d.shape}")

    # --- 6. 保存3D结果 ---
    print("\n步骤4: 保存3D处理结果...")
    # np.save(os.path.join(OUTPUT_DIR, 'amplitude_3d.npy'), amplitude_3d)
    # np.save(os.path.join(OUTPUT_DIR, 'background_amplitude_3d.npy'), background_amp_3d)
    # np.save(os.path.join(OUTPUT_DIR, 'lambda_3d.npy'), lambda_3d)
    np.save(os.path.join(OUTPUT_DIR, OUTPUT_FILE_NAME), final_seismic_3d)
    print(f"所有3D结果已保存到目录: {OUTPUT_DIR}")

    # --- 7. 可视化某个切片的结果 ---
    print(f"\n步骤5: 可视化 {('Inline' if VIS_SLICE_AXIS == 0 else 'Xline')} {VIS_SLICE_INDEX} 切片结果...")
    vis_save_path = os.path.join(OUTPUT_DIR,
                                 f'separation_results_{("inline" if VIS_SLICE_AXIS == 0 else "xline")}_{VIS_SLICE_INDEX}.png')

    # 确保切片索引在有效范围内
    max_idx = seismic_3d.shape[VIS_SLICE_AXIS] - 1
    if not (0 <= VIS_SLICE_INDEX <= max_idx):
        print(f"警告: 可视化切片索引 {VIS_SLICE_INDEX} 超出范围。将使用 {max_idx // 2}。")
        VIS_SLICE_INDEX = max_idx // 2

    visualize_separation_results_3d_slice(
        original_3d=seismic_3d,
        final_3d=final_seismic_3d,
        horizon_2d=horizon_2d,
        amp_actual_3d=amplitude_3d,
        amp_bg_3d=background_amp_3d,
        lambda_3d=lambda_3d,
        slice_index=VIS_SLICE_INDEX,
        slice_axis=VIS_SLICE_AXIS,
        params=algorithm_params,
        save_path=vis_save_path
    )
    print("\n3D强反射分离处理及结果保存完成。")