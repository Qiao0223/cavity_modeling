import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
import os

# 关键引入：Numba用于即时编译和并行加速
from numba import njit, prange

# 全局设置字体，解决乱码问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# =============================================================================
# 数据加载和可视化函数保持不变
# =============================================================================
def load_data_from_npy_files(
        seismic_path: str,
        dip_il_path: str,
        dip_xl_path: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """从三个独立的 .npy 文件中加载3D数据。(此函数无需修改)"""
    print(f"  - 从 {seismic_path} 加载地震数据...")
    seismic_data = np.load(seismic_path)
    print(f"  - 从 {dip_il_path} 加载Inline倾角...")
    dip_inline = np.load(dip_il_path)
    print(f"  - 从 {dip_xl_path} 加载Xline倾角...")
    dip_xline = np.load(dip_xl_path)
    print("\n数据加载完成:")
    print(f"  - 地震数据形状 (Inlines, Xlines, Samples): {seismic_data.shape}")
    print(f"  - Inline倾角形状: {dip_inline.shape}")
    print(f"  - Xline倾角形状: {dip_xline.shape}")
    if not (seismic_data.shape == dip_inline.shape == dip_xline.shape):
        raise ValueError("错误: 三个数据体的形状必须完全相同！")
    return seismic_data, dip_inline, dip_xline


def visualize_results_slice_2d(
        original_slice, background_slice, params, slice_info
):
    """可视化单个2D剖面的滤波效果。(此函数无需修改)"""
    residual_slice = original_slice - background_slice
    slice_axis = slice_info.get('axis', 1)
    if slice_axis == 1:
        dx_label = f"(道间距: {params.get('dy')} m)"
        xlabel = "Inline " + dx_label
        x_extent = original_slice.shape[1] * params.get('dy')
    else:
        dx_label = f"(道间距: {params.get('dx')} m)"
        xlabel = "Xline " + dx_label
        x_extent = original_slice.shape[1] * params.get('dx')

    z_extent = original_slice.shape[0] * params['dz']
    plot_extent = [0, x_extent, z_extent, 0]
    fig, axes = plt.subplots(1, 3, figsize=(24, 10), sharex=True, sharey=True)
    vmax = np.percentile(np.abs(original_slice), 99)
    axes[0].imshow(original_slice, cmap='seismic', aspect='auto', vmin=-vmax, vmax=vmax, extent=plot_extent)
    axes[0].set_title(f"1. 原始剖面 ({slice_info.get('name')})")
    axes[0].set_ylabel(f"深度 (m), dz={params['dz']} m")
    axes[0].set_xlabel(xlabel)
    axes[1].imshow(background_slice, cmap='seismic', aspect='auto', vmin=-vmax, vmax=vmax, extent=plot_extent)
    axes[1].set_title("2. 背景模型 (3D滤波结果)")
    axes[1].set_xlabel(xlabel)
    axes[2].imshow(residual_slice, cmap='seismic', aspect='auto', vmin=-vmax, vmax=vmax, extent=plot_extent)
    axes[2].set_title("3. 残差")
    axes[2].set_xlabel(xlabel)
    plt.suptitle(f"三维结构导向滤波效果展示 ({slice_info.get('name')})", fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# =============================================================================
# 核心功能：三维滤波函数 (重构为带进度的版本)
# =============================================================================
@njit(parallel=True)
def _core_filter_loop_numba(
        seismic_3d: np.ndarray,
        filtered_volume: np.ndarray,
        shift_per_il_step: np.ndarray,
        shift_per_xl_step: np.ndarray,
        n_xl: int, n_samples: int,
        il_half_len: int, xl_half_len: int,
        start_il: int, end_il: int  # 新增：定义此函数处理的Inline范围
):
    """
    (内部函数) Numba JIT加速的核心滤波循环。
    此函数处理一个Inline方向的“数据块”，以便于外部显示进度。
    """
    # prange会在此函数接收到的start_il和end_il范围内并行执行
    for i in prange(start_il, end_il):
        for j in range(xl_half_len, n_xl - xl_half_len):
            for k in range(n_samples):
                neighborhood_values = []
                for offset_il in range(-il_half_len, il_half_len + 1):
                    for offset_xl in range(-xl_half_len, xl_half_len + 1):
                        neighbor_i = i + offset_il
                        neighbor_j = j + offset_xl

                        total_vertical_shift = (offset_il * shift_per_il_step[i, j, k]) + \
                                               (offset_xl * shift_per_xl_step[i, j, k])

                        neighbor_k = int(round(k + total_vertical_shift))

                        if 0 <= neighbor_k < n_samples:
                            neighborhood_values.append(seismic_3d[neighbor_i, neighbor_j, neighbor_k])

                if neighborhood_values:
                    # 将结果直接写入传入的数组中
                    filtered_volume[i, j, k] = np.mean(np.array(neighborhood_values))


def structure_oriented_mean_filter_3d_with_progress(
        seismic_3d: np.ndarray,
        dip_il_smoothed: np.ndarray,
        dip_xl_smoothed: np.ndarray,
        dz: float, dy: float, dx: float,
        il_half_len: int, xl_half_len: int,
        chunk_size: int = 10  # 定义每次送入Numba函数处理的Inline数量
) -> np.ndarray:
    """
    对3D数据体应用结构导向滤波，并显示Tqdm进度条。
    """
    n_il, n_xl, n_samples = seismic_3d.shape
    # 初始化结果数组，并复制边缘值，因为核心循环不会处理它们
    filtered_volume = np.copy(seismic_3d)

    print("  - 预计算倾角转换参数...")
    shift_per_il_step = np.tan(np.deg2rad(dip_il_smoothed)) * dy / dz
    shift_per_xl_step = np.tan(np.deg2rad(dip_xl_smoothed)) * dx / dz

    # 定义主循环的处理范围
    main_loop_start = il_half_len
    main_loop_end = n_il - il_half_len

    print(f"  - 开始分块并行处理 {main_loop_end - main_loop_start} 个 Inlines...")
    # 使用tqdm包装Python级别的循环，以显示进度
    with tqdm(total=main_loop_end - main_loop_start, desc="3D滤波进度") as pbar:
        # 按块(chunk)循环，将数据块提交给Numba函数处理
        for start_il in range(main_loop_start, main_loop_end, chunk_size):
            end_il = min(start_il + chunk_size, main_loop_end)

            # 调用高性能的Numba核心函数处理当前数据块
            _core_filter_loop_numba(
                seismic_3d, filtered_volume,
                shift_per_il_step, shift_per_xl_step,
                n_xl, n_samples,
                il_half_len, xl_half_len,
                start_il, end_il
            )
            pbar.update(end_il - start_il)  # 更新进度条

    return filtered_volume


# --- 主程序 ---
if __name__ == '__main__':
    # --- 1. 设置文件路径 ---
    BASE_PATH = r'C:\Work\sunjie\Python\cavity_modeling\data'
    SEISMIC_NPY_PATH = r"C:\Work\sunjie\Python\cavity_modeling\data\input_npy\luchang\LC_CUT.npy"
    DIP_IL_NPY_PATH = r"C:\Work\sunjie\Python\cavity_modeling\data\input_npy\luchang\luchang_inline_dip.npy"
    DIP_XL_NPY_PATH = r"C:\Work\sunjie\Python\cavity_modeling\data\input_npy\luchang\luchang_crossline_dip.npy"

    # --- 新增: 定义输出路径 ---
    OUTPUT_DIR = r"C:\Work\sunjie\Python\cavity_modeling\data\output_npy"
    os.makedirs(OUTPUT_DIR, exist_ok=True)  # 确保输出目录存在
    BACKGROUND_SAVE_PATH = os.path.join(OUTPUT_DIR, 'E_background.npy')
    RESIDUAL_SAVE_PATH = os.path.join(OUTPUT_DIR, 'E_residual.npy')

    # --- 2. 定义数据采集参数 ---
    seismic_params_full = {'dz': 5.0, 'dy': 25, 'dx': 25}

    # --- 3. 定义3D滤波参数 ---
    FILTER_IL_HALF_LENGTH = 25
    FILTER_XL_HALF_LENGTH = 25
    DIP_SMOOTHING_SIGMA_3D = [3.0, 3.0, 1.5]

    # --- 4. 加载完整的3D数据 ---
    try:
        seismic_3d, dip_inline_3d, dip_xline_3d = load_data_from_npy_files(
            seismic_path=SEISMIC_NPY_PATH, dip_il_path=DIP_IL_NPY_PATH, dip_xl_path=DIP_XL_NPY_PATH)
    except Exception as e:
        print(f"数据加载失败，错误信息: {e}")
        exit()

    # --- 5. 对完整的3D倾角数据体进行平滑 ---
    print(f"\n正在对3D倾角数据体进行高斯滤波平滑 (sigma={DIP_SMOOTHING_SIGMA_3D})...")
    dip_inline_3d_smoothed = gaussian_filter(dip_inline_3d, sigma=DIP_SMOOTHING_SIGMA_3D)
    dip_xline_3d_smoothed = gaussian_filter(dip_xline_3d, sigma=DIP_SMOOTHING_SIGMA_3D)
    print("3D倾角数据体平滑完成！")

    # --- 6. 调用带进度的三维均值滤波函数 ---
    print("\n开始执行三维结构导向均值滤波...")
    print(f"  - 滤波窗口大小: Inlines={2 * FILTER_IL_HALF_LENGTH + 1}, Xlines={2 * FILTER_XL_HALF_LENGTH + 1}")

    background_model_3d = structure_oriented_mean_filter_3d_with_progress(
        seismic_3d=seismic_3d,
        dip_il_smoothed=dip_inline_3d_smoothed,
        dip_xl_smoothed=dip_xline_3d_smoothed,
        dz=seismic_params_full['dz'],
        dy=seismic_params_full['dy'],
        dx=seismic_params_full['dx'],
        il_half_len=FILTER_IL_HALF_LENGTH,
        xl_half_len=FILTER_XL_HALF_LENGTH
    )
    print("三维滤波完成！")

    # --- 7. 新增: 计算残差并保存结果 ---
    print("\n正在计算残差并保存结果...")
    residual_3d = seismic_3d - background_model_3d

    try:
        if BACKGROUND_SAVE_PATH:
            print(f"  - 保存背景模型到: {BACKGROUND_SAVE_PATH}")
            np.save(BACKGROUND_SAVE_PATH, background_model_3d)
        if RESIDUAL_SAVE_PATH:
            print(f"  - 保存残差数据体到: {RESIDUAL_SAVE_PATH}")
            np.save(RESIDUAL_SAVE_PATH, residual_3d)
        print("数据保存成功！")
    except IOError as e:
        print(f"错误：无法保存文件。原因: {e}")

    # --- 8. 提取剖面并进行可视化，以检验3D滤波效果 ---
    AXIS_TO_VIEW = 1
    SLICE_INDEX_TO_VIEW = 500

    print(f"\n从三维结果中提取剖面 (Axis={AXIS_TO_VIEW}, Index={SLICE_INDEX_TO_VIEW}) 进行可视化...")

    if AXIS_TO_VIEW == 0:
        slice_type_str = "Inline"
        original_slice_2d = seismic_3d[SLICE_INDEX_TO_VIEW, :, :].T
        filtered_slice_2d = background_model_3d[SLICE_INDEX_TO_VIEW, :, :].T
        params_2d_vis = {'dz': seismic_params_full['dz'], 'dx': seismic_params_full['dx']}

    elif AXIS_TO_VIEW == 1:
        slice_type_str = "Xline"
        original_slice_2d = seismic_3d[:, SLICE_INDEX_TO_VIEW, :].T
        filtered_slice_2d = background_model_3d[:, SLICE_INDEX_TO_VIEW, :].T
        params_2d_vis = {'dz': seismic_params_full['dz'], 'dy': seismic_params_full['dy']}
    else:
        print(f"错误: 无效的 AXIS_TO_VIEW: {AXIS_TO_VIEW}。请选择 0 或 1。")
        exit()

    visualize_results_slice_2d(
        original_slice=original_slice_2d,
        background_slice=filtered_slice_2d,
        params=params_2d_vis,
        slice_info={'name': f'{slice_type_str} {SLICE_INDEX_TO_VIEW}', 'axis': AXIS_TO_VIEW}
    )