import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

# 全局设置字体，解决乱码问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# =============================================================================
# 数据加载和滤波函数保持不变
# =============================================================================
def load_data_from_npy_files(
        seismic_path: str,
        dip_il_path: str,
        dip_xl_path: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """从三个独立的 .npy 文件中加载3D数据。 (此函数无需修改)"""
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


def structure_oriented_mean_filter_2d(
        seismic_slice: np.ndarray,
        dip_slice: np.ndarray,
        params: dict,
        filter_half_length: int = 4
) -> np.ndarray:
    """对单个2D剖面应用结构导向的均值滤波。(此函数无需修改)"""
    n_samples, n_traces = seismic_slice.shape
    dz = params['dz']
    dx = params.get('dx', params.get('dy'))
    print("\n步骤1 (2D): 将倾角转换为采样点偏移量...")
    dip_rad = np.deg2rad(dip_slice)
    gradient = np.tan(dip_rad)
    shift_per_trace = (gradient * dx) / dz
    filtered_slice = np.copy(seismic_slice)
    print("步骤2 (2D): 应用结构导向均值滤波...")
    trace_start, trace_end = filter_half_length, n_traces - filter_half_length
    with tqdm(total=n_samples * (trace_end - trace_start), desc="均值滤波2D剖面") as pbar:
        for j in range(trace_start, trace_end):
            for k in range(n_samples):
                neighborhood_values = []
                for offset in range(-filter_half_length, filter_half_length + 1):
                    neighbor_j = j + offset
                    vertical_shift = offset * shift_per_trace[k, j]
                    neighbor_k = int(round(k + vertical_shift))
                    if 0 <= neighbor_k < n_samples:
                        neighborhood_values.append(seismic_slice[neighbor_k, neighbor_j])
                if neighborhood_values:
                    filtered_slice[k, j] = np.mean(neighborhood_values)
                pbar.update(1)
    print("2D剖面均值滤波完成！")
    return filtered_slice


def visualize_results_slice_2d(
        original_slice, background_slice, params, slice_info
):
    """
    可视化单个2D剖面的滤波效果。
    (此版本已修改，为残差图使用独立的、自动调整的色标)
    """
    # 1. 计算残差
    residual_slice = original_slice - background_slice

    # 2. 设置坐标轴和范围 (与原代码相同)
    dx_label = f"(道间距: {params.get('dx', params.get('dy'))} m)"
    xlabel = "Xline " + dx_label if 'dx' in params else "Inline " + dx_label
    x_extent = original_slice.shape[1] * params.get('dx', params.get('dy'))
    z_extent = original_slice.shape[0] * params['dz']
    plot_extent = [0, x_extent, z_extent, 0]

    # 3. 创建图像布局
    fig, axes = plt.subplots(1, 3, figsize=(24, 10), sharex=True, sharey=True)

    # --- 修改核心 ---
    # 4. 为原始图和背景图设置色标
    #    这个色标基于原始数据的范围
    vmax_orig = np.percentile(np.abs(original_slice), 99)

    # 5. (新增) 为残差图单独设置色标
    #    这个色标基于残差数据自身的范围
    vmax_residual = np.percentile(np.abs(residual_slice), 99)
    # --- 修改结束 ---

    # 6. 绘制三个子图
    # 绘制 1: 原始剖面 (使用原始数据色标)
    im1 = axes[0].imshow(original_slice, cmap='magma', aspect='auto', vmin=-vmax_orig, vmax=vmax_orig,
                         extent=plot_extent)
    axes[0].set_title(f"1. 原始剖面 ({slice_info})")
    axes[0].set_ylabel(f"深度 (m), dz={params['dz']} m")
    axes[0].set_xlabel(xlabel)
    fig.colorbar(im1, ax=axes[0], orientation='vertical', fraction=0.05, pad=0.04)  # 添加色标

    # 绘制 2: 背景模型 (使用原始数据色标，保持对比一致性)
    im2 = axes[1].imshow(background_slice, cmap='magma', aspect='auto', vmin=-vmax_orig, vmax=vmax_orig,
                         extent=plot_extent)
    axes[1].set_title("2. 背景模型")
    axes[1].set_xlabel(xlabel)
    fig.colorbar(im2, ax=axes[1], orientation='vertical', fraction=0.05, pad=0.04)  # 添加色标

    # 绘制 3: 残差 (使用残差数据自身的色标)
    im3 = axes[2].imshow(residual_slice, cmap='seismic', aspect='auto', vmin=-vmax_residual, vmax=vmax_residual,
                         extent=plot_extent)
    axes[2].set_title("3. 残差 (自动调整色标)")
    axes[2].set_xlabel(xlabel)
    fig.colorbar(im3, ax=axes[2], orientation='vertical', fraction=0.05, pad=0.04)  # 添加色标

    plt.suptitle(f"倾角匹配的滤波效果 ({slice_info})", fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# --- 主程序 ---
if __name__ == '__main__':
    # --- 1. 设置文件路径 ---
    SEISMIC_NPY_PATH = r'C:\Work\sunjie\Python\cavity_modeling\data\input_npy\luchang\LC_CUT.npy'
    DIP_IL_NPY_PATH = r'C:\Work\sunjie\Python\cavity_modeling\data\input_npy\luchang\luchang_inline_dip.npy'
    DIP_XL_NPY_PATH = r'C:\Work\sunjie\Python\cavity_modeling\data\input_npy\luchang\luchang_crossline_dip.npy'

    # --- 2. 定义数据采集参数 ---
    seismic_params_full = {'dz': 5.0, 'dy': 12.5, 'dx': 12.5}

    # --- 3. 定义处理目标和滤波参数 ---
    PROCESSING_AXIS = 1
    SLICE_INDEX = 200
    FILTER_HALF_LENGTH = 20
    DIP_SMOOTHING_SIGMA = [1.5, 7.5]

    # --- 4. 加载完整的3D数据 ---
    try:
        seismic_3d, dip_inline_3d, dip_xline_3d = load_data_from_npy_files(
            seismic_path=SEISMIC_NPY_PATH, dip_il_path=DIP_IL_NPY_PATH, dip_xl_path=DIP_XL_NPY_PATH)
    except Exception as e:
        print(f"数据加载失败，错误信息: {e}")
        exit()

    # --- 5. 提取单个2D剖面进行处理 ---
    if PROCESSING_AXIS == 0:
        slice_type_str = "Inline"
    elif PROCESSING_AXIS == 1:
        slice_type_str = "Xline"
    else:
        print(f"错误: 无效的 PROCESSING_AXIS: {PROCESSING_AXIS}。请选择 0 或 1。")
        exit()

    print(f"\n正在提取 {slice_type_str} 剖面 {SLICE_INDEX} (axis={PROCESSING_AXIS})，并使用倾角进行匹配...")

    if PROCESSING_AXIS == 0:
        seismic_2d_slice = seismic_3d[SLICE_INDEX, :, :].T
        print("    为 Inline 剖面匹配了 Inline 倾角")
        dip_2d_slice = dip_inline_3d[SLICE_INDEX, :, :].T
        params_2d = {'dz': seismic_params_full['dz'], 'dx': seismic_params_full['dx']}
    elif PROCESSING_AXIS == 1:
        seismic_2d_slice = seismic_3d[:, SLICE_INDEX, :].T
        print("    为 Xline 剖面匹配了 Xline 倾角")
        dip_2d_slice = dip_xline_3d[:, SLICE_INDEX, :].T
        params_2d = {'dz': seismic_params_full['dz'], 'dy': seismic_params_full['dy']}

    # --- 6. 对提取出的(错误匹配的)倾角剖面进行平滑 ---
    print(f"\n正在对倾角剖面进行高斯滤波平滑 (sigma={DIP_SMOOTHING_SIGMA})...")
    dip_2d_slice_smoothed = gaussian_filter(dip_2d_slice, sigma=DIP_SMOOTHING_SIGMA)
    print("倾角剖面平滑完成！")

    # (可视化倾角的部分保持不变)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), sharey=True)
    dip_vmax = np.percentile(np.abs(dip_2d_slice), 98)
    ax1.imshow(dip_2d_slice, cmap='rainbow', aspect='auto', vmin=-dip_vmax, vmax=dip_vmax)
    ax1.set_title("匹配的原始倾角剖面")
    ax1.set_xlabel("道号")
    ax1.set_ylabel("采样点")
    ax2.imshow(dip_2d_slice_smoothed, cmap='rainbow', aspect='auto', vmin=-dip_vmax, vmax=dip_vmax)
    ax2.set_title("平滑后的倾角剖面 (用于滤波)")
    ax2.set_xlabel("道号")
    plt.suptitle("倾角数据平滑效果对比", fontsize=16)
    plt.show()

    # --- 7. 调用均值滤波函数 ---
    background_model_2d = structure_oriented_mean_filter_2d(
        seismic_slice=seismic_2d_slice,
        dip_slice=dip_2d_slice_smoothed,
        params=params_2d,
        filter_half_length=FILTER_HALF_LENGTH
    )

    np.save("sof.npy", seismic_2d_slice - background_model_2d)

    # --- 8. 可视化2D结果 ---
    visualize_results_slice_2d(
        seismic_2d_slice, background_model_2d, params_2d, slice_info=f"{slice_type_str} {SLICE_INDEX}")