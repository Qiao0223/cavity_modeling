import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

# 全局设置字体，解决乱码问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# =============================================================================
# 数据加载和滤波函数 (无需修改)
# =============================================================================
def load_data_from_npy_files(
        seismic_path: str,
        dip_il_path: str,
        dip_xl_path: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """从三个独立的 .npy 文件中加载3D数据。"""
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
        data_to_filter_slice: np.ndarray,
        dip_slice: np.ndarray,
        params: dict,
        filter_half_length: int = 4
) -> np.ndarray:
    """对单个2D剖面应用结构导向的均值滤波。"""
    n_samples, n_traces = data_to_filter_slice.shape
    dz = params['dz']
    dx = params.get('dx', params.get('dy'))
    shift_per_trace = (np.tan(np.deg2rad(dip_slice)) * dx) / dz
    filtered_slice = np.copy(data_to_filter_slice)
    trace_start, trace_end = filter_half_length, n_traces - filter_half_length
    with tqdm(total=n_samples * (trace_end - trace_start), desc="沿构造进行均值滤波") as pbar:
        for j in range(trace_start, trace_end):
            for k in range(n_samples):
                neighborhood_values = []
                for offset in range(-filter_half_length, filter_half_length + 1):
                    neighbor_j = j + offset
                    vertical_shift = offset * shift_per_trace[k, j]
                    neighbor_k = int(round(k + vertical_shift))
                    if 0 <= neighbor_k < n_samples:
                        neighborhood_values.append(data_to_filter_slice[neighbor_k, neighbor_j])
                if neighborhood_values:
                    filtered_slice[k, j] = np.mean(neighborhood_values)
                pbar.update(1)
    return filtered_slice


# =============================================================================
# 【核心修改区域 1】: 重写可视化函数为四图布局
# =============================================================================
def visualize_filtering_four_plots(
        original_attribute_slice,
        filtered_attribute_slice,
        seismic_slice,
        params,
        slice_info
):
    """
    以2x2的四图布局，可视化属性滤波效果，并加入原始地震剖面对比。
    (此版本已将“残差属性”和“原始地震”的位置对调)
    """
    # 1. 计算属性残差
    residual_attribute_slice = original_attribute_slice - filtered_attribute_slice

    # 2. 设置坐标轴和范围 (不变)
    dx_label = f"(道间距: {params.get('dx', params.get('dy'))} m)"
    xlabel = "Xline " + dx_label if 'dx' in params else "Inline " + dx_label
    x_extent = original_attribute_slice.shape[1] * params.get('dx', params.get('dy'))
    z_extent = original_attribute_slice.shape[0] * params['dz']
    plot_extent = [0, x_extent, z_extent, 0]

    # 3. 创建 2x2 图像布局 (不变)
    fig, axes = plt.subplots(2, 2, figsize=(20, 16), sharex=True, sharey=True)

    # 4. 计算各图的色标范围 (不变)
    vmin_attr = np.percentile(original_attribute_slice, 2)
    vmax_attr = np.percentile(original_attribute_slice, 98)
    vmax_seis = np.percentile(np.abs(seismic_slice), 99)
    vmax_resid = np.percentile(np.abs(residual_attribute_slice), 99)

    # --- 5. 绘制四个子图 (顺序调整) ---

    # -- 图 1 (左上): 原始属性 (不变) --
    im1 = axes[0, 0].imshow(original_attribute_slice, cmap='rainbow', aspect='auto',
                           vmin=vmin_attr, vmax=vmax_attr, extent=plot_extent)
    axes[0, 0].set_title(f"1. 原始属性 ({slice_info})")
    axes[0, 0].set_ylabel(f"深度 (m), dz={params['dz']} m")
    fig.colorbar(im1, ax=axes[0, 0], orientation='vertical', label="属性值")

    # -- 图 2 (右上): 滤波后属性 (不变) --
    im2 = axes[0, 1].imshow(filtered_attribute_slice, cmap='rainbow', aspect='auto',
                           vmin=vmin_attr, vmax=vmax_attr, extent=plot_extent)
    axes[0, 1].set_title("2. 滤波后属性 (背景模型)")
    fig.colorbar(im2, ax=axes[0, 1], orientation='vertical', label="属性值")

    # --- 【位置交换】 ---
    # -- 图 3 (左下): 残差属性 -- (原为地震剖面)
    im3 = axes[1, 0].imshow(residual_attribute_slice, cmap='seismic', aspect='auto',
                           vmin=-vmax_resid, vmax=vmax_resid, extent=plot_extent)
    axes[1, 0].set_title("3. 残差属性")
    axes[1, 0].set_ylabel(f"深度 (m), dz={params['dz']} m")
    axes[1, 0].set_xlabel(xlabel)
    fig.colorbar(im3, ax=axes[1, 0], orientation='vertical', label="残差值")

    # -- 图 4 (右下): 原始地震剖面 -- (原为残差属性)
    im4 = axes[1, 1].imshow(seismic_slice, cmap='seismic', aspect='auto',
                           vmin=-vmax_seis, vmax=vmax_seis, extent=plot_extent)
    axes[1, 1].set_title("4. 原始地震剖面 (参考)")
    axes[1, 1].set_xlabel(xlabel)
    fig.colorbar(im4, ax=axes[1, 1], orientation='vertical', label="振幅")
    # --- 【交换结束】 ---

    plt.suptitle(f"沿构造倾角对属性进行滤波 (四图对比)", fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# --- 主程序 ---
if __name__ == '__main__':
    # =============================================================================
    # 【核心修改区域 2】: 主程序逻辑保持不变，仅修改最后的可视化调用
    # =============================================================================

    # --- 1. 设置文件路径 (新增属性数据路径) ---
    SEISMIC_NPY_PATH = r'C:\Work\sunjie\Python\cavity_modeling\data\input_npy\luchang\LC_CUT.npy'
    # !!! 请将下方路径修改为您要滤波的属性数据 .npy 文件 !!!
    ATTRIBUTE_NPY_PATH = r'C:\Work\sunjie\Python\cavity_modeling\data\input_npy\luchang\LC_E_STA.npy' # 示例：暂时用地震数据代替属性数据
    DIP_IL_NPY_PATH = r'C:\Work\sunjie\Python\cavity_modeling\data\input_npy\luchang\luchang_inline_dip.npy'
    DIP_XL_NPY_PATH = r'C:\Work\sunjie\Python\cavity_modeling\data\input_npy\luchang\luchang_crossline_dip.npy'

    # --- 2. 定义数据采集参数 ---
    seismic_params_full = {'dz': 5.0, 'dy': 25, 'dx': 25}

    # --- 3. 定义处理目标和滤波参数 ---
    PROCESSING_AXIS = 1
    SLICE_INDEX = 188
    FILTER_HALF_LENGTH = 20
    DIP_SMOOTHING_SIGMA = [1, 1]

    # --- 4. 加载完整的3D数据 (包括属性数据) ---
    try:
        seismic_3d, dip_inline_3d, dip_xline_3d = load_data_from_npy_files(
            seismic_path=SEISMIC_NPY_PATH, dip_il_path=DIP_IL_NPY_PATH, dip_xl_path=DIP_XL_NPY_PATH)

        print(f"\n  - 从 {ATTRIBUTE_NPY_PATH} 加载属性数据...")
        attribute_3d = np.load(ATTRIBUTE_NPY_PATH)
        print(f"  - 属性数据形状: {attribute_3d.shape}")
        if attribute_3d.shape != seismic_3d.shape:
            raise ValueError("错误: 属性数据和地震数据的形状必须完全相同！")

    except Exception as e:
        print(f"数据加载失败，错误信息: {e}")
        exit()

    # --- 5. 提取所有需要的2D剖面 ---
    if PROCESSING_AXIS == 0:
        slice_type_str = "Inline"
        seismic_2d_slice = seismic_3d[SLICE_INDEX, :, :].T
        attribute_2d_slice = attribute_3d[SLICE_INDEX, :, :].T
        dip_2d_slice = dip_inline_3d[SLICE_INDEX, :, :].T
        params_2d = {'dz': seismic_params_full['dz'], 'dx': seismic_params_full['dx']}
    elif PROCESSING_AXIS == 1:
        slice_type_str = "Xline"
        seismic_2d_slice = seismic_3d[:, SLICE_INDEX, :].T
        attribute_2d_slice = attribute_3d[:, SLICE_INDEX, :].T
        dip_2d_slice = dip_xline_3d[:, SLICE_INDEX, :].T
        params_2d = {'dz': seismic_params_full['dz'], 'dy': seismic_params_full['dy']}
    else:
        print(f"错误: 无效的 PROCESSING_AXIS: {PROCESSING_AXIS}。请选择 0 或 1。")
        exit()

    print(f"\n已提取 {slice_type_str} 剖面 {SLICE_INDEX} 用于处理。")

    # --- 6. 对倾角剖面进行平滑 (逻辑不变) ---
    print(f"\n正在对倾角剖面进行高斯滤波平滑 (sigma={DIP_SMOOTHING_SIGMA})...")
    dip_2d_slice_smoothed = gaussian_filter(dip_2d_slice, sigma=DIP_SMOOTHING_SIGMA)
    print("倾角剖面平滑完成！")
    # (可视化倾角的部分保持不变)

    # --- 7. 调用均值滤波函数，对属性剖面进行滤波 ---
    print("\n正在沿构造方向对【属性】数据进行滤波...")
    filtered_attribute_2d = structure_oriented_mean_filter_2d(
        data_to_filter_slice=attribute_2d_slice, # 滤波对象是属性
        dip_slice=dip_2d_slice_smoothed,
        params=params_2d,
        filter_half_length=FILTER_HALF_LENGTH
    )
    print("属性滤波完成！")

    # --- 8. 调用新的四图可视化函数 ---
    visualize_filtering_four_plots(
        original_attribute_slice=attribute_2d_slice,
        filtered_attribute_slice=filtered_attribute_2d,
        seismic_slice=seismic_2d_slice, # 将地震剖面也传入
        params=params_2d,
        slice_info=f"{slice_type_str} {SLICE_INDEX}"
    )