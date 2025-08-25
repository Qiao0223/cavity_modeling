# -*- coding: utf-8 -*-
"""
一个完整的三维数据单剖面二维VMD处理与可视化脚本 (最终版 - 使用vmdpy)。

此版本解决了 `libvmd` 库的内部bug，切换到更稳定、广泛使用的 `vmdpy` 库。
采用“逐列VMD”策略，对二维剖面的每一道数据进行一维VMD分解，然后重组成二维模态。

工作流：
- 用户友好的参数配置区。
- 清晰的安装指引 (poetry remove libvmd, poetry add vmdpy)。
- 模块化的函数设计，包含新的 `vmd_2d_columnwise` 函数。
- 高质量的可视化对比图。
"""
import numpy as np
import time
import matplotlib.pyplot as plt
# 【【【【【 导入新的、正确的库 】】】】】
from vmdpy import VMD


# =============================================================================
# 核心VMD算法封装 (新策略：逐列处理)
# =============================================================================
def vmd_2d_columnwise(signal_2d, alpha, tau, K, DC, init, tol, verbose=False):
    """
    对二维剖面应用“逐列VMD (Column-wise VMD)”。
    该函数遍历二维数据的每一列，将其作为一维信号进行VMD分解，
    然后将所有列的结果重新组合成K个二维模态。
    """
    rows, cols = signal_2d.shape
    # 准备一个三维数组来存储所有分解出的模态
    # 形状为 (模态数, 时间采样点, 道数)
    decomposed_modes_3d = np.zeros((K, rows, cols), dtype=np.float32)

    if verbose:
        print(f"    开始对 {rows}x{cols} 的剖面进行逐列VMD分解...")

    # 遍历每一列（每一道数据）
    for i in range(cols):
        # 打印进度
        if verbose and (i % 100 == 0 or i == cols - 1):
            print(f"    正在处理第 {i + 1}/{cols} 道...")

        column_1d = signal_2d[:, i]

        # 对当前这一维的列信号执行VMD分解
        # vmdpy的VMD函数返回: 模态, 模态的频谱, 中心频率
        u, u_hat, omega = VMD(column_1d, alpha, tau, K, DC, init, tol)

        # 将分解出的 K 个一维模态存入结果数组的相应位置
        decomposed_modes_3d[:, :, i] = u

    if verbose:
        print("    逐列VMD分解完成。")

    # 返回重组后的K个二维模态
    return decomposed_modes_3d


# =============================================================================
# 可视化函数 (无需修改)
# =============================================================================
def visualize_vmd_comparison(original_slice, decomposed_modes, slice_index, display_axis=1):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    K = decomposed_modes.shape[0]
    num_plots = K + 1
    ncols = int(np.ceil(np.sqrt(num_plots)))
    nrows = int(np.ceil(num_plots / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False)
    axes = axes.flatten()

    slice_type, xlabel = ("Inline", "Crossline道号") if display_axis == 1 else ("Crossline", "Inline道号")
    vmax_original = np.percentile(np.abs(original_slice), 98)

    im = axes[0].imshow(original_slice.T, cmap='seismic', aspect='auto', vmin=-vmax_original, vmax=vmax_original)
    axes[0].set_title("原始剖面 (归一化后)", fontsize=14)
    axes[0].set_ylabel("时间采样点", fontsize=12)
    axes[0].set_xlabel(xlabel, fontsize=12)
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    for i in range(K):
        ax = axes[i + 1]
        mode_slice = decomposed_modes[i, :, :]
        vmax_mode = np.percentile(np.abs(mode_slice), 99) if np.any(mode_slice) else 1.0
        if vmax_mode < 1e-9: vmax_mode = np.max(np.abs(mode_slice))

        im_mode = ax.imshow(mode_slice.T, cmap='seismic', aspect='auto', vmin=-vmax_mode, vmax=vmax_mode)
        ax.set_title(f"分解模态 {i + 1}", fontsize=14)
        ax.set_xlabel(xlabel, fontsize=12)
        fig.colorbar(im_mode, ax=ax, fraction=0.046, pad=0.04)

    for i in range(num_plots, len(axes)):
        axes[i].axis('off')

    plt.suptitle(f"VMD单剖面分解效果 ({slice_type} 剖面: {slice_index})", fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# =============================================================================
# 主函数 (程序的入口)
# =============================================================================
def main():
    # ==================== 用户参数设置区域 ====================
    input_file_path = r'C:\Work\sunjie\Python\cavity_modeling\data\input_npy\yingxi_crop.npz'
    processing_axis = 1
    slice_to_preview = 500

    # VMD核心参数 (vmdpy库使用这些参数)
    alpha = 1000
    tau = 0.
    K = 8
    DC = 0
    init = 1
    tol = 1e-7
    # ============================================================

    print(f"--- 步骤 1: 从 '{input_file_path}' 加载数据 ---")
    try:
        if input_file_path.endswith('.npz'):
            with np.load(input_file_path) as data:
                D_3d = data["data"]
        else:
            D_3d = np.load(input_file_path)
        if D_3d.ndim != 3: raise ValueError("加载的数据不是三维的。")
        print(f"数据加载成功。原始形状: {D_3d.shape}, 数据类型: {D_3d.dtype}")
    except Exception as e:
        print(f"错误: 加载文件时出错: {e}");
        return

    print("\n--- 步骤 2: 对数据进行归一化处理 ---")
    data_max_abs = np.max(np.abs(D_3d))
    if data_max_abs == 0:
        print("错误：数据全为零。");
        return
    D_3d_normalized = D_3d / data_max_abs
    print(f"归一化完成。数据范围: [-1, 1]")

    print(f"\n--- 步骤 3: 提取剖面 #{slice_to_preview} (沿 axis={processing_axis}) ---")
    try:
        original_slice = np.take(D_3d_normalized, slice_to_preview, axis=processing_axis)
        print(f"剖面提取成功，形状: {original_slice.shape}")
    except IndexError:
        max_idx = D_3d.shape[processing_axis] - 1
        print(f"错误: 剖面索引 {slice_to_preview} 超出范围 (有效范围 0-{max_idx})。");
        return

    print(f"\n--- 步骤 4: 开始对选定剖面进行VMD分解 (K={K}) ---")
    start_time_2d = time.time()

    # 【【【【【 调用新的逐列处理函数 】】】】】
    # 注意：这个函数只返回重组后的模态
    decomposed_modes = vmd_2d_columnwise(
        original_slice, alpha, tau, K, DC, init, tol, verbose=True
    )
    print(f"单剖面VMD分解耗时: {time.time() - start_time_2d:.2f} 秒。")

    print("\n--- 步骤 5: 显示分解结果... ---")
    visualize_vmd_comparison(original_slice, decomposed_modes, slice_to_preview, processing_axis)

    print("\n脚本运行结束。")


if __name__ == "__main__":
    main()