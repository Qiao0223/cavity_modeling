# -*- coding: utf-8 -*-
"""
一个完整的三维数据单剖面二维VMD处理与可视化脚本 (实验性方法 - 绘图已修复)。

此版本根据用户要求，实现“拼接-分解-重塑”的策略。
【【【 关键修复 】】】
修正了 visualize_vmd_comparison 函数中的 UnboundLocalError bug。
"""
import numpy as np
import time
import matplotlib.pyplot as plt
from vmdpy import VMD


# =============================================================================
# 核心VMD算法封装 (新策略：拼接-分解-重塑)
# =============================================================================
def vmd_2d_flattened(signal_2d, alpha, tau, K, DC, init, tol, verbose=False):
    """
    对二维剖面应用“拼接-分解-重塑 (Flatten-VMD-Reshape)”策略。
    """
    original_shape = signal_2d.shape
    rows, cols = original_shape

    if verbose:
        print(f"--- 正在使用“拼接-分解-重塑”策略 ---")
        print(f"    原始剖面形状: {original_shape}")

    signal_1d = signal_2d.flatten()
    print(f"    展平后的一维信号长度: {signal_1d.shape[0]}")

    if signal_1d.shape[0] % 2 != 0:
        signal_1d = signal_1d[:-1]
        print("    信号长度为奇数，已截断为偶数长度。")

    print("    正在对拼接后的一维信号进行VMD分解...")
    u, u_hat, omega = VMD(signal_1d, alpha, tau, K, DC, init, tol)
    print("    VMD分解完成。")

    decomposed_modes_3d = np.zeros((K, rows, cols), dtype=np.float32)
    reshapable_length = u.shape[1]

    for k in range(K):
        mode_1d = u[k, :]
        flat_part = mode_1d.reshape(-1)
        # 使用 .flat 可以安全地填充，即使长度不完全匹配
        decomposed_modes_3d[k].flat[:reshapable_length] = flat_part

    print("    已将分解出的模态恢复为二维形状。")
    return decomposed_modes_3d


# =============================================================================
# 可视化函数 (已修复Bug)
# =============================================================================
def visualize_vmd_comparison(original_slice, decomposed_modes, slice_index, processing_axis):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    K = decomposed_modes.shape[0]
    num_plots = K + 1
    if K + 1 <= 9:
        ncols = 3
    else:
        ncols = 4
    nrows = int(np.ceil(num_plots / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    axes = axes.flatten()

    if processing_axis == 1:
        slice_type, xlabel = "Inline", "Inline道号"
    elif processing_axis == 0:
        slice_type, xlabel = "Crossline", "Crossline道号"
    else:
        slice_type, xlabel = "Time/Depth", "Inline道号"; plt.ylabel("Crossline道号", fontsize=12)

    vmax_original = np.percentile(np.abs(original_slice), 98)

    # --- 绘制原始剖面 ---
    im = axes[0].imshow(original_slice.T, cmap='seismic', aspect='auto', vmin=-vmax_original, vmax=vmax_original)
    axes[0].set_title("原始剖面 (归一化后)", fontsize=14)
    axes[0].set_ylabel("时间采样点", fontsize=12)
    axes[0].set_xlabel(xlabel, fontsize=12)

    # --- 【【【【【 Bug修复处 】】】】】 ---
    # 将 colorbar 与正确的 axes (axes[0]) 关联
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    # ------------------------------------

    # --- 绘制分解后的模态 ---
    for i in range(K):
        ax = axes[i + 1]  # ax在这里才被定义
        mode_slice = decomposed_modes[i, :, :]
        vmax_mode = np.percentile(np.abs(mode_slice), 99) if np.any(mode_slice) else 1.0
        if vmax_mode < 1e-9: vmax_mode = np.max(np.abs(mode_slice)) if np.any(mode_slice) else 1.0

        im_mode = ax.imshow(mode_slice.T, cmap='seismic', aspect='auto', vmin=-vmax_mode, vmax=vmax_mode)
        ax.set_title(f"分解模态 {i + 1}", fontsize=14)
        ax.set_xlabel(xlabel, fontsize=12)
        fig.colorbar(im_mode, ax=ax, fraction=0.046, pad=0.04)

    # --- 关闭多余的子图 ---
    for i in range(num_plots, len(axes)): axes[i].axis('off')

    plt.suptitle(f"VMD单剖面分解效果 ({slice_type} 剖面: {slice_index})", fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# =============================================================================
# 主函数 (程序的入口)
# =============================================================================
def main():
    # ==================== 用户参数设置区域 ====================
    input_file_path = r'C:\Work\sunjie\Python\cavity_modeling\data\input_npy\fuyuan3_crop.npz'
    processing_axis = 1
    slice_to_preview = 500

    # VMD核心参数
    alpha = 2000
    tau = 0.
    K = 3
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
        print(f"错误: 加载文件时出错: {e}"); return

    print("\n--- 步骤 2: 对数据进行归一化处理 ---")
    data_max_abs = np.max(np.abs(D_3d))
    if data_max_abs == 0: print("错误：数据全为零。"); return
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

    print(f"\n--- 步骤 4: 开始对选定剖面进行【拼接-VMD-重塑】分解 (K={K}) ---")
    start_time_2d = time.time()

    decomposed_modes = vmd_2d_flattened(
        original_slice, alpha, tau, K, DC, init, tol, verbose=True
    )
    print(f"单剖面分解耗时: {time.time() - start_time_2d:.2f} 秒。")

    print("\n--- 步骤 5: 显示分解结果... ---")
    visualize_vmd_comparison(original_slice, decomposed_modes, slice_to_preview, processing_axis)

    print("\n脚本运行结束。")


if __name__ == "__main__":
    main()