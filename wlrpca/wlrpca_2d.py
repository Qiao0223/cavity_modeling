# -*- coding: utf-8 -*-
"""
一个用于从三维地震数据体中提取单个二维剖面进行WLRPCA处理与可视化的脚本。

修改版特性：
- 读取三维 .npy 或 .npz 文件，假定数据形状为 (inline, xline, dt)。
- 用户可以指定任意一个剖面 (Inline, Crossline 或 Time Slice) 进行二维处理和分析。
- 只进行处理和可视化，不进行完整三维处理，也不保存任何结果文件。
- 用户可以自定义最大迭代次数。
"""
import numpy as np
import time
import matplotlib.pyplot as plt
from numpy.linalg import svd, norm


# =============================================================================
# 核心二维WLRPCA算法 (此部分无需修改)
# =============================================================================
def logarithmic_threshold(matrix, lambda_val, beta):
    """根据论文中的公式(11)实现对数阈值函数"""
    x0 = np.sqrt(2 * lambda_val - beta) if (2 * lambda_val - beta) > 0 else 0
    result = np.zeros_like(matrix, dtype=float)
    pos_mask = matrix > x0
    if np.any(pos_mask):
        x_pos = matrix[pos_mask]
        term_inside_sqrt = (x_pos + beta) ** 2 - 4 * lambda_val
        term_inside_sqrt[term_inside_sqrt < 0] = 0
        result[pos_mask] = 0.5 * (x_pos - beta + np.sqrt(term_inside_sqrt))
    neg_mask = matrix < -x0
    if np.any(neg_mask):
        x_neg = matrix[neg_mask]
        term_inside_sqrt = (x_neg - beta) ** 2 - 4 * lambda_val
        term_inside_sqrt[term_inside_sqrt < 0] = 0
        result[neg_mask] = 0.5 * (x_neg + beta - np.sqrt(term_inside_sqrt))
    return result


def wlrpca_2d(D, gamma=None, max_iter=200, tolerance=1e-7, verbose=False):
    """二维WLRPCA算法"""
    m, n = D.shape
    if gamma is None:
        gamma = 1 / np.sqrt(max(m, n))
    X, E = np.zeros_like(D), np.zeros_like(D)
    Y = D / max(norm(D, 2), norm(D, np.inf) / gamma)
    mu = 1.25 / norm(D, 2)
    rho, beta1, beta2 = 1.5, 0.1, 0.1
    for i in range(max_iter):
        temp_X = D - E + (1 / mu) * Y
        U, S, Vt = svd(temp_X, full_matrices=False)
        X = U @ np.diag(logarithmic_threshold(S, 1 / mu, beta1)) @ Vt
        temp_E = D - X + (1 / mu) * Y
        E = logarithmic_threshold(temp_E, gamma / mu, beta2)
        residual = D - X - E
        Y += mu * residual
        mu *= rho
        diff = norm(residual, 'fro') / norm(D, 'fro')
        if verbose and (i % 20 == 0 or i == max_iter - 1):  # 每20次迭代打印一次
            print(f"    Iter {i}: relative error = {diff:.6f}")
        if diff < tolerance:
            if verbose: print(f"    Converged at iteration {i}.")
            break
    return X, E


# =============================================================================
# 可视化函数 (已修改以适应不同剖面类型)
# =============================================================================
def visualize_slice_comparison(original_slice, background_slice, target_slice, slice_index, display_axis):
    """可视化单个剖面的WLRPCA处理前后对比 (使用独立色标)"""
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 根据剖面类型动态设置标题和标签
    if display_axis == 0:  # Inline
        slice_type, xlabel, ylabel = "Inline", "Crossline道号", "时间采样点"
    elif display_axis == 1:  # Crossline
        slice_type, xlabel, ylabel = "Crossline", "Inline道号", "时间采样点"
    else:  # Time Slice
        slice_type, xlabel, ylabel = "Time Slice", "Crossline道号", "Inline道号"

    fig, axes = plt.subplots(1, 3, figsize=(24, 9), sharex=True, sharey=True)

    # 为原始图和背景图计算共享的颜色范围
    vmax_shared = np.percentile(np.abs(original_slice), 98)

    # 绘制原始剖面
    im_aspect = 'auto' if display_axis != 2 else 'equal'  # 时间切片通常用1:1的比例
    axes[0].imshow(original_slice.T, cmap='seismic', aspect=im_aspect, vmin=-vmax_shared, vmax=vmax_shared)
    axes[0].set_title("原始剖面 (归一化后)", fontsize=14)
    axes[0].set_ylabel(ylabel, fontsize=12)
    axes[0].set_xlabel(xlabel, fontsize=12)

    # 绘制恢复的背景
    axes[1].imshow(background_slice.T, cmap='seismic', aspect=im_aspect, vmin=-vmax_shared, vmax=vmax_shared)
    axes[1].set_title("恢复的背景 (低秩部分)", fontsize=14)
    axes[1].set_xlabel(xlabel, fontsize=12)

    # 为目标图计算独立的颜色范围
    vmax_target = np.percentile(np.abs(target_slice), 99) if np.any(target_slice) else 1.0
    if vmax_target < 1e-9: vmax_target = np.max(np.abs(target_slice))

    # 绘制恢复的目标
    im = axes[2].imshow(target_slice.T, cmap='seismic', aspect=im_aspect, vmin=-vmax_target, vmax=vmax_target)
    axes[2].set_title("恢复的目标 (稀疏部分)", fontsize=14)
    axes[2].set_xlabel(xlabel, fontsize=12)
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    plt.suptitle(f"WLRPCA 单剖面处理效果 ({slice_type} 剖面: {slice_index})", fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# =============================================================================
# 主函数 (程序的入口 - 已修改)
# =============================================================================
def main():
    """主函数：从三维数据中提取一个二维剖面进行处理和可视化。"""
    # ==================== 用户参数设置区域 ====================
    # 1. 输入文件路径 (三维 .npy 或 .npz 文件)
    # input_file_path = r'C:\Work\sunjie\Python\cavity_modeling\data\input_npy\yingxi_crop.npy'
    input_file_path = r'C:\Work\sunjie\Python\cavity_modeling\data\batch\residual_3d.npy'

    # 2. 选择要处理的剖面方向 (0: Inline, 1: Crossline, 2: Time Slice)
    #    假设数据形状为 (inline, xline, dt)
    processing_axis = 1

    # 3. 选择要处理的剖面索引号
    slice_to_process = 500

    # 4. 正则化参数 gamma (关键调试参数)
    gamma_value = 0.0008

    # 5. 设置最大迭代次数
    max_iterations = 300
    # ============================================================

    # --- 步骤 1: 加载三维数据 ---
    print(f"--- 步骤 1: 从 '{input_file_path}' 加载三维数据 ---")
    try:
        D_3d = np.load(input_file_path)
        if D_3d.ndim != 3: raise ValueError("加载的数据不是三维的。")
        print(f"数据加载成功。原始形状: {D_3d.shape}")
    except Exception as e:
        print(f"错误: 加载文件时出错: {e}");
        return

    # --- 步骤 2: 数据归一化 ---
    print("\n--- 步骤 2: 对数据进行归一化处理 ---")
    data_max_abs = np.max(np.abs(D_3d))
    if data_max_abs == 0:
        print("错误：数据全为零。");
        return
    D_3d_normalized = D_3d / data_max_abs
    print(f"归一化完成。数据范围: [-1, 1]")

    # --- 步骤 3: 提取并处理指定的单个剖面 (已修正逻辑) ---
    print(f"\n--- 步骤 3: 提取并处理单个剖面 (axis={processing_axis}, index={slice_to_process}) ---")
    try:
        if processing_axis == 0:  # 提取 Inline
            original_slice = D_3d_normalized[slice_to_process, :, :]
        elif processing_axis == 1:  # 提取 Crossline
            original_slice = D_3d_normalized[:, slice_to_process, :]
        elif processing_axis == 2:  # 提取 Time Slice
            original_slice = D_3d_normalized[:, :, slice_to_process]
        else:
            print(f"错误: 无效的 processing_axis: {processing_axis}。请选择 0, 1, 或 2。")
            return
    except IndexError:
        max_idx = D_3d.shape[processing_axis] - 1
        print(f"错误: 剖面索引 {slice_to_process} 超出范围 (有效 0-{max_idx})。");
        return

    print(f"算法参数: gamma={gamma_value}, max_iter={max_iterations}")
    start_time_2d = time.time()
    X_slice_rec, E_slice_rec = wlrpca_2d(original_slice,
                                         gamma=gamma_value,
                                         max_iter=max_iterations,
                                         verbose=True)
    print(f"单剖面处理耗时: {time.time() - start_time_2d:.2f} 秒。")

    # --- 步骤 4: 可视化处理结果 ---
    print("\n--- 步骤 4: 显示处理结果... ---")
    visualize_slice_comparison(original_slice, X_slice_rec, E_slice_rec, slice_to_process, processing_axis)

    # --- 脚本结束 ---
    print("\n脚本运行结束。关闭图像窗口后程序将退出。")


if __name__ == "__main__":
    main()