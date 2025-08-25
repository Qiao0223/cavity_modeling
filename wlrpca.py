# -*- coding: utf-8 -*-
"""
一个完整的三维地震数据WLRPCA处理与可视化脚本。

功能：
1. 从.npz文件加载三维地震数据。
2. 应用三维WLRPCA算法，将数据分解为背景（低秩）和目标（稀疏）两部分。
3. 将分解后的两个三维数据体保存为.npy文件。
4. 可视化处理前后的剖面对比图。

"""
import numpy as np
import time
import matplotlib.pyplot as plt
from numpy.linalg import svd, norm


# =============================================================================
# 核心二维WLRPCA算法
# =============================================================================
def logarithmic_threshold(matrix, lambda_val, beta):
    """根据论文中的公式(11)实现对数阈值函数"""
    x0 = np.sqrt(2 * lambda_val - beta) if (2 * lambda_val - beta) > 0 else 0
    result = np.zeros_like(matrix, dtype=float)

    # 处理 x > x0 的情况
    pos_mask = matrix > x0
    if np.any(pos_mask):
        x_pos = matrix[pos_mask]
        term_inside_sqrt = (x_pos + beta) ** 2 - 4 * lambda_val
        term_inside_sqrt[term_inside_sqrt < 0] = 0
        result[pos_mask] = 0.5 * (x_pos - beta + np.sqrt(term_inside_sqrt))

    # 处理 x < -x0 的情况
    neg_mask = matrix < -x0
    if np.any(neg_mask):
        x_neg = matrix[neg_mask]
        term_inside_sqrt = (x_neg - beta) ** 2 - 4 * lambda_val
        term_inside_sqrt[term_inside_sqrt < 0] = 0
        result[neg_mask] = 0.5 * (x_neg + beta - np.sqrt(term_inside_sqrt))

    return result


def wlrpca_2d(D, gamma=None, max_iter=100, tolerance=1e-6, verbose=False):
    """二维WLRPCA算法"""
    m, n = D.shape
    if gamma is None:
        gamma = 1 / np.sqrt(max(m, n))
    X, E = np.zeros_like(D), np.zeros_like(D)
    Y = D / max(norm(D, 2), norm(D, np.inf) / gamma)
    mu = 1.25 / norm(D, 2)
    rho, beta1, beta2 = 1.6, 0.1, 0.1

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
        if verbose and i % 10 == 0:
            print(f"    Iter {i}: relative error = {diff:.5f}")
        if diff < tolerance:
            break

    return X, E


# =============================================================================
# 三维处理函数
# =============================================================================
def wlrpca_3d(D_3d, axis_to_process=1, **kwargs):
    """对三维数据体逐个剖面应用二维WLRPCA算法"""
    if D_3d.ndim != 3:
        raise ValueError("Input data must be a 3D array.")
    X_3d, E_3d = np.zeros_like(D_3d), np.zeros_like(D_3d)
    start_time = time.time()

    if axis_to_process == 1:
        num_slices = D_3d.shape[1]
        for i in range(num_slices):
            print(f"Processing slice {i + 1}/{num_slices} (axis 1)...")
            slice_2d = D_3d[:, i, :]
            X_slice, E_slice = wlrpca_2d(slice_2d, **kwargs)
            X_3d[:, i, :] = X_slice
            E_3d[:, i, :] = E_slice
    elif axis_to_process == 2:
        num_slices = D_3d.shape[2]
        for j in range(num_slices):
            print(f"Processing slice {j + 1}/{num_slices} (axis 2)...")
            slice_2d = D_3d[:, :, j]
            X_slice, E_slice = wlrpca_2d(slice_2d, **kwargs)
            X_3d[:, :, j] = X_slice
            E_3d[:, :, j] = E_slice
    else:
        raise ValueError("axis_to_process must be 1 or 2.")

    end_time = time.time()
    print(f"\n3D processing finished in {end_time - start_time:.2f} seconds.")
    return X_3d, E_3d


# =============================================================================
# 可视化函数
# =============================================================================
def visualize_wlrpca_comparison(original_data, background_data, target_data, slice_index, display_axis=1):
    """可视化WLRPCA处理前后的剖面对比"""
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    if display_axis == 1:
        original_slice = original_data[:, slice_index, :]
        background_slice = background_data[:, slice_index, :]
        target_slice = target_data[:, slice_index, :]
        slice_type = "Inline"
        xlabel = "Crossline道号"
    elif display_axis == 2:
        original_slice = original_data[:, :, slice_index]
        background_slice = background_data[:, :, slice_index]
        target_slice = target_data[:, :, slice_index]
        slice_type = "Crossline"
        xlabel = "Inline道号"
    else:
        raise ValueError("display_axis 必须为 1 或 2")

    fig, axes = plt.subplots(1, 3, figsize=(22, 9), sharex=True, sharey=True)
    vmax = np.percentile(np.abs(original_slice), 98)

    # 绘制原始剖面
    axes[0].imshow(original_slice.T, cmap='seismic', aspect='auto', vmin=-vmax, vmax=vmax)
    axes[0].set_title("原始剖面", fontsize=14)
    axes[0].set_ylabel("时间采样点", fontsize=12)
    axes[0].set_xlabel(xlabel, fontsize=12)

    # 绘制恢复的背景
    axes[1].imshow(background_slice.T, cmap='seismic', aspect='auto', vmin=-vmax, vmax=vmax)
    axes[1].set_title("恢复的背景 (低秩部分)", fontsize=14)
    axes[1].set_xlabel(xlabel, fontsize=12)

    # 绘制恢复的目标
    axes[2].imshow(target_slice.T, cmap='seismic', aspect='auto', vmin=-vmax, vmax=vmax)
    axes[2].set_title("恢复的目标 (稀疏部分)", fontsize=14)
    axes[2].set_xlabel(xlabel, fontsize=12)

    plt.suptitle(f"WLRPCA 分解效果对比 ({slice_type} 剖面: {slice_index})", fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# =============================================================================
# 主函数 (程序的入口)
# =============================================================================
def main():
    """
    主函数：加载数据、运行处理、保存结果并进行可视化。
    """
    # ==================== 用户参数设置区域 ====================
    # 1. 输入文件路径
    input_file_path = r'C:\Work\sunjie\Python\cavity_modeling\data\input_npy\yingxi_crop.npz'

    # 2. 输出文件的前缀
    output_file_prefix = "final_wlrpca_output"

    # 3. 指定沿着哪个轴进行切片处理
    #    1: 沿着Inline轴切片 (处理 Crossline 剖面, 即 Time x Crossline)
    #    2: 沿着Crossline轴切片 (处理 Inline 剖面, 即 Time x Inline)
    processing_axis = 1

    # 4. 正则化参数 gamma (可选)
    #    - 设置为 None: 程序会为每个2D剖面自动计算 gamma 值 (推荐)
    #    - 设置为一个浮点数 (例如 0.05): 对所有剖面使用固定的 gamma 值
    gamma_value = None

    # 5. 可视化参数：选择要显示的剖面
    #    - slice_to_display: 剖面的索引号 (例如第 50 道)
    #    - visualization_axis: 剖面的方向 (通常与 processing_axis 保持一致)
    slice_to_display = 50
    visualization_axis = 1
    # ============================================================

    # --- 步骤 1: 加载输入数据 ---
    print(f"--- 步骤 1: 从 '{input_file_path}' 加载数据 ---")
    try:
        npz = np.load(input_file_path)
        D_3d = npz["data"]
        if D_3d.ndim != 3:
            raise ValueError("加载的数据不是三维的。")
        print(f"数据加载成功。形状: {D_3d.shape}")
    except FileNotFoundError:
        print(f"错误: 输入文件未找到 '{input_file_path}'")
        return
    except Exception as e:
        print(f"错误: 加载或验证文件时出错: {e}")
        return

    # --- 步骤 2: 运行3D WLRPCA算法 ---
    print("\n--- 步骤 2: 运行三维WLRPCA分解 ---")
    X_3d_recovered, E_3d_recovered = wlrpca_3d(
        D_3d,
        axis_to_process=processing_axis,
        gamma=gamma_value
    )

    # --- 步骤 3: 保存结果 ---
    print("\n--- 步骤 3: 保存结果 ---")
    output_background_path = f"{output_file_prefix}_background.npy"
    output_target_path = f"{output_file_prefix}_target.npy"
    try:
        np.save(output_background_path, X_3d_recovered)
        print(f"背景部分已保存到: '{output_background_path}'")
        np.save(output_target_path, E_3d_recovered)
        print(f"目标部分已保存到: '{output_target_path}'")
    except Exception as e:
        print(f"错误: 保存输出文件时出错: {e}")

    # --- 步骤 4: 可视化结果 ---
    print("\n--- 步骤 4: 可视化结果 ---")
    try:
        if slice_to_display >= D_3d.shape[visualization_axis]:
            max_idx = D_3d.shape[visualization_axis] - 1
            print(f"警告: 选择的剖面索引 {slice_to_display} 超出范围。有效索引为 0 到 {max_idx}。")
            slice_to_display_safe = max_idx // 2
            print(f"将自动显示中间的剖面: {slice_to_display_safe}")
        else:
            slice_to_display_safe = slice_to_display

        visualize_wlrpca_comparison(
            original_data=D_3d,
            background_data=X_3d_recovered,
            target_data=E_3d_recovered,
            slice_index=slice_to_display_safe,
            display_axis=visualization_axis
        )
    except Exception as e:
        print(f"错误: 可视化时出错: {e}")

    print("\n处理和可视化全部完成。")


# Python程序的标准入口
if __name__ == "__main__":
    main()