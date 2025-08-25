# -*- coding: utf-8 -*-
"""
一个完整的三维地震数据WLRPCA处理与可视化脚本 (PyCharm适用)。

最终版V3特性：
- 非交互式工作流：先快速预览单个剖面，关闭预览图后自动继续完整处理。
- 独立色标：修复了目标信号因能量弱而无法显示的问题。
- 数据归一化：将数据缩放到[-1, 1]，增强算法的数值稳定性。
- 参数优化：增加了默认迭代次数，以适应复杂真实数据。
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
            print(f"Processing full volume: slice {i + 1}/{num_slices} (axis 1)...")
            slice_2d = D_3d[:, i, :]
            X_slice, E_slice = wlrpca_2d(slice_2d, verbose=False, **kwargs)
            X_3d[:, i, :] = X_slice
            E_3d[:, i, :] = E_slice
    elif axis_to_process == 2:
        num_slices = D_3d.shape[2]
        for j in range(num_slices):
            print(f"Processing full volume: slice {j + 1}/{num_slices} (axis 2)...")
            slice_2d = D_3d[:, :, j]
            X_slice, E_slice = wlrpca_2d(slice_2d, verbose=False, **kwargs)
            X_3d[:, :, j] = X_slice
            E_3d[:, :, j] = E_slice
    end_time = time.time()
    print(f"\nFull 3D processing finished in {end_time - start_time:.2f} seconds.")
    return X_3d, E_3d


# =============================================================================
# 可视化函数
# =============================================================================
def visualize_slice_comparison(original_slice, background_slice, target_slice, slice_index, display_axis=1):
    """可视化单个剖面的WLRPCA处理前后对比 (使用独立色标)"""
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    slice_type, xlabel = ("Inline", "Crossline道号") if display_axis == 1 else ("Crossline", "Inline道号")
    fig, axes = plt.subplots(1, 3, figsize=(24, 9), sharex=True, sharey=True)

    # 为原始图和背景图计算共享的颜色范围
    vmax_shared = np.percentile(np.abs(original_slice), 98)

    # 绘制原始剖面
    axes[0].imshow(original_slice.T, cmap='seismic', aspect='auto', vmin=-vmax_shared, vmax=vmax_shared)
    axes[0].set_title("原始剖面 (归一化后)", fontsize=14)
    axes[0].set_ylabel("时间采样点", fontsize=12)
    axes[0].set_xlabel(xlabel, fontsize=12)

    # 绘制恢复的背景
    axes[1].imshow(background_slice.T, cmap='seismic', aspect='auto', vmin=-vmax_shared, vmax=vmax_shared)
    axes[1].set_title("恢复的背景 (低秩部分)", fontsize=14)
    axes[1].set_xlabel(xlabel, fontsize=12)

    # 为目标图计算独立的颜色范围
    vmax_target = np.percentile(np.abs(target_slice), 99) if np.any(target_slice) else 1.0
    if vmax_target < 1e-9: vmax_target = np.max(np.abs(target_slice))

    # 绘制恢复的目标
    im = axes[2].imshow(target_slice.T, cmap='seismic', aspect='auto', vmin=-vmax_target, vmax=vmax_target)
    axes[2].set_title("恢复的目标 (稀疏部分)", fontsize=14)
    axes[2].set_xlabel(xlabel, fontsize=12)
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    plt.suptitle(f"WLRPCA 单剖面测试效果 ({slice_type} 剖面: {slice_index})", fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# =============================================================================
# 主函数 (程序的入口)
# =============================================================================
def main():
    """主函数：执行先预览、后处理的完整流程"""
    # ==================== 用户参数设置区域 ====================
    # 1. 输入文件路径
    input_file_path = r'C:\Work\sunjie\Python\cavity_modeling\data\input_npy\yingxi_crop.npz'

    # 2. 输出文件的前缀
    output_file_prefix = "final_output"

    # 3. 选择要测试和处理的剖面方向 (1: Inline, 2: Crossline)
    processing_axis = 1

    # 4. 选择要预览的剖面索引号
    slice_to_preview = 500

    # 5. 正则化参数 gamma (关键调试参数)
    #    建议从 0.01 开始，如果目标信号太少则减小，如果太多则增大。
    gamma_value = 0.00005
    # ============================================================

    # --- 步骤 1: 加载数据 ---
    print(f"--- 步骤 1: 从 '{input_file_path}' 加载数据 ---")
    try:
        D_3d = np.load(input_file_path)["data"]
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

    # --- 步骤 3: 提取并处理单个剖面以供预览 ---
    print(f"\n--- 步骤 3: 提取并处理单个剖面 #{slice_to_preview} 以供预览 (gamma={gamma_value}) ---")
    try:
        if processing_axis == 1:
            original_slice = D_3d_normalized[:, slice_to_preview, :]
        else:
            original_slice = D_3d_normalized[:, :, slice_to_preview]
    except IndexError:
        max_idx = D_3d.shape[processing_axis] - 1
        print(f"错误: 剖面索引 {slice_to_preview} 超出范围 (有效 0-{max_idx})。");
        return

    start_time_2d = time.time()
    X_slice_rec, E_slice_rec = wlrpca_2d(original_slice, gamma=gamma_value, verbose=True)
    print(f"单剖面处理耗时: {time.time() - start_time_2d:.2f} 秒。")

    # --- 步骤 4: 可视化预览结果 (程序会在此暂停) ---
    print("\n--- 步骤 4: 显示预览结果... ---")
    print(">>> 请在查看后【手动关闭】弹出的图像窗口，程序将自动继续进行完整处理。<<<")
    visualize_slice_comparison(original_slice, X_slice_rec, E_slice_rec, slice_to_preview, processing_axis)

    # # --- 步骤 5: 开始完整的三维处理 ---
    # print("\n--- 步骤 5: 预览窗口已关闭。开始完整的三维数据处理... ---")
    # # 注意：我们将归一化后的数据送入处理
    # X_3d_rec_norm, E_3d_rec_norm = wlrpca_3d(D_3d_normalized, axis_to_process=processing_axis, gamma=gamma_value)
    #
    # # --- 步骤 6: 反归一化，恢复原始量纲 ---
    # print("\n--- 步骤 6: 对结果进行反归一化 ---")
    # X_3d_final = X_3d_rec_norm * data_max_abs
    # E_3d_final = E_3d_rec_norm * data_max_abs
    # print("反归一化完成。")
    #
    # # --- 步骤 7: 保存完整的三维结果 ---
    # print("\n--- 步骤 7: 保存完整的三维结果 ---")
    # try:
    #     bg_path = f"{output_file_prefix}_background.npy"
    #     tg_path = f"{output_file_prefix}_target.npy"
    #     np.save(bg_path, X_3d_final)
    #     print(f"背景部分已保存到: '{bg_path}'")
    #     np.save(tg_path, E_3d_final)
    #     print(f"目标部分已保存到: '{tg_path}'")
    # except Exception as e:
    #     print(f"错误: 保存输出文件时出错: {e}")
    #
    # print("\n脚本运行结束。")


if __name__ == "__main__":
    main()