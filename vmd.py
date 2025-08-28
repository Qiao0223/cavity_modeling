# -*- coding: utf-8 -*-
"""
一个完整的三维数据单剖面VMD处理脚本，采用【全局频率约束】策略。

工作流:
1. 【学习阶段】: 从整个3D数据体中随机采样数千道，进行初步VMD分解，
   并使用K-Means聚类算法，计算出K个“全局主频”。
2. 【应用阶段】: 对用户选定的二维剖面进行逐道VMD分解，但在分解时，
   强制使用上一步计算出的全局主频作为约束。
3. 【交互式融合】: 用户可以选择融合任意模态，查看背景压制效果。
"""
import numpy as np
import time
import matplotlib.pyplot as plt
from vmdpy import VMD
from sklearn.cluster import KMeans  # 用于频率聚类
from scipy.signal import hilbert  # (可选) 用于计算最终属性


# =============================================================================
# 阶段一：学习全局频率的函数
# =============================================================================
def get_global_frequencies(data_3d, K, alpha, num_samples=2000, random_seed=42):
    """
    从三维数据体中学习并返回K个全局主频。
    """
    print(f"--- 阶段一：开始学习全局频率 (采样 {num_samples} 道) ---")
    inlines, crosslines, _ = data_3d.shape

    # 设置随机种子以保证结果可复现
    np.random.seed(random_seed)
    sample_inlines = np.random.randint(0, inlines, num_samples)
    sample_crosslines = np.random.randint(0, crosslines, num_samples)

    all_omegas = []
    print("    正在对采样道进行初步VMD分解以收集频率...")
    for i in range(num_samples):
        if i % 200 == 0:
            print(f"      已处理 {i}/{num_samples}...")
        trace = data_3d[sample_inlines[i], sample_crosslines[i], :]

        if trace.shape[0] % 2 != 0:
            trace = trace[:-1]

        _, _, omegas = VMD(trace, alpha, tau=0., K=K, DC=0, init=1, tol=1e-7)
        all_omegas.extend(omegas)

    all_omegas = np.array(all_omegas).reshape(-1, 1)

    print("    正在使用K-Means对收集到的频率进行聚类...")
    kmeans = KMeans(n_clusters=K, random_state=random_seed, n_init='auto').fit(all_omegas)
    global_freqs = np.sort(kmeans.cluster_centers_.flatten())

    print(f"--- 全局频率学习完成 ---")
    print(f"    识别出的全局主频 (omega) 为: {np.round(global_freqs, 4)}")

    return global_freqs


# =============================================================================
# 阶段二：应用约束分解的函数
# =============================================================================
def vmd_2d_rowwise_constrained(signal_2d, alpha, K, global_omegas, verbose=False):
    """
    使用【全局频率约束】对二维剖面进行逐行VMD分解。
    """
    print("--- 阶段二：开始应用约束VMD分解 ---")
    rows, cols = signal_2d.shape
    decomposed_modes_3d = np.zeros((K, rows, cols), dtype=np.float32)

    # vmdpy库使用字典来传递初始频率作为约束
    init_dict = {'omegas': global_omegas}

    for i in range(rows):
        if verbose and (i % 100 == 0 or i == rows - 1):
            print(f"    正在处理第 {i + 1}/{rows} 道...")

        row_1d = signal_2d[i, :]

        signal_to_process = row_1d
        if row_1d.shape[0] % 2 != 0:
            signal_to_process = row_1d[:-1]

        u, _, _ = VMD(signal_to_process, alpha, tau=0., K=K, DC=0, init=init_dict, tol=1e-7)

        decomposed_modes_3d[:, i, :u.shape[1]] = u

    print("    约束VMD分解完成。")
    return decomposed_modes_3d


# ... (可视化函数 visualize_vmd_comparison 和 visualize_fusion_result 与之前相同，无需修改) ...
def visualize_vmd_comparison(original_slice, decomposed_modes, slice_index, processing_axis):
    """显示原始剖面和所有分解出的模态。"""
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

    im = axes[0].imshow(original_slice.T, cmap='seismic', aspect='auto', vmin=-vmax_original, vmax=vmax_original)
    axes[0].set_title("原始剖面 (归一化后)", fontsize=14)
    axes[0].set_ylabel("时间采样点", fontsize=12)
    axes[0].set_xlabel(xlabel, fontsize=12)
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    for i in range(K):
        ax = axes[i + 1]
        mode_slice = decomposed_modes[i, :, :]
        vmax_mode = np.percentile(np.abs(mode_slice), 99) if np.any(mode_slice) else 1.0
        if vmax_mode < 1e-9: vmax_mode = np.max(np.abs(mode_slice)) if np.any(mode_slice) else 1.0
        im_mode = ax.imshow(mode_slice.T, cmap='seismic', aspect='auto', vmin=-vmax_mode, vmax=vmax_mode)
        ax.set_title(f"分解模态 {i + 1}", fontsize=14)
        ax.set_xlabel(xlabel, fontsize=12)
        fig.colorbar(im_mode, ax=ax, fraction=0.046, pad=0.04)

    for i in range(num_plots, len(axes)): axes[i].axis('off')
    plt.suptitle(f"VMD单剖面分解效果 ({slice_type} 剖面: {slice_index})", fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def visualize_fusion_result(original_slice, fused_mode, selected_indices_str):
    """在一个新窗口中显示融合后的结果。"""
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    vmax_original = np.percentile(np.abs(original_slice), 98)
    im1 = axes[0].imshow(original_slice.T, cmap='seismic', aspect='auto', vmin=-vmax_original, vmax=vmax_original)
    axes[0].set_title("原始剖面", fontsize=14)
    axes[0].set_ylabel("时间采样点", fontsize=12)
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    vmax_fused = np.percentile(np.abs(fused_mode), 99) if np.any(fused_mode) else 1.0
    im2 = axes[1].imshow(fused_mode.T, cmap='seismic', aspect='auto', vmin=-vmax_fused, vmax=vmax_fused)
    axes[1].set_title(f"融合结果 (模态: {selected_indices_str})", fontsize=14)
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


# =============================================================================
# 主函数 (整合了新流程)
# =============================================================================
def main():
    # ==================== 用户参数设置区域 ====================
    input_file_path = r'C:\Work\sunjie\Python\cavity_modeling\data\input_npy\fuyuan3_crop.npz'
    processing_axis = 1
    slice_to_preview = 500
    alpha, tau, K, DC, init, tol = 2000, 0., 8, 0, 1, 1e-7
    # ============================================================

    print(f"--- 步骤 1: 从 '{input_file_path}' 加载数据 ---")
    # ... (数据加载和归一化代码与之前相同) ...
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

    # --- 【【【【【 新增的核心步骤：学习全局频率 】】】】】 ---
    # 这个过程可能需要几分钟，因为它要处理数千道
    start_time_learning = time.time()
    global_frequencies = get_global_frequencies(D_3d_normalized, K, alpha)
    print(f"全局频率学习阶段耗时: {time.time() - start_time_learning:.2f} 秒。")

    print(f"\n--- 步骤 3: 提取剖面 #{slice_to_preview} (沿 axis={processing_axis}) ---")
    # ... (提取剖面代码与之前相同) ...
    try:
        original_slice = np.take(D_3d_normalized, slice_to_preview, axis=processing_axis)
        print(f"剖面提取成功，形状: {original_slice.shape}")
    except IndexError:
        max_idx = D_3d.shape[processing_axis] - 1
        print(f"错误: 剖面索引 {slice_to_preview} 超出范围 (有效范围 0-{max_idx})。");
        return

    print(f"\n--- 步骤 4: 开始对选定剖面进行【约束VMD分解】 (K={K}) ---")
    start_time_2d = time.time()
    # --- 【【【【【 调用新的约束分解函数 】】】】】 ---
    decomposed_modes = vmd_2d_rowwise_constrained(
        original_slice, alpha, K, global_frequencies, verbose=True
    )
    print(f"单剖面约束VMD分解耗时: {time.time() - start_time_2d:.2f} 秒。")

    print("\n--- 步骤 5: 显示所有分解结果... ---")
    # ... (交互式融合循环与之前完全相同) ...
    print(">>> 请查看弹出的图像窗口，然后关闭它以继续...")
    visualize_vmd_comparison(original_slice, decomposed_modes, slice_to_preview, processing_axis)

    while True:
        print("\n" + "=" * 50)
        print("进入交互式模态融合模式。")
        user_input = input(f">>> 请输入您希望融合的模态编号 (1-{K})，用逗号或空格隔开 (例如: 6 7 8)。\n"
                           f">>> 输入 'q' 退出程序: ")

        if user_input.lower() == 'q':
            break

        try:
            str_indices = user_input.replace(',', ' ').split()
            indices_to_combine = [int(i) - 1 for i in str_indices]

            if any(i < 0 or i >= K for i in indices_to_combine):
                print(f"!!! 错误: 输入的模态编号必须在 1 到 {K} 之间。请重试。")
                continue

            print(f"--- 正在融合模态: {[i + 1 for i in indices_to_combine]} ---")
            selected_modes = decomposed_modes[indices_to_combine, :, :]
            fused_mode = np.sum(selected_modes, axis=0)

            print(">>> 显示融合结果... 请查看新弹出的窗口。")
            visualize_fusion_result(original_slice, fused_mode, user_input)

        except ValueError:
            print("!!! 错误: 输入无效，请输入数字、逗号或空格。请重试。")
        except Exception as e:
            print(f"!!! 发生未知错误: {e}")

    print("\n脚本运行结束。")


if __name__ == "__main__":
    main()