import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 全局设置，解决中文乱码问题
# =============================================================================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def fuse_and_visualize(
        sof_residual_path: str,
        wlrpca_sparse_path: str,
        title_info: str = "协同约束效果对比",
        cmap_fused: str = 'RdBu_r'  # +++ 新增: 为融合结果指定色标 +++
):
    """
    加载两个二维NPY文件，将它们融合，并可视化对比。
    (*** 新版本: 优化了融合结果的色标 ***)
    """
    # --- 步骤 1: 加载数据 (保持不变) ---
    print("--- 步骤 1: 加载两个二维输入文件 ---")
    try:
        sof_residual = np.load(sof_residual_path)
        print(f"  - 成功加载 SOF 残差: {sof_residual_path}, 形状: {sof_residual.shape}")
        wlrpca_sparse_original = np.load(wlrpca_sparse_path)
        print(f"  - 成功加载 WLRPCA 稀疏部分 (原始形状): {wlrpca_sparse_original.shape}")
        wlrpca_sparse = wlrpca_sparse_original.T
        print(f"  - 已将 WLRPCA 数组转置，新形状: {wlrpca_sparse.shape}")
        if sof_residual.shape != wlrpca_sparse.shape:
            raise ValueError(f"错误: 转置后形状仍然不匹配！ "
                             f"SOF: {sof_residual.shape}, WLRPCA: {wlrpca_sparse.shape}")
    except Exception as e:
        print(f"加载数据时出错: {e}")
        return

    # --- 步骤 2: 数据归一化与融合 (保持不变) ---
    print("\n--- 步骤 2: 对输入数据进行归一化并融合 ---")
    epsilon = 1e-9
    norm_sof_residual = sof_residual / (np.max(np.abs(sof_residual)) + epsilon)
    norm_wlrpca_sparse = wlrpca_sparse / (np.max(np.abs(wlrpca_sparse)) + epsilon)
    # fused_result = norm_sof_residual + norm_wlrpca_sparse
    fused_result = norm_wlrpca_sparse * norm_wlrpca_sparse
    print("  - 融合（逐点相乘）完成。")

    # --- 步骤 3: 可视化对比 ---
    print("\n--- 步骤 3: 生成可视化对比图 ---")
    fig, axes = plt.subplots(1, 3, figsize=(24, 10), sharex=True, sharey=True)

    vmax_sof = np.percentile(np.abs(sof_residual), 99)
    vmax_wlrpca = np.percentile(np.abs(wlrpca_sparse), 99)
    vmax_fused = np.percentile(np.abs(fused_result), 99) if np.any(fused_result) else 1.0

    # 1. 显示SOF残差 (使用 seismic)
    axes[0].imshow(sof_residual, cmap='seismic', aspect='auto', vmin=-vmax_sof, vmax=vmax_sof)
    axes[0].set_title("输入 1: SOF 残差", fontsize=16)
    axes[0].set_ylabel("时间采样点", fontsize=14)
    axes[0].set_xlabel("道号", fontsize=14)

    # 2. 显示WLRPCA稀疏部分 (使用 seismic)
    axes[1].imshow(wlrpca_sparse, cmap='seismic', aspect='auto', vmin=-vmax_wlrpca, vmax=vmax_wlrpca)
    axes[1].set_title("输入 2: WLRPCA 稀疏部分 (已转置)", fontsize=16)
    axes[1].set_xlabel("道号", fontsize=14)

    # 3. 显示融合结果 (+++ 使用新的、可指定的色标 +++)
    im = axes[2].imshow(fused_result, cmap=cmap_fused, aspect='auto', vmin=-vmax_fused, vmax=vmax_fused)
    axes[2].set_title(f"融合结果 (使用 '{cmap_fused}' 色标)", fontsize=16)
    axes[2].set_xlabel("道号", fontsize=14)

    # 调整色标的标签大小
    cbar = fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=12)

    plt.suptitle(title_info, fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    print("\n可视化完成。关闭图像窗口后程序将退出。")


if __name__ == '__main__':
    # ==================== 用户设置区域 ====================
    # --- 请在这里修改您的两个二维 .npy 文件路径 ---

    # 您的SOF残差结果文件路径
    SOF_RESIDUAL_NPY_PATH = r"sof.npy"

    # 您的WLRPCA稀疏部分结果文件路径
    WLRPCASPARSE_NPY_PATH = r"wlrpca.npy"

    # 图像的标题信息，可以自定义
    SLICE_INFO = "Xline 500"

    # =======================================================

    # --- 运行融合与可视化程序 ---
    # 检查路径是否被修改，如果未修改则提示用户
    if "path/to/your" in SOF_RESIDUAL_NPY_PATH or "path/to/your" in WLRPCASPARSE_NPY_PATH:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! 警告: 请在代码中修改 SOF_RESIDUAL_NPY_PATH 和     !!!")
        print("!!!       WLRPCASPARSE_NPY_PATH 为您的实际文件路径！   !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    else:
        fuse_and_visualize(
            sof_residual_path=SOF_RESIDUAL_NPY_PATH,
            wlrpca_sparse_path=WLRPCASPARSE_NPY_PATH,
            title_info=f"协同约束效果对比 ({SLICE_INFO})",
            cmap_fused = 'RdBu_r'
        )