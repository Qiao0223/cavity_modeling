import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import itertools  # 我们将使用这个库来创建文件对
import matplotlib as mpl

# ==============================================================================
# --- 1. 配置 ---
# ==============================================================================
# 包含所有2D属性切片 (.npy) 的文件夹
ATTRIBUTE_SLICES_FOLDER = r"C:\Work\sunjie\Python\cavity_modeling\data\2d"

# 您之前创建的标签掩模文件路径
LABEL_MASK_PATH = r"C:\Work\sunjie\Python\cavity_modeling\data\labels\label_mask_inline118.npy"

# 保存生成的交会图图片的文件夹 (如果不存在，会自动创建)
OUTPUT_IMAGES_FOLDER = r"C:\Work\sunjie\Python\cavity_modeling\data\crossplots"

# --- 功能开关 ---
# 设置为 True, 会将每张交会图保存为.png文件
SAVE_PLOTS = True
# 设置为 True, 会在屏幕上显示每张交会图 (如果文件很多，建议设为False)
SHOW_PLOTS = True


# ==============================================================================
# --- 2. 批量交会分析程序 ---
# ==============================================================================

def batch_crossplot_analysis(attr_folder, label_path, output_folder,
                             save_plots=True, show_plots=True):
    """
    自动对文件夹内所有属性切片进行两两交会图分析。
    """

    # 设置中文字体
    try:
        mpl.rcParams['font.sans-serif'] = ['SimHei']
        mpl.rcParams['axes.unicode_minus'] = False
    except Exception:
        print("警告：未找到'SimHei'字体，图表中的中文可能显示为方框。")

    # 步骤 1: 加载标签掩模
    try:
        label_mask = np.load(label_path)
        print(f"成功加载标签掩模: {label_path}\n")
    except FileNotFoundError:
        print(f"致命错误：未找到标签掩模文件！请检查路径: {label_path}")
        return

    # 步骤 2: 查找所有属性切片文件
    search_path = os.path.join(attr_folder, '*.npy')
    attr_files = glob.glob(search_path)

    if len(attr_files) < 2:
        print(f"错误：文件夹 {attr_folder} 中必须至少有两个.npy属性文件才能进行交会分析。")
        return

    print(f"找到 {len(attr_files)} 个属性文件，将生成 {len(list(itertools.combinations(attr_files, 2)))} 张交会图。")

    # 步骤 3: 创建所有文件的两两组合
    # 例如, [A, B, C] -> (A, B), (A, C), (B, C)
    file_pairs = itertools.combinations(attr_files, 2)

    # 步骤 4: 循环处理每一对文件
    for attr_a_path, attr_b_path in file_pairs:
        try:
            # 获取简短的文件名用于图表标题和保存
            attr_a_name = os.path.splitext(os.path.basename(attr_a_path))[0]
            attr_b_name = os.path.splitext(os.path.basename(attr_b_path))[0]

            print(f"\n--- 正在绘制: '{attr_a_name}' vs '{attr_b_name}' ---")

            # 加载两个属性切片
            attr_a_slice = np.load(attr_a_path)
            attr_b_slice = np.load(attr_b_path)

            # 检查形状是否与标签一致
            if not (attr_a_slice.shape == label_mask.shape and attr_b_slice.shape == label_mask.shape):
                print(f"  - 警告：文件形状与标签不匹配，跳过此组合。")
                print(
                    f"    - {attr_a_name}: {attr_a_slice.shape}, {attr_b_name}: {attr_b_slice.shape}, Label: {label_mask.shape}")
                continue

            # 压平数据
            attr_a_flat = attr_a_slice.flatten()
            attr_b_flat = attr_b_slice.flatten()
            mask_flat = label_mask.flatten()

            # 分离前景 (缝洞体) 和背景数据点
            attr_a_fg = attr_a_flat[mask_flat == 1]
            attr_b_fg = attr_b_flat[mask_flat == 1]
            attr_a_bg = attr_a_flat[mask_flat == 0]
            attr_b_bg = attr_b_flat[mask_flat == 0]

            # 开始绘图
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.scatter(attr_a_bg, attr_b_bg, s=5, c='blue', alpha=0.1, label='背景 (非缝洞体)')
            ax.scatter(attr_a_fg, attr_b_fg, s=10, c='red', alpha=0.5, label='缝洞体')
            ax.set_title(f"交会分析: {attr_a_name} vs {attr_b_name}", fontsize=16)
            ax.set_xlabel(attr_a_name, fontsize=12)
            ax.set_ylabel(attr_b_name, fontsize=12)
            ax.legend(fontsize=12, markerscale=3)
            ax.grid(True, linestyle='--', alpha=0.6)

            # 步骤 5: 保存和/或显示图像
            if save_plots:
                os.makedirs(output_folder, exist_ok=True)
                output_filename = f"{attr_a_name}_vs_{attr_b_name}.png"
                output_path = os.path.join(output_folder, output_filename)
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                print(f"  -> 成功保存图像到: {output_filename}")

            if show_plots:
                plt.show()

            # 关闭图形，释放内存，准备下一个循环
            plt.close(fig)

        except Exception as e:
            print(f"  - 处理组合 ({attr_a_name}, {attr_b_name}) 时发生错误: {e}")

    print("\n所有交会图处理完毕！")


# --- 运行主程序 ---
if __name__ == "__main__":
    batch_crossplot_analysis(
        attr_folder=ATTRIBUTE_SLICES_FOLDER,
        label_path=LABEL_MASK_PATH,
        output_folder=OUTPUT_IMAGES_FOLDER,
        save_plots=SAVE_PLOTS,
        show_plots=SHOW_PLOTS
    )