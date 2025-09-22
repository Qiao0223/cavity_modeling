import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import matplotlib as mpl

# ==============================================================================
# --- 1. 请在这里配置您的参数 ---
# ==============================================================================

# 输入文件夹：包含所有原始3D .npy文件的文件夹
SOURCE_FOLDER = r"C:\Work\sunjie\Python\cavity_modeling\data\batch"

# 输出文件夹：用于存放提取出的2D切片的文件夹
DESTINATION_FOLDER = r"C:\Work\sunjie\Python\cavity_modeling\data\2d"

# --- 新增功能开关 ---
# 设置为 True，则在保存每个切片前会弹窗显示图像预览
# 设置为 False，则直接进行批量保存，不显示图像
SHOW_PLOT_PREVIEW = True

# --- 切片和裁剪参数 ---
SLICE_DIM = 'inline'
SLICE_INDEX = 118
TIME_CROP_START, TIME_CROP_END = 1400, 1750
TRACE_CROP_START, TRACE_CROP_END = 600, 1100


# ==============================================================================
# --- 2. 辅助函数：用于显示切片预览 (新增) ---
# ==============================================================================

def show_slice_preview(slice_data, title):
    """
    显示单个2D切片的图像预览，使用对称色标。
    """
    # 设置中文字体，确保标题能正确显示
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    mpl.rcParams['axes.unicode_minus'] = False

    # 计算对称色标范围
    max_pos = np.max(slice_data)
    min_neg = np.min(slice_data)
    vmin, vmax = None, None
    if max_pos > 0 and min_neg < 0:
        limit = min(max_pos, abs(min_neg))
        vmin, vmax = -limit, limit

    # 绘图
    plt.figure(figsize=(10, 7))
    plt.imshow(slice_data, cmap='seismic', aspect='auto', vmin=vmin, vmax=vmax)

    # 美化
    plt.colorbar(label='振幅 (Amplitude)')
    plt.title(title, fontsize=16)
    plt.xlabel('道号 (Trace Index)')
    plt.ylabel('时间采样点 (Time Sample Index)')
    plt.grid(True, linestyle='--', alpha=0.5)

    # 显示图像，脚本会在此暂停，直到用户关闭窗口
    print("    -> 正在显示预览图... (请关闭图像窗口以继续)")
    plt.show()


# ==============================================================================
# --- 3. 脚本主程序 (已修改) ---
# ==============================================================================

def batch_extract_and_crop_slices(source_dir, dest_dir, dim, index,
                                  t_start, t_end, trace_start, trace_end,
                                  show_plot=False):  # <-- 新增参数
    """
    批量处理文件夹中的npy文件，提取、显示并裁剪指定的切片。
    """
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        print(f"成功创建输出文件夹: {dest_dir}")

    search_path = os.path.join(source_dir, '*.npy')
    npy_files = glob.glob(search_path)

    if not npy_files:
        print(f"警告: 在文件夹 {source_dir} 中未找到任何 .npy 文件。")
        return

    print(f"\n找到 {len(npy_files)} 个 .npy 文件，开始处理...")

    for file_path in npy_files:
        try:
            base_filename = os.path.basename(file_path)
            print(f"  > 正在处理: {base_filename}")

            data_3d = np.load(file_path)

            if dim.lower() == 'inline':
                full_slice_2d = data_3d[index, :, :].T
            elif dim.lower() == 'xline':
                full_slice_2d = data_3d[:, index, :].T
            else:
                print(f"    - 错误: 不支持的维度 '{dim}'。跳过。")
                continue

            cropped_slice = full_slice_2d[t_start:t_end, trace_start:trace_end]

            # --- 【关键修改】: 在保存前显示图像 ---
            if show_plot:
                preview_title = f"预览: {base_filename}\n裁剪后的 {dim.capitalize()} {index} 切片"
                show_slice_preview(cropped_slice, preview_title)

            # 构建并保存文件
            filename_without_ext = os.path.splitext(base_filename)[0]
            output_filename = f"{filename_without_ext}_{dim}{index}_cropped.npy"
            output_path = os.path.join(dest_dir, output_filename)
            np.save(output_path, cropped_slice)
            print(f"    -> 成功保存切片，形状为 {cropped_slice.shape} 到: {output_filename}")

        except Exception as e:
            print(f"    - 处理文件 {base_filename} 时发生严重错误: {e}")

    print("\n所有文件处理完毕！")


# --- 运行主程序 ---
if __name__ == "__main__":
    batch_extract_and_crop_slices(
        source_dir=SOURCE_FOLDER,
        dest_dir=DESTINATION_FOLDER,
        dim=SLICE_DIM,
        index=SLICE_INDEX,
        t_start=TIME_CROP_START,
        t_end=TIME_CROP_END,
        trace_start=TRACE_CROP_START,
        trace_end=TRACE_CROP_END,
        show_plot=SHOW_PLOT_PREVIEW  # 将配置开关传入主函数
    )