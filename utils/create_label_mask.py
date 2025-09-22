# --- 【关键修复】: 强制使用交互式绘图后端 ---
import matplotlib

matplotlib.use('TkAgg')  # 必须在导入pyplot之前设置！

import numpy as np
import matplotlib.pyplot as plt
import os
from roipoly import RoiPoly
import matplotlib as mpl

# ==============================================================================
# --- 1. 配置 ---
# ==============================================================================
BASE_SLICE_PATH = r"C:\Work\sunjie\Python\cavity_modeling\data\2d\yingxi_crop_wl_inline118_cropped.npy"
LABELS_FOLDER = r"C:\Work\sunjie\Python\cavity_modeling\data\labels"
MASK_FILENAME = "label_mask_inline118.npy"

# ==============================================================================
# --- 2. 交互式绘图程序 (已修复) ---
# ==============================================================================

# 解决中文字体警告
try:
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
except Exception:
    print("警告：系统中未找到 'SimHei' 字体，标题中的中文可能显示为方框。")

# 加载底图数据
try:
    base_slice = np.load(BASE_SLICE_PATH)
    print(f"成功加载底图: {BASE_SLICE_PATH}")
except FileNotFoundError:
    print(f"错误：底图文件未找到！请检查路径: {BASE_SLICE_PATH}")
    exit()

os.makedirs(LABELS_FOLDER, exist_ok=True)

# --- 开始交互式绘图 ---
print("\n" + "=" * 50)
print("交互式绘图窗口已弹出。")
print("操作指南:")
print("  - 左键单击: 在图像上添加多边形顶点。")
print("  - 右键单击: 完成当前多边形的绘制。")
print("  - 您可以绘制多个不连续的多边形区域。")
print("  - 全部绘制完成后，关闭绘图窗口即可。")
print("=" * 50)

fig, ax = plt.subplots()
ax.imshow(base_slice, cmap='seismic', aspect='auto')
ax.set_title("请圈定'缝洞体'区域 (左键添加点, 右键闭合)")

# 强制显示窗口并进入主循环，确保窗口弹出并等待
plt.show(block=False)

# 在显示的图上启动RoiPoly
my_roi = RoiPoly(fig=fig, ax=ax, color='r')

# --- 【重要】: 再次调用 show() 并设置 block=True ---
# 这会暂停脚本，直到 roipoly 操作完成且窗口被关闭
plt.show(block=True)

# --- 后处理 ---
if not my_roi.x or not my_roi.y:
    print("\n警告：您没有绘制任何区域就关闭了窗口，未生成标签文件。")
else:
    mask = my_roi.get_mask(base_slice)
    if mask is None or not np.any(mask):
        print("\n警告：未能成功创建掩模，未生成标签文件。")
    else:
        mask_int = mask.astype(np.uint8)
        output_path = os.path.join(LABELS_FOLDER, MASK_FILENAME)
        np.save(output_path, mask_int)

        print(f"\n成功！标签掩模已保存到: {output_path}")
        print(f"掩模形状: {mask_int.shape}")
        print(f"缝洞体像素点数: {np.sum(mask_int)}")

        # (可选) 显示一下您创建的掩模
        plt.figure()
        plt.title("您创建的标签掩模 (1=缝洞体, 0=背景)")
        plt.imshow(mask_int, cmap='viridis', aspect='auto')
        plt.colorbar()
        plt.show()

print("\n程序执行完毕。")