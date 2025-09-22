import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- 设置Matplotlib以支持中文显示 ---
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("--- 开始使用 Matplotlib 进行三维散点图可视化 (XY轴在底面) ---")

# --- 1. 设置文件路径和参数 ---
file_path = r'C:\Work\sunjie\Python\cavity_modeling\data\input_npy\luchang_cavity_module.npy'
threshold = 0.7
downsample_factor = 8

# --- 2. 加载、筛选和降采样数据 (与之前相同) ---
try:
    seismic_data = np.load(file_path)
    print(f"成功加载数据，原始维度为: {seismic_data.shape}")
except FileNotFoundError:
    print(f"错误：文件未找到，请检查路径: {file_path}")
    exit()

print(f"正在应用阈值 {threshold} 并提取坐标...")
z, y, x = np.where(seismic_data > threshold)
print(f"找到 {len(x)} 个高于阈值的点。")

if len(x) > 200000:
    print(f"点数过多，将进行 1/{downsample_factor} 的降采样...")
    x = x[::downsample_factor]
    y = y[::downsample_factor]
    z = z[::downsample_factor]
    print(f"降采样后剩余 {len(x)} 个点。")

values = seismic_data[z, y, x]

# --- 3. 开始绘图 ---
print("正在创建三维图形...")
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(x, y, z, c=values, cmap='viridis', s=1, marker='.')

# --- 4. 【关键步骤】将XY轴固定在底面 ---

# (1) 将三个坐标轴的背景“墙面”设置为完全透明
#    RGBA颜色中，最后一个值(alpha)为0代表完全透明
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

# (2) 开启网格线，它们会画在透明的背景面上，形成参考框架
ax.grid(True)


# --- 5. 美化图形 ---
ax.set_title('缝洞体三维散点图可视化 (XY轴在底面)', fontsize=16)
ax.set_xlabel('X轴 (Crossline/Inline)', fontsize=12)
ax.set_ylabel('Y轴 (Crossline/Inline)', fontsize=12)
ax.set_zlabel('Z轴 (Time/Depth)', fontsize=12)

# 保持Z轴反转，这对于将XY平面置于“顶部”（视觉上的底部）至关重要
ax.invert_zaxis()

fig.colorbar(scatter, shrink=0.7, aspect=20, label='缝洞体概率')

print("绘图完成，正在显示窗口...")
plt.show()