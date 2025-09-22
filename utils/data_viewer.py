import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# --- 0. 准备工作：设置中文字体 ---
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False


# --- 1. 定义最终版函数 ---
def plot_seismic_slice(data, slice_dim, slice_index, cmap='seismic',
                       vmin=None, vmax=None, xlim=None, ylim=None,
                       symmetrical_cbar=True):
    """
    以纯彩色风格显示三维地震数据的指定切片。
    此版本使用 `extent` 参数来正确处理坐标轴和图像方向。
    """
    n_inline, n_xline, n_time = data.shape

    # --- 提取切片并设置标签 (这部分逻辑不变) ---
    if slice_dim.lower() == 'inline':
        if not 0 <= slice_index < n_inline:
            print(f"错误: Inline 索引 {slice_index} 超出范围。")
            return
        slice_2d = data[slice_index, :, :].T
        title = f'Inline 切片 (Inline = {slice_index})'
        xlabel = 'Crossline'
        ylabel = 'Time / Depth'
        # 获取默认的坐标范围
        default_xlim = [0, n_xline]
        default_ylim = [0, n_time]

    elif slice_dim.lower() == 'xline':
        if not 0 <= slice_index < n_xline:
            print(f"错误: Crossline 索引 {slice_index} 超出范围。")
            return
        slice_2d = data[:, slice_index, :].T
        title = f'Crossline 切片 (Crossline = {slice_index})'
        xlabel = 'Inline'
        ylabel = 'Time / Depth'
        default_xlim = [0, n_inline]
        default_ylim = [0, n_time]

    elif slice_dim.lower() == 'time':
        if not 0 <= slice_index < n_time:
            print(f"错误: Time 索引 {slice_index} 超出范围。")
            return
        slice_2d = data[:, :, slice_index]
        title = f'Time / Depth 切片 (Sample = {slice_index})'
        xlabel = 'Crossline'
        ylabel = 'Inline'
        default_xlim = [0, n_xline]
        default_ylim = [0, n_inline]
    else:
        print("错误: `slice_dim` 参数必须是 'inline', 'xline' 或 'time'。")
        return

    # --- 对称色标逻辑 (不变) ---
    if symmetrical_cbar and vmin is None and vmax is None:
        max_pos = np.max(slice_2d)
        min_neg = np.min(slice_2d)
        if max_pos > 0 and min_neg < 0:
            limit = min(max_pos, abs(min_neg))
            vmin = -limit
            vmax = limit
            print(f"自动设置对称色标 (基于较小振幅): [{vmin:.2f}, {vmax:.2f}] (原始范围: [{min_neg:.2f}, {max_pos:.2f}])")
        else:
            abs_max = np.percentile(np.abs(slice_2d), 99)
            if abs_max > 0:
                vmin = -abs_max
                vmax = abs_max
                print(f"数据单边，自动设置对称色标 (99th percentile): [{vmin:.2f}, {vmax:.2f}]")

    # --- 【关键修改】: 构建 extent 参数 ---
    # 如果用户没有提供范围，则使用数据的维度作为默认范围
    final_xlim = xlim if xlim is not None else default_xlim
    final_ylim = ylim if ylim is not None else default_ylim

    # 对于垂直剖面（inline, xline），Y轴（时间/深度）应该向下增加
    # 所以 extent 的 bottom > top
    if slice_dim.lower() in ['inline', 'xline']:
        plot_extent = [final_xlim[0], final_xlim[1], final_ylim[1], final_ylim[0]]
    else:  # 对于时间切片，Y轴（Inline号）正常向上增加
        plot_extent = [final_xlim[0], final_xlim[1], final_ylim[0], final_ylim[1]]


    # --- 绘图核心 ---
    plt.figure(figsize=(12, 8))
    # 使用 extent 参数，同时 aspect='auto' 保证图像填满坐标轴
    plt.imshow(slice_2d, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax, extent=plot_extent)

    # --- 图表美化 ---
    plt.colorbar(label='振幅 (Amplitude)')
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)

    # 【不再需要】: 不再需要手动设置 xlim 和 ylim，extent 已经处理好了
    # if xlim is not None:
    #     plt.xlim(xlim)
    # if ylim is not None:
    #     plt.ylim(ylim)

    plt.show()


# --- 2. 加载数据 ---
file_path = r"C:\Work\sunjie\Python\cavity_modeling\data\input_npy\luchang\LC_CUT.npy"
try:
    seismic_data_3d = np.load(file_path)
    print(f"成功加载数据: {file_path}")
except FileNotFoundError:
    print(f"错误：文件未找到，请检查路径: {file_path}")
    exit()

# --- 3. 调用函数 ---
# 您无需更改调用方式，函数内部的逻辑已经更新
print("\n--- 显示剖面，自动启用新的对称色标逻辑 ---")

x_axis_range = [0, 781]
y_axis_range = [0, 565]

plot_seismic_slice(data=seismic_data_3d,
                   slice_dim='inline',
                   slice_index=200,
                   xlim=x_axis_range,
                   ylim=y_axis_range)