import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import my_segyio

# def map_positions_to_array(array, positions, origin, steps):
#     """
#     将三列表示坐标的 NumPy 数组映射到已有的三维 NumPy 数组 `array` 上。
#     - 如果计算出的索引超出 `array` 的边界，则跳过并警告。
#     - 位置坐标四舍五入到最近的 `array` 点。
#     - 赋值 `1` 到映射点上。
#
#     参数：
#     - array: 现有的 3D NumPy 数组
#     - positions: (N, 3) 形状的 NumPy 数组，每行存储 (x, y, z) 位置
#     - origin: (origin_x, origin_y, origin_z) `array[0,0,0]` 对应的实际坐标
#     - steps: (step_x, step_y, step_z) `array` 中点之间的间距
#
#     返回：
#     - 处理后的 `array`
#     """
#     origin_x, origin_y, origin_z = origin
#     step_x, step_y, step_z = steps
#     shape_x, shape_y, shape_z = array.shape  # 获取 array 形状
#
#     for x, y, z in positions:
#         # 计算索引并四舍五入
#         i = round((x - origin_x) / step_x)
#         j = round((y - origin_y) / step_y)
#         k = round((z - origin_z) / step_z)
#
#         # 检查是否超出 array 范围
#         if 0 <= i < shape_x and 0 <= j < shape_y and 0 <= k < shape_z:
#             array[i, j, k] = 1  # 赋值 1
#         else:
#             print(f"⚠️ Warning: Position ({x}, {y}, {z}) maps to ({i}, {j}, {k}), out of bounds!")
#
#     return array

def rotate_index(i, j, origin_i, origin_j, angle_degrees):
    """
    以 (origin_i, origin_j) 为中心，将索引 (i, j) 顺时针旋转指定角度（度数）。

    参数：
    - i, j: 需要旋转的索引
    - origin_i, origin_j: 旋转中心（在索引空间）
    - angle_degrees: 旋转角度（单位：度）

    返回：
    - 旋转后的 (new_i, new_j) 索引
    """
    angle_radians = np.radians(-angle_degrees)  # 负号表示顺时针旋转
    cos_theta = np.cos(angle_radians)
    sin_theta = np.sin(angle_radians)

    # 平移到原点进行旋转，再平移回去
    i_shifted = i - origin_i
    j_shifted = j - origin_j
    new_i = i_shifted * cos_theta - j_shifted * sin_theta + origin_i
    new_j = i_shifted * sin_theta + j_shifted * cos_theta + origin_j

    # 由于索引必须是整数，四舍五入
    return round(new_i), round(new_j)


def map_positions_to_array(array, positions, origin, steps, angle_degrees):
    """
    先将 `positions` 映射到 `array` 索引空间，再对索引进行顺时针旋转，并赋值 `1`。

    参数：
    - array: 现有的 3D NumPy 数组
    - positions: (N, 3) 形状的 NumPy 数组，每行存储 (x, y, z) 位置
    - origin: (origin_x, origin_y, origin_z) `array[0,0,0]` 对应的实际坐标
    - steps: (step_x, step_y, step_z) `array` 中点之间的间距
    - angle_degrees: 顺时针旋转角度（单位：度）

    返回：
    - 处理后的 `array`
    """
    origin_x, origin_y, origin_z = origin
    step_x, step_y, step_z = steps
    shape_x, shape_y, shape_z = array.shape  # 获取 array 形状

    # 计算 `origin` 在 `array` 索引空间的位置
    origin_i = round((origin_x - origin_x) / step_x)  # 一定是 0
    origin_j = round((origin_y - origin_y) / step_y)  # 一定是 0

    for x, y, z in positions:
        # 计算索引
        i = round((x - origin_x) / step_x)
        j = round((y - origin_y) / step_y)
        k = round((z - origin_z) / step_z)

        # 旋转索引 (i, j)
        rotated_i, rotated_j = rotate_index(i, j, origin_i, origin_j, angle_degrees)

        # 检查是否超出 array 范围
        if 0 <= rotated_i < shape_x and 0 <= rotated_j < shape_y and 0 <= k < shape_z:
            array[rotated_i, rotated_j, k] = 1  # 赋值 1
        else:
            print(f"⚠️ Warning: Rotated index ({rotated_i}, {rotated_j}, {k}) is out of bounds!")

    return array


def plot_nonzero_points(array):
    """
    在 3D 空间中显示三维 NumPy 数组中非零点的位置。

    参数：
    - array: 3D NumPy 数组
    """
    # 找到所有非零点的索引
    nonzero_indices = np.argwhere(array != 0)

    # 获取 x, y, z 坐标
    x = nonzero_indices[:, 0]
    y = nonzero_indices[:, 1]
    z = nonzero_indices[:, 2]

    # 创建 3D 绘图
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制非零点
    ax.scatter(x, y, z, c='red', marker='o', label='Nonzero Points')

    # 设置轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Visualization of Nonzero Points in 3D Array')

    # 显示图例
    ax.legend()

    # 显示图像
    plt.show()

def apply_influence_sphere(matrix, r, value_function):
    """
    以 3D 矩阵中的 `1` 为中心点，在半径 `r` 的球体内影响其他点，并显示进度条。

    参数：
    - matrix: (3D NumPy array) 仅包含 0 和 1 的三维矩阵。
    - r: (float) 影响球体的半径。
    - value_function: (function) 计算值的函数，参数是到中心点的距离，返回值范围为 [0,1]。

    返回:
    - 更新后的三维矩阵
    """
    # 复制矩阵，避免修改原始数据
    updated_matrix = matrix.copy()
    shape_x, shape_y, shape_z = matrix.shape

    # 找到所有值为 1 的点
    ones_positions = np.argwhere(matrix == 1)

    # 设置 tqdm 进度条
    total_points = len(ones_positions)  # 总共要处理的 1 的个数
    progress_bar = tqdm(ones_positions, desc="Processing Points", unit="point")

    # 遍历每个 1 的点
    for center_x, center_y, center_z in progress_bar:
        # 遍历该点 r 范围内的所有可能点
        for i in range(-r, r + 1):
            for j in range(-r, r + 1):
                for k in range(-r, r + 1):
                    # 计算目标点坐标
                    new_x, new_y, new_z = center_x + i, center_y + j, center_z + k

                    # 计算到中心点的欧几里得距离
                    distance = np.sqrt(i ** 2 + j ** 2 + k ** 2)

                    # 只处理球体内部 (distance ≤ r)
                    if distance <= r:
                        # 边界检查，确保索引合法
                        if 0 <= new_x < shape_x and 0 <= new_y < shape_y and 0 <= new_z < shape_z:
                            new_value = value_function(distance)
                            # 只有当新计算值大于当前值时才更新
                            if new_value > updated_matrix[new_x, new_y, new_z]:
                                updated_matrix[new_x, new_y, new_z] = min(1, new_value)  # 确保不超过 1

    return updated_matrix



if __name__ == "__main__":
    #seis = np.load("numpy/LC1JX_BGP_20221026-TTI-psdmr.npy")
    fault_point = np.loadtxt("fault_point.txt")
    shape = (1041, 781, 2701)
    array = np.zeros(shape, dtype=np.float32)

    angle = 1.534
    origin = (14603605, 4510088, -1500)
    steps = (25, 25, 5)
    array = map_positions_to_array(array, fault_point, origin, steps,angle)

    r = 8
    # 自定义影响函数 (线性衰减)
    def custom_value_function(distance, r):
        return max(0, 1 - distance / r)  # 线性衰减，确保范围在 [0,1]

    array = apply_influence_sphere(array,r,lambda d: custom_value_function(d, r))

    # 保存为 .npy 文件
    np.save("numpy/array.npy", array)

    # 指定 inline 和 xline 的数量
    inline_start = 1400
    inline_end = 2440
    xline_start = 1000
    xline_end = 1780

    num_inline = inline_end-inline_start+1
    num_xline = xline_end-xline_start+1

