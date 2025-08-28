import json
import os
import pandas as pd
import numpy as np


class HorizonProcessor:
    """
    (类的文档字符串保持不变)
    """

    def __init__(self, config_path: str):
        print(f"--- 初始化 HorizonProcessor ---")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"错误: 配置文件未找到 -> {config_path}")
        with open(config_path, 'r') as f:
            self.full_config = json.load(f)
        print("配置文件加载成功。")
        self.config = None
        self.grid_shape = None

    def _load_horizon_data(self, horizon_path: str) -> pd.DataFrame:
        if not os.path.exists(horizon_path):
            raise FileNotFoundError(f"错误: 层位文件未找到 -> {horizon_path}")
        print(f"正在读取层位文件: {horizon_path}")
        col_names = ['type1', 'colon1', 'inline', 'type2', 'colon2', 'xline', 'x_coord', 'y_coord', 'depth']
        df = pd.read_csv(
            horizon_path, header=None, sep=r'\s+', names=col_names, engine='python'
        )
        horizon_df = df[['inline', 'xline', 'depth']].copy()
        print(f"成功解析 {len(horizon_df)} 个层位点。")
        return horizon_df

    # --- 关键修改点 1: 函数签名增加 segy_npy_path 参数 ---
    def create_index_grid(self, horizon_path: str, survey_name: str, segy_npy_path: str,
                          invalid_value: int = -1) -> np.ndarray:
        """
        创建并返回层位索引网格。

        Args:
            horizon_path (str): 层位数据文件的路径。
            survey_name (str): 要在配置文件中使用的工区名称。
            segy_npy_path (str): 对应的地震数据 .npy 文件路径，用于获取Z轴范围。
            invalid_value (int, optional): 用于填充无数据位置的值。默认为 -1。

        Returns:
            np.ndarray: 生成的二维索引网格。
        """
        print(f"\n--- 开始为工区 '{survey_name}' 创建索引网格 ---")
        if survey_name not in self.full_config:
            raise KeyError(f"错误: 在 config.json 中未找到名为 '{survey_name}' 的工区配置。")
        self.config = self.full_config[survey_name]

        # --- 关键修改点 2: 自动获取 num_samples ---
        if not os.path.exists(segy_npy_path):
            raise FileNotFoundError(f"错误: 找不到用于获取Z轴范围的地震数据文件 -> {segy_npy_path}")

        # 使用 memory map 模式，即使文件很大，也只读取元数据，速度很快
        seismic_data = np.load(segy_npy_path, mmap_mode='r')
        num_samples = seismic_data.shape[2]
        print(f"从 {os.path.basename(segy_npy_path)} 自动获取到 Z 轴采样点数: {num_samples}")

        # 后续逻辑与之前版本完全相同
        horizon_df = self._load_horizon_data(horizon_path)

        inline_start, inline_end = self.config['inline_start'], self.config['inline_end']
        xline_start, xline_end = self.config['xline_start'], self.config['xline_end']
        num_inline = inline_end - inline_start + 1
        num_xline = xline_end - xline_start + 1
        self.grid_shape = (num_inline, num_xline)
        print(f"目标网格维度 (Inlines, Xlines): {self.grid_shape}")

        index_grid = np.full(self.grid_shape, invalid_value, dtype=np.int32)

        row_indices = horizon_df['inline'].values - inline_start
        col_indices = horizon_df['xline'].values - xline_start
        spatial_mask = (row_indices >= 0) & (row_indices < num_inline) & \
                       (col_indices >= 0) & (col_indices < num_xline)

        if np.sum(spatial_mask) < len(horizon_df):
            print(f"警告: {len(horizon_df) - np.sum(spatial_mask)} 个层位点超出了 Inline/Xline 范围，将被忽略。")

        valid_rows = row_indices[spatial_mask]
        valid_cols = col_indices[spatial_mask]
        valid_depths = horizon_df['depth'].values[spatial_mask]

        if len(valid_depths) == 0:
            print("错误: 没有任何层位点落在定义的网格范围内。")
            return index_grid

        z_start, dt = self.config['z_start'], self.config['dt']
        sample_indices = np.round((valid_depths - z_start) / dt).astype(np.int32)

        z_range_mask = (sample_indices >= 0) & (sample_indices < num_samples)
        if np.sum(z_range_mask) < len(sample_indices):
            print(f"警告: {len(sample_indices) - np.sum(z_range_mask)} 个层位点的深度超出了有效范围，将被忽略。")
            print(f"       有效索引范围为 [0, {num_samples - 1}]。")

        final_rows = valid_rows[z_range_mask]
        final_cols = valid_cols[z_range_mask]
        final_indices = sample_indices[z_range_mask]

        index_grid[final_rows, final_cols] = final_indices
        print("深度值已成功转换为采样点索引并填充到网格中。")

        return index_grid

    def save_grid(self, grid: np.ndarray, output_path: str):
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        np.save(output_path, grid)
        print(f"\n索引网格已成功保存到: {output_path}")


# --- 示例调用 ---
if __name__ == '__main__':
    CONFIG_FILE_PATH = 'config.json'

    # --- 关键修改点 3: 增加地震NPY文件的路径 ---
    # 这个路径指向您的 SegyConverter 转换出的地震数据
    SEGY_NPY_PATH = r'C:\Work\sunjie\Python\cavity_modeling\data\input_npy\yingxi_crop.npy'

    HORIZON_FILE_PATH = r'C:\Work\sunjie\Python\cavity_modeling\data\input_segy\yingxi_hor_TO3t.txt'
    OUTPUT_DIR = r'C:\Work\sunjie\Python\cavity_modeling\data\input_npy'
    SURVEY_NAME = 'yingxi_crop'

    base_name = os.path.basename(HORIZON_FILE_PATH)
    file_name_without_ext = os.path.splitext(base_name)[0]
    OUTPUT_NPY_PATH = os.path.join(OUTPUT_DIR, f"{file_name_without_ext}.npy")

    try:
        processor = HorizonProcessor(config_path=CONFIG_FILE_PATH)

        # --- 关键修改点 4: 传递地震NPY路径 ---
        horizon_index_grid = processor.create_index_grid(
            horizon_path=HORIZON_FILE_PATH,
            survey_name=SURVEY_NAME,
            segy_npy_path=SEGY_NPY_PATH  # 传递路径
        )

        if horizon_index_grid is not None:
            processor.save_grid(horizon_index_grid, output_path=OUTPUT_NPY_PATH)

        print("\n--- 验证 ---")
        print(f"输出网格的形状: {horizon_index_grid.shape}")
        valid_points_in_grid = np.sum(horizon_index_grid != -1)
        print(f"网格中有效(非-1)数据点的数量: {valid_points_in_grid}")

    except (FileNotFoundError, KeyError, Exception) as e:
        print(f"\n程序执行出错: {e}")