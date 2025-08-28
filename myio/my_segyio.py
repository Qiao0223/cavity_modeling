import json
import os
import glob
import segyio
import numpy as np


class SegyConverter:
    """
    一个健壮的 SEG-Y <-> NumPy (.npy) 转换工具。

    该工具的核心是解耦数据和元数据：
    - .npy 文件仅存储纯粹的N维数组。
    - 导入 (SEG-Y -> .npy): 依赖外部信息（如配置文件）来定义三维数据体的维度，
      适用于道头信息不规范的SEG-Y文件。
    - 导出 (.npy -> SEG-Y): 依赖一个“参考SEG-Y文件”来提供完整的头部信息，
      确保生成的SEG-Y文件格式正确。
    """

    def import_segy_to_npy(self, segy_path: str, out_dir: str,
                           num_inline: int, num_xline: int) -> None:
        """
        将单个 SEG-Y 文件转换为 .npy 文件，并自动在输出目录中创建同名文件。

        Args:
            segy_path (str): 输入的 SEG-Y 文件路径。
            out_dir (str): 保存 .npy 文件的输出目录。
            num_inline (int): 手动指定的 inline 数量。
            num_xline (int): 手动指定的 xline 数量。
        """
        # --- 新增：自动生成输出路径 ---
        base_name = os.path.splitext(os.path.basename(segy_path))[0]
        out_npy_path = os.path.join(out_dir, f"{base_name}.npy")
        # --- 修改结束 ---

        print(f"开始导入: {segy_path}")
        print(f"维度: Inlines={num_inline}, Xlines={num_xline}")

        try:
            with segyio.open(segy_path, 'r', ignore_geometry=True) as segy_file:
                traces = segyio.collect(segy_file.trace)
                arr2d = np.asarray(list(traces))

                expected_traces = num_inline * num_xline
                if arr2d.shape[0] != expected_traces:
                    raise ValueError(
                        f"道数不匹配！期望 {expected_traces}, 但文件中有 {arr2d.shape[0]} 道。"
                    )

                cube = arr2d.reshape(num_inline, num_xline, arr2d.shape[1])

                os.makedirs(os.path.dirname(out_npy_path), exist_ok=True)
                np.save(out_npy_path, cube)
                print(f"成功保存到: {out_npy_path}, 数据维度: {cube.shape}")

        except Exception as e:
            print(f"处理文件 {segy_path} 时出错: {e}")

    def export_npy_to_segy(self, npy_path: str, ref_segy_path: str, out_dir: str) -> None:
        """
        将 .npy 文件导出为 SEG-Y 文件，并自动在输出目录中创建同名文件。

        Args:
            npy_path (str): 输入的 .npy 文件路径。
            ref_segy_path (str): 用于提供头部和几何信息的参考 SEG-Y 文件。
            out_dir (str): 保存 SEG-Y 文件的输出目录。
        """
        # --- 新增：自动生成输出路径 ---
        base_name = os.path.splitext(os.path.basename(npy_path))[0]
        out_segy_path = os.path.join(out_dir, f"{base_name}.segy")
        # --- 修改结束 ---

        print(f"开始从 {npy_path} 导出...")

        data_cube = np.load(npy_path)
        if data_cube.ndim != 3:
            raise ValueError(f"输入 NumPy 数组必须是三维的，但当前维度为 {data_cube.ndim}")

        with segyio.open(ref_segy_path, 'r', ignore_geometry=True) as src_segy:
            spec = segyio.spec()
            spec.ilines = src_segy.ilines
            spec.xlines = src_segy.xlines
            spec.samples = range(data_cube.shape[2])
            spec.format = src_segy.format
            spec.sorting = src_segy.sorting

            if (data_cube.shape[0] != len(src_segy.ilines) or
                    data_cube.shape[1] != len(src_segy.xlines)):
                print(f"[警告] NumPy 数组维度 ({data_cube.shape[:2]}) "
                      f"与参考SEG-Y维度 ({len(src_segy.ilines)}, {len(src_segy.xlines)}) 不匹配。")

            os.makedirs(os.path.dirname(out_segy_path), exist_ok=True)

            with segyio.create(out_segy_path, spec) as dst_segy:
                dst_segy.text[0] = src_segy.text[0]
                dst_segy.bin = src_segy.bin
                dst_segy.header = src_segy.header
                dst_segy.trace = data_cube.reshape(-1, data_cube.shape[2])

        print(f"成功导出 SEG-Y 文件到: {out_segy_path}")

    def import_all(self, in_dir: str, out_dir: str, num_inline: int, num_xline: int) -> None:
        """批量导入。此函数逻辑无需更改，因为它已经实现了自动命名。"""
        os.makedirs(out_dir, exist_ok=True)
        patterns = ['*.sgy', '*.segy']
        for pattern in patterns:
            for segy_file in glob.glob(os.path.join(in_dir, pattern)):
                # 这里直接调用修改后的单文件导入函数
                self.import_segy_to_npy(segy_file, out_dir, num_inline, num_xline)

    def export_all(self, in_dir: str, ref_segy_path: str, out_dir: str) -> None:
        """批量导出。此函数逻辑无需更改，因为它已经实现了自动命名。"""
        os.makedirs(out_dir, exist_ok=True)
        for npy_file in glob.glob(os.path.join(in_dir, '*.npy')):
            # 这里直接调用修改后的单文件导出函数
            self.export_npy_to_segy(npy_file, ref_segy_path, out_dir)


# --- 示例调用 ---
if __name__ == '__main__':
    converter = SegyConverter()

    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print("错误: 找不到 'config.json' 文件。")
        exit()

    params = config['yingxi_crop']
    num_inline = params['inline_end'] - params['inline_start'] + 1
    num_xline = params['xline_end'] - params['xline_start'] + 1

    INPUT_SEGY_DIR = '../data/input_segy'
    INPUT_NPY_DIR = '../data/input_npy'
    OUTPUT_SEGY_DIR = '../data/output_segy'
    ref_segy_for_export_path = ""

    # --- 场景1: 导入单个 SEGY 文件 (调用更简洁) ---
    print("\n--- 场景1: 导入单个文件 ---")
    segy_to_import_path = os.path.join(INPUT_SEGY_DIR, 'yingxi_crop.segy')
    converter.import_segy_to_npy(
        segy_path=segy_to_import_path,
        out_dir=INPUT_NPY_DIR,  # 只需提供输出目录
        num_inline=num_inline,
        num_xline=num_xline
    )

    # --- 场景2: 导出单个 NPY 文件 (调用更简洁) ---
    # print("\n--- 场景2: 导出单个文件 ---")
    # npy_to_export_path = os.path.join(INPUT_NPY_DIR, 'yingxi_xline_dip.npy')
    # converter.export_npy_to_segy(
    #     npy_path=npy_to_export_path,
    #     ref_segy_path=ref_segy_for_export_path,
    #     out_dir=OUTPUT_SEGY_DIR  # 只需提供输出目录
    # )

    # --- 场景3: 批量导入 (调用不变，但内部逻辑更统一) ---
    # print("\n--- 场景3: 批量导入 ---")
    # converter.import_all(
    #        in_dir = INPUT_SEGY_DIR,
    #     out_dir=INPUT_NPY_DIR,
    #     num_inline=num_inline,
    #     num_xline=num_xline
    # )

    # --- 场景4: 批量导出 (调用不变，但内部逻辑更统一) ---
    # print("\n--- 场景4: 批量导出 ---")
    # converter.export_all(
    #     in_dir=INPUT_NPY_DIR,
    #     ref_segy_path=ref_segy_for_export_path,
    #     out_dir=OUTPUT_SEGY_DIR
    # )