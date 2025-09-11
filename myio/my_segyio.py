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
        将单个 SEG-Y 文件转换为 .npy 文件。

        Args:
            segy_path (str): 输入的 SEG-Y 文件的完整路径。
            out_dir (str): 保存 .npy 文件的输出目录。
            num_inline (int): 手动指定的 inline 数量。
            num_xline (int): 手动指定的 xline 数量。
        """
        base_name = os.path.splitext(os.path.basename(segy_path))[0]
        out_npy_path = os.path.join(out_dir, f"{base_name}.npy")

        print(f"开始导入: {segy_path}")
        print(f"维度: Inlines={num_inline}, Xlines={num_xline}")

        try:
            # 导入时可以忽略几何，因为我们是根据指定的维度来重塑数据的
            with segyio.open(segy_path, 'r', ignore_geometry=True) as segy_file:
                traces = segyio.collect(segy_file.trace)
                arr2d = np.asarray(list(traces))

                expected_traces = num_inline * num_xline
                if arr2d.shape[0] != expected_traces:
                    raise ValueError(
                        f"道数不匹配！期望 {expected_traces} 道, 但文件中有 {arr2d.shape[0]} 道。"
                    )

                cube = arr2d.reshape(num_inline, num_xline, arr2d.shape[1])

                os.makedirs(out_dir, exist_ok=True)
                np.save(out_npy_path, cube)
                print(f"成功保存到: {out_npy_path}, 数据维度: {cube.shape}")

        except Exception as e:
            print(f"处理文件 {segy_path} 时出错: {e}")

    def export_npy_to_segy(self, npy_path: str, ref_segy_path: str, out_dir: str) -> None:
        """
        将 .npy 文件导出为 SEG-Y 文件，完整复用参考文件的头部信息。
        采用精确复制模式，确保生成的文件结构与参考文件一致。

        Args:
            npy_path (str): 输入的 .npy 文件的完整路径。
            ref_segy_path (str): 用于提供头部信息的模板 SEG-Y 文件的完整路径。
            out_dir (str): 保存 SEG-Y 文件的输出目录。
        """
        base_name = os.path.splitext(os.path.basename(npy_path))[0]
        out_segy_path = os.path.join(out_dir, f"{base_name}.segy")

        print(f"开始从 {npy_path} 导出，并复用 {os.path.basename(ref_segy_path)} 的道头...")

        data_cube = np.load(npy_path)
        if data_cube.ndim != 3:
            raise ValueError(f"输入 NumPy 数组必须是三维的，但当前维度为 {data_cube.ndim}")

        # 打开参考文件，必须读取其几何结构
        with segyio.open(ref_segy_path, 'r', ignore_geometry=False) as src_segy:

            # --- 严格的维度检查 ---
            num_traces_npy = data_cube.shape[0] * data_cube.shape[1]
            num_samples_npy = data_cube.shape[2]

            num_traces_segy = src_segy.tracecount
            num_samples_segy = src_segy.samples.size

            if num_traces_npy != num_traces_segy:
                raise ValueError(
                    f"道数不匹配！NPY文件有 {num_traces_npy} 道，"
                    f"而参考SEGY文件有 {num_traces_segy} 道。"
                )

            if num_samples_npy != num_samples_segy:
                raise ValueError(
                    f"采样点数（道长）不匹配！NPY文件有 {num_samples_npy} 个采样点，"
                    f"而参考SEGY文件有 {num_samples_segy} 个。"
                )
            # --- 检查结束 ---

            # 创建一个空spec对象，然后从源文件手动复制属性，确保结构一致
            spec = segyio.spec()
            spec.ilines = src_segy.ilines
            spec.xlines = src_segy.xlines
            spec.samples = src_segy.samples
            spec.format = src_segy.format
            spec.sorting = src_segy.sorting

            os.makedirs(out_dir, exist_ok=True)

            with segyio.create(out_segy_path, spec) as dst_segy:
                # 1. 完整复制文本头和二进制卷头
                dst_segy.text[0] = src_segy.text[0]
                dst_segy.bin = src_segy.bin

                # 2. 完整复制所有的道头
                dst_segy.header = src_segy.header

                # 3. 写入新的地震道数据
                # 将3D的numpy数组重塑为2D的(道数, 采样点数)
                dst_segy.trace = data_cube.reshape(num_traces_segy, num_samples_segy)

        print(f"成功导出 SEG-Y 文件到: {out_segy_path}")

    def import_all(self, in_dir: str, out_dir: str, num_inline: int, num_xline: int) -> None:
        """批量导入目录中的所有 SEG-Y 文件。"""
        os.makedirs(out_dir, exist_ok=True)
        patterns = ['*.sgy', '*.segy']
        print(f"\n--- 开始批量导入: 从 {in_dir} ---")
        for pattern in patterns:
            for segy_file in glob.glob(os.path.join(in_dir, pattern)):
                self.import_segy_to_npy(segy_file, out_dir, num_inline, num_xline)
        print("--- 批量导入完成 ---")

    def export_all(self, in_dir: str, ref_segy_path: str, out_dir: str) -> None:
        """批量导出目录中的所有 .npy 文件。"""
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n--- 开始批量导出: 从 {in_dir} ---")
        for npy_file in glob.glob(os.path.join(in_dir, '*.npy')):
            try:
                self.export_npy_to_segy(npy_file, ref_segy_path, out_dir)
            except Exception as e:
                print(f"批量导出文件 {npy_file} 失败: {e}")
        print("--- 批量导出完成 ---")


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

    # --- 路径定义 (推荐使用相对路径和os.path.join) ---
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'data')

    INPUT_SEGY_DIR = os.path.join(DATA_DIR, 'input_segy')
    INPUT_NPY_DIR = os.path.join(DATA_DIR, 'input_npy')
    OUTPUT_NPY_DIR = os.path.join(DATA_DIR, 'output_npy')
    OUTPUT_SEGY_DIR = os.path.join(DATA_DIR, 'output_segy')
    BATCH_DIR = os.path.join(DATA_DIR, 'batch')
    # 定义一个统一的参考文件路径
    ref_segy_for_export_path = r"C:\Work\sunjie\Python\cavity_modeling\data\input_segy\yingxi_crop.segy"

    # =============================================================================
    #   使用时，请取消下面您想运行的场景的注释
    # =============================================================================

    # --- 场景1: 导入单个文件 ---
    # print("\n==================== 场景1: 导入单个文件 ====================")
    # segy_to_import_path = os.path.join(INPUT_SEGY_DIR, 'yingxi_crop.segy')
    # if not os.path.exists(segy_to_import_path):
    #     print(f"错误：要导入的SEGY文件不存在于 '{segy_to_import_path}'")
    # else:
    #     converter.import_segy_to_npy(
    #         segy_path=segy_to_import_path,
    #         out_dir=INPUT_NPY_DIR, # 将导入结果放入 input_npy 目录
    #         num_inline=num_inline,
    #         num_xline=num_xline
    #     )


    # --- 场景2: 导出单个文件 ---
    # print("\n==================== 场景2: 导出单个文件 ====================")
    # npy_to_export_path = r"C:\Work\sunjie\Python\cavity_modeling\data\output_npy\separation.npy"
    # if not os.path.exists(ref_segy_for_export_path):
    #     print(f"错误：参考SEGY文件不存在于 '{ref_segy_for_export_path}'")
    # elif not os.path.exists(npy_to_export_path):
    #     print(f"错误：要导出的NPY文件不存在于 '{npy_to_export_path}'")
    # else:
    #     try:
    #         converter.export_npy_to_segy(
    #             npy_path=npy_to_export_path,
    #             ref_segy_path=ref_segy_for_export_path,
    #             out_dir=OUTPUT_SEGY_DIR
    #         )
    #     except Exception as e:
    #         print(f"导出失败: {e}")


    # --- 场景3: 批量导入 ---
    # print("\n==================== 场景3: 批量导入 ====================")
    # converter.import_all(
    #     in_dir=INPUT_SEGY_DIR,
    #     out_dir=INPUT_NPY_DIR,
    #     num_inline=num_inline,
    #     num_xline=num_xline
    # )


    # --- 场景4: 批量导出 ---
    print("\n==================== 场景4: 批量导出 ====================")
    converter.export_all(
        in_dir=BATCH_DIR,
        ref_segy_path=ref_segy_for_export_path,
        out_dir=OUTPUT_SEGY_DIR
    )