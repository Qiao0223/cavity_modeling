import json
import os
import glob
import segyio
import numpy as np
from datetime import datetime, timezone


class MySegyio:
    """
    SEG-Y ↔ NumPy(.npz) 转换工具，支持单文件与批量操作。
    """

    def __init__(self,
                 num_inline: int | None = None,
                 num_xline: int | None = None,
                 inline_start: int | None = None,
                 xline_start: int | None = None,
                 dt: float | None = None,
                 domain: str | None = None,
                 z_start: float | None = 0.0):
        self.num_inline = num_inline
        self.num_xline = num_xline
        self.inline_start = inline_start
        self.xline_start = xline_start
        self.dt = dt
        self.z_start = z_start
        self.domain = domain
        self._preloaded_data = None

    def import_file(self, segy_path: str, out_dir: str) -> None:
        """
        将单个 SEG-Y 文件转换为 .npz，内嵌数据和元信息。
        """
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(segy_path))[0]
        out_npz = os.path.join(out_dir, base + '.npz')

        with segyio.open(segy_path, 'r', strict=False) as src:
            traces = segyio.collect(src.trace)
            arr2d = np.array(traces)

        if arr2d.shape[0] != self.num_inline * self.num_xline:
            raise ValueError(f"Trace count mismatch: {arr2d.shape[0]}")

        nt = arr2d.shape[1]
        arr3d = arr2d.reshape((self.num_inline, self.num_xline, nt))

        meta = {
            "num_inline": self.num_inline,
            "num_xline": self.num_xline,
            "inline_start": self.inline_start,
            "xline_start": self.xline_start,
            "z_start": self.z_start,
            "dt": self.dt,
            "domain": self.domain,
            "created_time": datetime.now(timezone.utc).isoformat()
        }

        np.savez_compressed(out_npz, data=arr3d, meta=meta)
        print(f"Imported and saved NPZ: {out_npz}")

    def import_all(self, directory: str, out_dir: str = None) -> None:
        """
        批量转换目录中的所有 SEG-Y 文件。
        """
        out_dir = out_dir or os.path.join(directory, 'numpy_arrays')
        if os.path.isdir(directory):
            patterns = ['*.sgy', '*.segy']
            files = []
            for p in patterns:
                files.extend(glob.glob(os.path.join(directory, p)))
        elif os.path.isfile(directory):
            files = [directory]
        else:
            raise FileNotFoundError(f"Invalid path: {directory}")
        for segy_file in files:
            self.import_file(segy_file, out_dir)

    def export_file(self, numpy_path: str, ref_segy: str, out_dir: str) -> None:
        """
        自动判断 .npz 或 .npy 文件，并导出为 SEG-Y。
        """
        os.makedirs(out_dir, exist_ok=True)
        base, ext = os.path.splitext(os.path.basename(numpy_path))
        out_sgy = os.path.join(out_dir, base + '.segy')

        if ext == '.npz':
            arch = np.load(numpy_path)
            arr3d = arch['data']
            num_il = int(arch['num_inline'])
            num_xl = int(arch['num_xline'])
        elif ext == '.npy':
            arr3d = np.load(numpy_path, allow_pickle=True)
            num_il, num_xl = arr3d.shape[:2]
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

        if arr3d.shape[:2] != (num_il, num_xl):
            raise ValueError(f"Shape mismatch: {arr3d.shape}")

        arr2d = arr3d.reshape((num_il * num_xl, arr3d.shape[2]))
        self._copy_segy(srcpath=ref_segy, dstpath=out_sgy, new_trace=arr2d)
        print(f"Exported SEG-Y: {out_sgy}")

    def export_all(self, numpy_dir: str, ref_segy: str, out_dir: str) -> None:
        """
        批量导出目录下所有 .npz 或 .npy 文件为 SEG-Y。
        """
        if os.path.isdir(numpy_dir):
            files = glob.glob(os.path.join(numpy_dir, '*.npz')) + \
                    glob.glob(os.path.join(numpy_dir, '*.npy'))
        elif os.path.isfile(numpy_dir) and (numpy_dir.endswith('.npz') or numpy_dir.endswith('.npy')):
            files = [numpy_dir]
        else:
            raise FileNotFoundError(f"Invalid numpy input: {numpy_dir}")
        for np_file in files:
            self.export_file(np_file, ref_segy, out_dir)

    def _copy_segy(self, srcpath: str, dstpath: str, new_trace: np.ndarray) -> None:
        """
        复制 SEG-Y 头部并替换 trace 数据。
        """
        with segyio.open(srcpath, 'r') as src:
            spec = segyio.spec()
            spec.sorting = src.sorting
            spec.format = src.format
            spec.samples = src.samples
            spec.ilines = src.ilines
            spec.xlines = src.xlines
            with segyio.create(dstpath, spec) as dst:
                dst.text[0] = src.text[0]
                dst.bin = src.bin
                dst.header = src.header
                dst.trace = new_trace

    @staticmethod
    def load_data_only(npz_path: str) -> np.ndarray:
        """
        只读取 .npz 中的 3D 数据，不关心 meta。
        优先使用 'data' 键。若无 'data'，尝试用形状和独立字段重建。
        """
        if not os.path.isfile(npz_path):
            raise FileNotFoundError(npz_path)

        arch = np.load(npz_path, allow_pickle=True)

        # 常规路径：写入时使用 data=arr3d
        if 'data' in arch:
            arr3d = arch['data']
            if arr3d.ndim != 3:
                raise ValueError(f"'data' 不是 3D: shape={arr3d.shape}")
            return arr3d

        # 兼容：存在分散的 num_inline / num_xline 字段，且可能有扁平化存法(不建议，但处理)
        keys = arch.files
        required = {'num_inline', 'num_xline'}
        if required.issubset(keys):
            num_inline = int(arch['num_inline'])
            num_xline = int(arch['num_xline'])

            # 尝试寻找某个可能的数组字段
            # 1) 如果有 raw_flat 或 traces 之类
            candidate_arrays = [k for k in keys if k not in required and k not in ('inline_start','xline_start','z_start','dt','domain','meta')]
            for k in candidate_arrays:
                arr = arch[k]
                # 如果已经是 3D 则直接用
                if isinstance(arr, np.ndarray):
                    if arr.ndim == 3 and arr.shape[0] == num_inline and arr.shape[1] == num_xline:
                        return arr
                    # 若是 2D 并能 reshape
                    if arr.ndim == 2 and arr.shape[0] == num_inline * num_xline:
                        arr3d = arr.reshape(num_inline, num_xline, arr.shape[1])
                        return arr3d
                    # 若是一维可能是展平
                    if arr.ndim == 1 and (arr.size % (num_inline * num_xline) == 0):
                        nt = arr.size // (num_inline * num_xline)
                        arr3d = arr.reshape(num_inline, num_xline, nt)
                        return arr3d

        raise ValueError(f"无法在 {npz_path} 中定位 3D 数据（缺少 'data' 或可识别字段）")

    @staticmethod
    def load_all_data_only(directory: str, pattern: str = "*.npz") -> dict:
        """
        批量读取目录中所有 npz 的 3D 数据（只读 data），返回 {basename: ndarray}
        出错文件跳过并打印警告。
        """
        if not os.path.isdir(directory):
            raise NotADirectoryError(directory)

        paths = glob.glob(os.path.join(directory, pattern))
        result = {}
        for p in paths:
            base = os.path.splitext(os.path.basename(p))[0]
            try:
                result[base] = MySegyio.load_data_only(p)
            except Exception as e:
                print(f"[WARN] 跳过 {p}: {e}")
        return result

# 示例调用：无需命令行，直接在代码里配置参数并执行
if __name__ == '__main__':
    # 读取 JSON 配置文件
    with open('config.json', 'r') as f:
        config = json.load(f)

    # 获取 yingxi_crop 的配置
    params = config['yingxi_crop']

    # 计算派生参数
    num_inline = params['inline_end'] - params['inline_start'] + 1
    num_xline = params['xline_end'] - params['xline_start'] + 1

    # 初始化 MySegyio
    my = MySegyio(
        num_inline, num_xline,
        params['inline_start'], params['xline_start'],
        params['dt'], params['domain'], params['z_start']
    )

    # 导入单文件
    # my.import_file(
    #     segy_path=r'../input_segy/yingxi_velocity_crop.segy',
    #     out_dir=r'../input_npy'
    # )

    # 批量导入
    # my.import_all(r'../data/input_segy', '../data/input_numpy')

    # 单文件导出
    # my.export_file(r'../output_npy/frequency.npy', r'../input_segy/yingxi_crop.segy', r'../output_segy')

    # 批量导出
    my.export_all(r'../data/output_npy', '../data/input_segy/yingxi_crop.segy', '../data/output_segy')
