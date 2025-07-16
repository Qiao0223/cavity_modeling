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
                 num_inline: int,
                 num_xline: int,
                 inline_start: int,
                 xline_start: int,
                 dt: float,
                 domin: str,
                 z_start: float = 0.0):
        self.num_inline = num_inline
        self.num_xline = num_xline
        self.inline_start = inline_start
        self.xline_start = xline_start
        self.dt = dt
        self.z_start = z_start
        self.domin = domin

    def import_file(self, segy_path: str, out_dir: str) -> None:
        """
        将单个 SEG-Y 文件转换为 .npz，内嵌数据和元信息。
        """
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(segy_path))[0]
        out_npz = os.path.join(out_dir, base + '.npz')
        with segyio.open(segy_path, 'r', strict=False) as src:
            traces = segyio.collect(src.trace)
            arr2d  = np.array(traces)
        if arr2d.shape[0] != self.num_inline * self.num_xline:
            raise ValueError(f"Trace count mismatch: {arr2d.shape[0]}")
        nt = arr2d.shape[1]
        arr3d = arr2d.reshape((self.num_inline, self.num_xline, nt))
        np.savez_compressed(
            out_npz,
            data=arr3d,
            num_inline=self.num_inline,
            num_xline=self.num_xline,
            inline_start=self.inline_start,
            xline_start=self.xline_start,
            z_start=self.z_start,
            dt=self.dt,
            domain=self.domin,
            created_time = datetime.now(timezone.utc).isoformat()
        )
        print(f"Imported and saved NPZ: {out_npz}")

    def import_all(self, directory: str, out_dir: str = None) -> None:
        """
        批量转换目录中的所有 SEG-Y 文件。
        """
        out_dir = out_dir or os.path.join(directory, 'numpy_arrays')
        if os.path.isdir(directory):
            patterns = ['*.sgy','*.segy']
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
            spec.format  = src.format
            spec.samples = src.samples
            spec.ilines  = src.ilines
            spec.xlines  = src.xlines
            with segyio.create(dstpath, spec) as dst:
                dst.text[0] = src.text[0]
                dst.bin      = src.bin
                dst.header   = src.header
                dst.trace    = new_trace

# 示例调用：无需命令行，直接在代码里配置参数并执行
if __name__ == '__main__':
    inline_start = 1600
    inline_end   = 2440
    xline_start  = 1000
    xline_end    = 1780
    dt           = 5
    z_start   = 5712.50
    domain = "depth"

    num_inline = inline_end - inline_start + 1
    num_xline  = xline_end  - xline_start + 1

    my = MySegyio(
        num_inline, num_xline,
        inline_start, xline_start,
        dt, domain, z_start
    )

    # 导入单文件
    # my.import_file(
    #     segy_path = 'input_segy/pps_lc.segy',
    #     out_dir   = 'input_numpy'
    # )

    # 批量导入
    # my.import_all('E:/Bonan/Seismic/...', 'input_numpy')

    # 单文件导出
    my.export_file('input_numpy/results_3d.npy', 'output_segy/recon_layer_2.segy', 'out_segy')

    # 批量导出
    # my.export_all('input_numpy', 'input_segy/ref.sgy', 'out_segy')