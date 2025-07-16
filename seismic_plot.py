import numpy as np
import matplotlib.pyplot as plt


class SeismicPlotter:
    """
    SeismicPlotter 支持手动或自动按数据分布裁剪显示范围。
    self.config 字段:
      - clip: 'none' | 'manual' | 'percentile'
      - vmin, vmax: float or None (manual 模式必填)
      - lower_percentile, upper_percentile: int (percentile 模式)
      - cmap: str (颜色映射)
    """
    def __init__(self, config: dict = None):
        self.config = {
            'clip': 'none',           # 'none', 'manual', 'percentile'
            'vmin': None,
            'vmax': None,
            'lower_percentile': 1,
            'upper_percentile': 99,
            'cmap': 'seismic',
        }
        if config:
            self.config.update(config)

    def plot_from_npz(self,
                      npz_path: str,
                      slice_type: str,
                      slice_coord: int,
                      config: dict = None) -> None:
        arch = np.load(npz_path)
        arr3d = arch['data']
        meta = {
            'num_inline':   int(arch['num_inline']),
            'num_xline':    int(arch['num_xline']),
            'inline_start': int(arch['inline_start']),
            'xline_start':  int(arch['xline_start']),
            'z_start':      float(arch['z_start']),
            'dt':           float(arch['dt']),
            'domain':       arch.get('domain', 'time')
        }
        self._plot_slice(arr3d, meta, slice_type, slice_coord, config)

    def plot_from_array(self,
                        arr3d: np.ndarray,
                        meta: dict,
                        slice_type: str,
                        slice_coord: int,
                        config: dict = None) -> None:
        self._plot_slice(arr3d, meta, slice_type, slice_coord, config)

    def plot_from_2d(self,
                     arr2d: np.ndarray,
                     axis_h: np.ndarray,
                     axis_v: np.ndarray,
                     title: str,
                     config: dict = None) -> None:
        cfg = self._merge_config(config)
        vmin, vmax = self._prepare_clipping(arr2d, cfg)
        data = np.clip(arr2d, vmin, vmax) if vmin is not None and vmax is not None else arr2d
        extent = [axis_h[0], axis_h[-1], axis_v[-1], axis_v[0]]
        plt.figure(figsize=(8, 5))
        plt.imshow(data.T, aspect='auto', cmap=cfg['cmap'], vmin=vmin, vmax=vmax, extent=extent)
        plt.title(title)
        plt.xlabel('Trace Index')
        plt.ylabel('Z')
        plt.tight_layout()
        plt.colorbar()
        plt.show()

    def plot_2d_with_meta(self,
                          arr2d: np.ndarray,
                          meta: dict,
                          slice_type: str,
                          slice_coord: int,
                          config: dict = None) -> None:
        ni = int(meta['num_inline'])
        nx = int(meta['num_xline'])
        il0 = int(meta['inline_start'])
        xl0 = int(meta['xline_start'])
        z0 = float(meta['z_start'])
        dt = float(meta['dt'])
        domain = meta.get('domain', 'time')

        # 计算水平坐标起点和标签
        if slice_type == 'inline':
            start = xl0
            xlabel = 'Xline'
            file_coord = il0 + slice_coord
        elif slice_type == 'xline':
            start = il0
            xlabel = 'Inline'
            file_coord = xl0 + slice_coord
        else:
            raise ValueError("slice_type 必须是 'inline' 或 'xline'")

        count = arr2d.shape[0]
        axis_h = start + np.arange(count)
        axis_v = z0 + np.arange(arr2d.shape[1]) * dt

        title = f"{slice_type.capitalize()}={file_coord} ({domain})"
        self.plot_from_2d(arr2d, axis_h, axis_v, title, config)

    def quick_show(self,
                   arr2d: np.ndarray,
                   title: str = 'Quick View',
                   config: dict = None) -> None:
        axis_h = np.arange(arr2d.shape[0])
        axis_v = np.arange(arr2d.shape[1])
        self.plot_from_2d(arr2d, axis_h, axis_v, title, config)

    def _merge_config(self, config: dict) -> dict:
        cfg = self.config.copy()
        if config:
            cfg.update(config)
        return cfg

    def _prepare_clipping(self, data: np.ndarray, cfg: dict) -> tuple:
        clip = cfg.get('clip', 'none')
        vmin = None
        vmax = None

        if clip == 'manual':
            vmin = cfg.get('vmin')
            vmax = cfg.get('vmax')
            if vmin is None or vmax is None:
                raise ValueError("manual 模式下必须提供 'vmin' 和 'vmax'")
            if vmin > vmax:
                raise ValueError("vmin 必须小于等于 vmax")
        elif clip == 'percentile':
            lower = cfg.get('lower_percentile', 1)
            upper = cfg.get('upper_percentile', 99)
            vmin = np.percentile(data, lower)
            vmax = np.percentile(data, upper)

        return vmin, vmax

    def _plot_slice(self,
                    arr3d: np.ndarray,
                    meta: dict,
                    slice_type: str,
                    slice_coord: int,
                    config: dict = None) -> None:
        # 使用 slice_coord 作为数组索引
        if slice_type == 'inline':
            arr2d = arr3d[slice_coord, :, :]
        elif slice_type == 'xline':
            arr2d = arr3d[:, slice_coord, :]
        else:
            raise ValueError("slice_type 必须是 'inline' 或 'xline'")

        self.plot_2d_with_meta(arr2d, meta, slice_type, slice_coord, config)
