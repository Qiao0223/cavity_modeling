import numpy as np
import matplotlib.pyplot as plt


class SeismicPlotter:
    """
    SeismicPlotter 支持手动或自动按数据分布裁剪显示范围。
    config:
      - clip: 'none' | 'manual' | 'percentile'
      - vmin, vmax: float or None (manual 模式必填)
      - lower_percentile, upper_percentile: int (percentile 模式)
      - cmap: str
    """
    DIM_NAME_MAP = {
        0: 'inline',
        1: 'xline',
        2: 'time'   # 或 'depth' 由 meta['domain'] 决定，这里默认 time
    }
    NAME_DIM_MAP = {
        'inline': 0,
        'xline': 1,
        'time': 2,
        'depth': 2
    }

    def __init__(self, config: dict = None):
        self.config = {
            'clip': 'none',
            'vmin': None,
            'vmax': None,
            'lower_percentile': 1,
            'upper_percentile': 99,
            'cmap': 'seismic',
        }
        if config:
            self.config.update(config)

    # ---------- Public API 原有功能 ----------
    def plot_from_npz(self,
                      npz_path: str,
                      slice_type: str,
                      slice_coord: int,
                      config: dict = None) -> None:
        arch = np.load(npz_path, allow_pickle=True)
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
        # extent: [xmin, xmax, ymin, ymax]；imshow 默认 y 轴向下，这里反转 axis_v
        extent = [axis_h[0], axis_h[-1], axis_v[-1], axis_v[0]]
        plt.figure(figsize=(8, 5))
        plt.imshow(data.T, aspect='auto', cmap=cfg['cmap'], vmin=vmin, vmax=vmax, extent=extent)
        plt.title(title)
        plt.xlabel('Trace / X')
        plt.ylabel('Z / Time')
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

        if slice_type == 'inline':
            start = xl0
            xlabel = 'Xline'
            file_coord = il0 + slice_coord
            horiz_len = nx
        elif slice_type == 'xline':
            start = il0
            xlabel = 'Inline'
            file_coord = xl0 + slice_coord
            horiz_len = ni
        else:
            raise ValueError("slice_type 必须是 'inline' 或 'xline'")

        axis_h = start + np.arange(horiz_len)
        axis_v = z0 + np.arange(arr2d.shape[1]) * dt

        title = f"{slice_type.capitalize()}={file_coord} ({domain})"
        self._plot_custom(arr2d, axis_h, axis_v, title, xlabel, domain, config)

    def quick_show2d(self,
                     arr2d: np.ndarray,
                     title: str = 'Quick View 2D',
                     config: dict = None) -> None:
        """
        简单二维矩阵展示（不使用 meta），横轴为第一维索引，纵轴为第二维索引。
        """
        axis_h = np.arange(arr2d.shape[0])
        axis_v = np.arange(arr2d.shape[1])
        self.plot_from_2d(arr2d, axis_h, axis_v, title, config)

    def quick_show3d(self,
                     arr3d: np.ndarray,
                     dim,
                     index: int,
                     meta: dict = None,
                     title: str = None,
                     config: dict = None) -> None:
        """
        对 3D 数组抽取一个 2D 切片并显示。
        dim: int 或 字符串 ('inline'|'xline'|'time'|'depth')
        index: 该维度的切片索引（可为负）
        meta: 可选，用于真实坐标轴与标题
        title: 可选自定义标题
        """
        dim_idx = self._normalize_dim(dim)
        arr2d, axes_info = self._extract_slice_with_axes(arr3d, dim_idx, index, meta)
        if title is None:
            if meta:
                # 若有物理坐标，title 用实际文件坐标
                title = axes_info['title']
            else:
                dim_name = self.DIM_NAME_MAP.get(dim_idx, f"dim{dim_idx}")
                title = f"Slice {dim_name}[{axes_info['slice_index']}]"

        # 使用带 meta 的绘制函数（不带 meta 时 axes_info 已计算好）
        axis_h = axes_info['axis_h']
        axis_v = axes_info['axis_v']
        xlabel = axes_info['xlabel']
        domain_label = axes_info['domain']

        self._plot_custom(arr2d, axis_h, axis_v, title, xlabel, domain_label, config)

    # ---------- Internal Helpers ----------
    def _plot_custom(self,
                     arr2d: np.ndarray,
                     axis_h: np.ndarray,
                     axis_v: np.ndarray,
                     title: str,
                     xlabel: str,
                     domain_label: str,
                     config: dict = None) -> None:
        cfg = self._merge_config(config)
        vmin, vmax = self._prepare_clipping(arr2d, cfg)
        data = np.clip(arr2d, vmin, vmax) if vmin is not None and vmax is not None else arr2d
        extent = [axis_h[0], axis_h[-1], axis_v[-1], axis_v[0]]
        plt.figure(figsize=(8, 5))
        plt.imshow(data.T, aspect='auto', cmap=cfg['cmap'], vmin=vmin, vmax=vmax, extent=extent)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(domain_label.capitalize())
        plt.tight_layout()
        plt.colorbar()
        plt.show()

    def _normalize_dim(self, dim) -> int:
        if isinstance(dim, int):
            if dim not in (0, 1, 2):
                raise ValueError("dim 只能是 0,1,2")
            return dim
        if isinstance(dim, str):
            d = dim.lower()
            if d not in self.NAME_DIM_MAP:
                raise ValueError("dim 字符串必须是 'inline' | 'xline' | 'time' | 'depth'")
            return self.NAME_DIM_MAP[d]
        raise TypeError("dim 必须是 int 或 str")

    def _extract_slice_with_axes(self,
                                 arr3d: np.ndarray,
                                 dim: int,
                                 index: int,
                                 meta: dict | None):
        if arr3d.ndim != 3:
            raise ValueError("arr3d 必须是三维数组")
        n0, n1, n2 = arr3d.shape
        dims = [n0, n1, n2]
        # 处理负索引
        if not (-dims[dim] <= index < dims[dim]):
            raise IndexError(f"index {index} 超出维度 {dim} 范围 0..{dims[dim]-1}")
        index_norm = index % dims[dim]

        # 取切片
        if dim == 0:
            arr2d = arr3d[index_norm, :, :]   # shape (n1, n2)
            major1_len, major2_len = n1, n2
            slice_type = 'inline'
        elif dim == 1:
            arr2d = arr3d[:, index_norm, :]   # shape (n0, n2)
            major1_len, major2_len = n0, n2
            slice_type = 'xline'
        else:  # dim == 2 纵向切 时间/深度切片 -> 水平平面
            arr2d = arr3d[:, :, index_norm]   # shape (n0, n1)
            major1_len, major2_len = n0, n1
            slice_type = 'time'  # 或 depth

        if meta:
            ni = int(meta['num_inline'])
            nx = int(meta['num_xline'])
            il0 = int(meta['inline_start'])
            xl0 = int(meta['xline_start'])
            z0 = float(meta['z_start'])
            dt = float(meta['dt'])
            domain = meta.get('domain', 'time')

            if dim == 0:  # inline 切片：平面是 xline vs time
                file_coord = il0 + index_norm
                axis_h = xl0 + np.arange(nx)
                axis_v = z0 + np.arange(n2) * dt
                xlabel = 'Xline'
                domain_label = domain
                title = f"Inline={file_coord} ({domain})"
            elif dim == 1:  # xline 切片
                file_coord = xl0 + index_norm
                axis_h = il0 + np.arange(ni)
                axis_v = z0 + np.arange(n2) * dt
                xlabel = 'Inline'
                domain_label = domain
                title = f"Xline={file_coord} ({domain})"
            else:  # 时间/深度 层切片：平面是 inline vs xline
                axis_h = il0 + np.arange(ni)
                axis_v = xl0 + np.arange(nx)
                xlabel = 'Inline'
                domain_label = 'Xline'
                t_val = z0 + index_norm * dt
                title = f"{domain.capitalize()}={t_val:.3f}"
        else:
            # 无 meta 时全部用简单索引
            if dim == 0:
                axis_h = np.arange(major1_len)  # xline
                axis_v = np.arange(major2_len)  # time
                xlabel = 'Xline'
                domain_label = 'Time/Depth'
                title = f"Inline[{index_norm}]"
            elif dim == 1:
                axis_h = np.arange(major1_len)  # inline
                axis_v = np.arange(major2_len)
                xlabel = 'Inline'
                domain_label = 'Time/Depth'
                title = f"Xline[{index_norm}]"
            else:
                axis_h = np.arange(major1_len)  # inline
                axis_v = np.arange(major2_len)  # xline
                xlabel = 'Inline'
                domain_label = 'Xline'
                title = f"Level[{index_norm}]"

        return arr2d, {
            'axis_h': axis_h,
            'axis_v': axis_v,
            'xlabel': xlabel,
            'domain': domain_label,
            'title': title,
            'slice_index': index_norm,
            'slice_type': slice_type
        }

    def _merge_config(self, config: dict) -> dict:
        cfg = self.config.copy()
        if config:
            cfg.update(config)
        return cfg

    def _prepare_clipping(self, data: np.ndarray, cfg: dict) -> tuple:
        clip = cfg.get('clip', 'none')
        vmin = vmax = None

        if clip == 'manual':
            vmin = cfg.get('vmin')
            vmax = cfg.get('vmax')
            if vmin is None or vmax is None:
                raise ValueError("manual 模式下必须提供 'vmin' 和 'vmax'")
            if vmin > vmax:
                raise ValueError("vmin 必须 <= vmax")
        elif clip == 'percentile':
            lower = cfg.get('lower_percentile', 1)
            upper = cfg.get('upper_percentile', 99)
            if not (0 <= lower < upper <= 100):
                raise ValueError("percentile 范围需满足 0 <= lower < upper <= 100")
            vmin = np.percentile(data, lower)
            vmax = np.percentile(data, upper)

        return vmin, vmax

    def _plot_slice(self,
                    arr3d: np.ndarray,
                    meta: dict,
                    slice_type: str,
                    slice_coord: int,
                    config: dict = None) -> None:
        if slice_type == 'inline':
            arr2d = arr3d[slice_coord, :, :]
        elif slice_type == 'xline':
            arr2d = arr3d[:, slice_coord, :]
        else:
            raise ValueError("slice_type 必须是 'inline' 或 'xline'")
        self.plot_2d_with_meta(arr2d, meta, slice_type, slice_coord, config)
