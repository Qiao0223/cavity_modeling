import numpy as np
import pywt
from scipy.signal import fftconvolve
from tqdm import tqdm
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple, Optional
import concurrent.futures
import itertools

@dataclass
class BandResult:
    """
    封装单个频段重构结果。
    frequency: 频段中心频率
    scale: 对应的 CWT 尺度
    freq_range: 频段上下限 (f_min, f_max)
    """
    data: np.ndarray
    frequency: float
    scale: Optional[float] = None
    freq_range: Optional[Tuple[float, float]] = None


def _reconstruct_section(
        section: np.ndarray,
        dt: float,
        params: Dict[int, Tuple[float, float, float, float, np.ndarray, str]]
) -> Dict[int, np.ndarray]:
    """
    对二维截面进行小波重构，params: {idx: (scale, freq, f_min, f_max, wave_func, wavelet_name)}
    返回: {idx: reconstructed_section}
    """
    n_tr, n_samples = section.shape
    sec_results = {idx: np.zeros_like(section) for idx in params}
    for j in range(n_tr):
        trace = section[j, :]
        for idx, (scale, _, _, _, wave_func, wavelet_name) in params.items():
            coeffs, _ = pywt.cwt(trace, [scale], wavelet_name, sampling_period=dt)
            sec_results[idx][j, :] = np.real(
                fftconvolve(coeffs[0], wave_func, mode='same')
            )
    return sec_results

class SeismicWaveletProcessor:
    """
    输入自定义 scales 列表，自动计算对应中心频率及频段范围。
    调用 reconstruct_volume/slice 可直接得到每个频段的重构结果。
    """
    def __init__(
        self,
        dt: float,
        scales: Sequence[float],
        wavelet: str = 'morl',
    ):
        """
        :param dt: 采样间隔
        :param scales: CWT 尺度列表
        :param wavelet: 母小波名称
        """
        self.dt = dt
        self.wavelet = wavelet
        # 强制提供 scales
        self.scales = np.asarray(scales, dtype=float)
        # 计算对应中心频率
        fs = 1.0 / dt
        cf = pywt.central_frequency(wavelet)
        self.frequencies = cf * fs / self.scales
        # 根据频率分布自动生成频段上下限
        freqs = self.frequencies
        n = len(freqs)
        # 对频段排序，获取邻频差用于边界
        sorted_idx = np.argsort(freqs)
        sorted_f = freqs[sorted_idx]
        # 计算边界
        bounds = np.zeros((n+1,))
        bounds[1:-1] = (sorted_f[:-1] + sorted_f[1:]) / 2
        # 边缘延展相对间隔
        bounds[0] = sorted_f[0] - (bounds[1] - sorted_f[0])
        bounds[-1] = sorted_f[-1] + (sorted_f[-1] - bounds[-2])
        # 每个 scale 的频段范围
        self.freq_ranges = {}
        for i, idx_orig in enumerate(sorted_idx):
            fmin = bounds[i]
            fmax = bounds[i+1]
            self.freq_ranges[idx_orig] = (fmin, fmax)
        # levels 直接作为索引
        self.levels = list(range(len(scales)))

    def reconstruct_volume(
        self,
        volume: np.ndarray
    ) -> Dict[int, BandResult]:
        if volume.ndim != 3:
            raise ValueError("volume must be a 3D array")
        ni, nx, nt = volume.shape
        # 缓存 wave_func 和参数
        params: Dict[int, Tuple[float, float, float, float, np.ndarray, str]] = {}
        wavelet_obj = pywt.ContinuousWavelet(self.wavelet)
        for idx in self.levels:
            scale = self.scales[idx]
            freq = self.frequencies[idx]
            fmin, fmax = self.freq_ranges[idx]
            kernel_size = min(int(10 * scale), nt)
            wave_func, _ = wavelet_obj.wavefun(length=kernel_size)
            params[idx] = (scale, freq, fmin, fmax, wave_func, self.wavelet)
        # 初始化结果
        results = {
            idx: BandResult(
                data=np.zeros_like(volume),
                frequency=params[idx][1],
                scale=params[idx][0],
                freq_range=(params[idx][2], params[idx][3])
            )
            for idx in self.levels
        }
        # 并行处理
        with concurrent.futures.ProcessPoolExecutor() as executor:
            sections = (volume[i, :, :] for i in range(ni))
            dt_rep = itertools.repeat(self.dt)
            params_rep = itertools.repeat(params)
            for i, sec_res in enumerate(tqdm(
                    executor.map(_reconstruct_section, sections, dt_rep, params_rep),
                    total=ni, desc="Reconstruct volume", unit="slice")):
                for idx in self.levels:
                    results[idx].data[i, :, :] = sec_res[idx]
        return results

    def reconstruct_slice(
        self,
        section: np.ndarray
    ) -> Dict[int, BandResult]:
        if section.ndim != 2:
            raise ValueError("section must be a 2D array")
        _, nt = section.shape
        # 缓存 wave_func 和参数
        params: Dict[int, Tuple[float, float, float, float, np.ndarray, str]] = {}
        wavelet_obj = pywt.ContinuousWavelet(self.wavelet)
        for idx in self.levels:
            scale = self.scales[idx]
            freq = self.frequencies[idx]
            fmin, fmax = self.freq_ranges[idx]
            kernel_size = min(int(10 * scale), nt)
            wave_func, _ = wavelet_obj.wavefun(length=kernel_size)
            params[idx] = (scale, freq, fmin, fmax, wave_func, self.wavelet)
        sec_res = _reconstruct_section(section, self.dt, params)
        return {
            idx: BandResult(
                data=sec_res[idx],
                frequency=params[idx][1],
                scale=params[idx][0],
                freq_range=(params[idx][2], params[idx][3])
            )
            for idx in self.levels
        }

# 使用示例：
# processor = SeismicWaveletProcessor(dt=0.004, scales=[1.0, 2.0, 4.0])
# results = processor.reconstruct_slice(seis2d)
# for band in results.values():
#     print(band.scale, band.frequency, band.freq_range)
