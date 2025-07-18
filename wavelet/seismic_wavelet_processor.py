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
    """
    data: np.ndarray
    frequency: float
    scale: Optional[float] = None
    freq_range: Optional[Tuple[float, float]] = None


def _reconstruct_section(section: np.ndarray,
                          dt: float,
                          levels: Sequence[int],
                          params: Dict[int, Tuple[float, float, str]]) -> Dict[int, np.ndarray]:
    shape = section.shape  # (nx, nt)
    sec_results = {lvl: np.zeros(shape, dtype=section.dtype) for lvl in levels}
    wavelet_name = next(iter(params.values()))[2]
    wavelet_obj = pywt.ContinuousWavelet(wavelet_name)
    for j in range(shape[0]):
        trace = section[j, :]
        for lvl in levels:
            scale, _, _ = params[lvl]
            coeffs, _ = pywt.cwt(trace, [scale], wavelet_name, sampling_period=dt)
            # 逆小波重构
            length = coeffs.shape[-1]
            kernel_size = min(int(10 * scale), length)
            wave_func, _ = wavelet_obj.wavefun(length=kernel_size)
            sec_results[lvl][j, :] = np.real(fftconvolve(coeffs[0], wave_func, mode='same'))
    return sec_results

class SeismicWaveletProcessor:
    """
    小波重构与频段提取处理器，支持自定义母小波。
    构造时指定采样间隔(dt)、小波层级数量(level_count)、关心的 levels 及母小波名称。
    支持多核并行计算加速 3D 体数据的处理。
    """

    def __init__(self,
                 dt: float,
                 level_count: int,
                 levels: Sequence[int],
                 wavelet: str = 'morl'):
        self.dt = dt
        self.level_count = level_count
        self.levels = levels
        self.wavelet = wavelet
        self.scales, self.frequencies = self._compute_scales_and_frequencies()

    def _compute_scales_and_frequencies(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算连续小波变换的 scales 和对应中心频率。
        基于 pywt.central_frequency 动态适配母小波。
        """
        fs = 1.0 / self.dt
        cf = pywt.central_frequency(self.wavelet)
        s_min = 2 * cf
        scales = s_min * (2.0 ** np.arange(self.level_count))
        freqs = cf * fs / scales
        return scales, freqs

    def reconstruct_volume(
        self,
        volume: np.ndarray
    ) -> Dict[int, BandResult]:
        if volume.ndim != 3:
            raise ValueError("volume must be a 3D array")
        ni, nx, _ = volume.shape
        # 参数准备：scale, freq, wavelet
        params = {lvl: (self.scales[lvl-1], self.frequencies[lvl-1], self.wavelet)
                  for lvl in self.levels}
        results = {
            lvl: BandResult(
                data=np.zeros_like(volume),
                frequency=params[lvl][1],
                scale=params[lvl][0]
            )
            for lvl in self.levels
        }
        with concurrent.futures.ProcessPoolExecutor() as executor:
            dt_rep = itertools.repeat(self.dt)
            lvl_rep = itertools.repeat(self.levels)
            params_rep = itertools.repeat(params)
            sections = (volume[i, :, :] for i in range(ni))
            wave_rep = itertools.repeat(self.wavelet)
            map_iter = executor.map(_reconstruct_section,
                                    sections,
                                    dt_rep,
                                    lvl_rep,
                                    params_rep,
                                    wave_rep)
            for i, sec_res in enumerate(tqdm(map_iter,
                                             total=ni,
                                             desc="Reconstruct volume",
                                             unit="slice")):
                for lvl in self.levels:
                    results[lvl].data[i, :, :] = sec_res[lvl]
        return results

    def reconstruct_slice(
        self,
        section: np.ndarray
    ) -> Dict[int, BandResult]:
        if section.ndim != 2:
            raise ValueError("section must be a 2D array")
        n_tr, _ = section.shape
        params = {lvl: (self.scales[lvl-1], self.frequencies[lvl-1])
                  for lvl in self.levels}
        results = {
            lvl: BandResult(
                data=np.zeros_like(section),
                frequency=params[lvl][1],
                scale=params[lvl][0]
            )
            for lvl in self.levels
        }
        wavelet_obj = pywt.ContinuousWavelet(self.wavelet)
        for idx in tqdm(range(n_tr), desc="Reconstruct slice", unit="trace"):
            trace = section[idx, :]
            for lvl in self.levels:
                scale, _ = params[lvl]
                coeffs, _ = pywt.cwt(trace, [scale], self.wavelet, sampling_period=self.dt)
                # 直接调用内部逆变换
                length = coeffs.shape[-1]
                kernel_size = min(int(10 * scale), length)
                wave_func, _ = wavelet_obj.wavefun(length=kernel_size)
                results[lvl].data[idx, :] = np.real(fftconvolve(coeffs[0], wave_func, mode='same'))
        return results

    def extract_frequency_bands(
        self,
        data: np.ndarray,
        freq_min: float,
        freq_max: float
    ) -> Dict[int, BandResult]:
        fs = 1.0 / self.dt
        band_count = self.level_count
        edges = np.linspace(freq_min, freq_max, band_count + 1)
        for idx in self.levels:
            if idx < 1 or idx > band_count:
                raise ValueError(f"band index {idx} out of 1..{band_count}")
        band_params = {
            idx: (
                pywt.central_frequency(self.wavelet) * fs / ((edges[idx-1] + edges[idx]) / 2.0),
                (edges[idx-1] + edges[idx]) / 2.0,
                (edges[idx-1], edges[idx])
            )
            for idx in self.levels
        }
        shape = data.shape
        results = {
            idx: BandResult(
                data=np.zeros(shape, dtype=data.dtype),
                frequency=band_params[idx][1],
                scale=band_params[idx][0],
                freq_range=band_params[idx][2]
            )
            for idx in self.levels
        }
        wavelet_obj = pywt.ContinuousWavelet(self.wavelet)
        def _process(trace: np.ndarray):
            scales = [band_params[idx][0] for idx in self.levels]
            coeffs, _ = pywt.cwt(trace, scales, self.wavelet, sampling_period=self.dt)
            out = {}
            for k, lvl in enumerate(self.levels):
                length = coeffs[k].shape[-1]
                kernel_size = min(int(10 * scales[k]), length)
                wave_func, _ = wavelet_obj.wavefun(length=kernel_size)
                out[lvl] = np.real(fftconvolve(coeffs[k], wave_func, mode='same'))
            return out
        if data.ndim == 3:
            ni, nx, _ = data.shape
            for i in tqdm(range(ni), desc="Extract bands volume", unit="inline"):
                for j in range(nx):
                    out = _process(data[i, j, :])
                    for idx, arr in out.items():
                        results[idx].data[i, j, :] = arr
        elif data.ndim == 2:
            n_tr, _ = data.shape
            for i in tqdm(range(n_tr), desc="Extract bands slice", unit="trace"):
                out = _process(data[i, :])
                for idx, arr in out.items():
                    results[idx].data[i, :] = arr
        else:
            raise ValueError("data must be 2D or 3D array")
        return results
