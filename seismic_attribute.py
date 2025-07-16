# rms_calc.py

import numpy as np

def compute_rms(data: np.ndarray, window: int) -> np.ndarray:
    """
    Compute fixed-window RMS amplitude along the last axis (time) of the input array.

    Parameters:
        data (np.ndarray): Input array of shape (..., T), dtype float32
        window (int): Size of the sliding time window

    Returns:
        np.ndarray: Output array of same shape with RMS values
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input must be a NumPy ndarray.")

    if data.dtype != np.float32:
        raise ValueError("Input array must be of dtype float32.")

    if window < 1:
        raise ValueError("Window size must be at least 1.")

    T = data.shape[-1]
    pad_width = [(0, 0)] * data.ndim
    pad_width[-1] = (window - 1, 0)

    squared = np.square(data)
    padded = np.pad(squared, pad_width, mode='constant', constant_values=0)

    cumsum = np.cumsum(padded, axis=-1)
    sum_window = cumsum[..., window:] - cumsum[..., :-window]

    rms = np.sqrt(sum_window / window)

    return rms.astype(np.float32)

# Example usage:
# data = np.load("seismic.npy")
# rms = compute_rms_along_time(data, window=25)
# np.save("rms_output.npy", rms)