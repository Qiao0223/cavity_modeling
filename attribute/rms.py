import numpy as np
from scipy.ndimage import generic_filter

def rms_filter(data: np.ndarray, window: int = 11) -> np.ndarray:
    """
    Compute RMS amplitude for 2D or 3D seismic data using a sliding window.

    Parameters
    ----------
    data : np.ndarray
        Input seismic data. 2D array (traces, time) or 3D array (iline, xline, time).
    window : int, optional
        Number of samples for the sliding window along the time axis (last dimension), by default 11.

    Returns
    -------
    np.ndarray
        RMS attribute volume of the same shape as input data.
    """
    # Define RMS function for a sliding window
    def _rms(vals):
        arr = np.asarray(vals)
        return np.sqrt(np.mean(arr ** 2))

    # Determine footprint size based on input dimensions
    if data.ndim == 2:
        size = (1, window)
    elif data.ndim == 3:
        size = (1, 1, window)
    else:
        raise ValueError("Input data must be 2D or 3D array")

    # Apply generic_filter with reflective boundary handling
    return generic_filter(data, _rms, size=size, mode='reflect')
