import numpy as np

from seismic_wavelet_processor import SeismicWaveletProcessor
from seismic_plot import SeismicPlotter

if __name__ == '__main__':

    npz = np.load('input_npy/yingxi_crop.npz', allow_pickle=True)
    seis3d = npz['data']
    meta = npz['meta'].item()

    slice_coord = 758
    seis2d =seis3d[slice_coord]

    plotterSeis = SeismicPlotter(config={
            'clip': 'none',           # 'none', 'manual', 'percentile'
            'vmin': None,
            'vmax': None,
            'lower_percentile': 1,
            'upper_percentile': 99,
            'cmap': 'seismic',
        })

    plotterBand = SeismicPlotter(config={
            'clip': 'manual',           # 'none', 'manual', 'percentile'
            'vmin': -4,
            'vmax': 4,
            'lower_percentile': 1,
            'upper_percentile': 99,
            'cmap': 'seismic',
    })

    plotterRms = SeismicPlotter(config={
            'clip': 'manual',           # 'none', 'manual', 'percentile'
            'vmin': 0.2,
            'vmax': 1,
            'lower_percentile': 1,
            'upper_percentile': 99,
            'cmap': 'Oranges',
    })

    processor = SeismicWaveletProcessor(dt=0.008, level_count=10, levels=[1,2,3,4,5,6,7,8,9,10])
    # results = processor.reconstruct_slice(seis2d)

    # min 50, max 100, dt 0.008, level_count 10, level 1
    results = processor.extract_frequency_bands(seis2d, 50, 100)

    plotterSeis.plot_2d_with_meta(seis2d, meta, "inline", slice_coord)
    for band in results.values():
        plotterBand.plot_2d_with_meta(band.data, meta, "inline", slice_coord)


