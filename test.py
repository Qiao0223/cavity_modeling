import numpy as np
import seismic_attribute as SA

from seismic_wavelet_processor import SeismicWaveletProcessor
from seismic_plot import SeismicPlotter

if __name__ == '__main__':

    data3d = np.load("input_numpy/YingXi_crop.npy")
    data2d = data3d[470]

    processor = SeismicWaveletProcessor(dt=0.008, level_count=10, levels=[1,2,3,4,5,6,7,8,9,10])
    # results = processor.reconstruct_slice(data2d)

    # min 50, max 100, dt 0.008, level_count 10, level 1
    results = processor.extract_frequency_bands(data2d, 50, 100)
    plotterSeis = SeismicPlotter(config={
            'clip': 'none',           # 'none', 'manual', 'percentile'
            'vmin': None,
            'vmax': None,
            'lower_percentile': 1,
            'upper_percentile': 99,
            'cmap': 'seismic',
        })
    plotterSeis.quick_show(data2d)

    plotterBand = SeismicPlotter(config={
            'clip': 'manual',           # 'none', 'manual', 'percentile'
            'vmin': -3,
            'vmax': 3,
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
    plotterBand.quick_show(results[1].data)

        # plotterRms.quick_show(SA.compute_rms(band.data, 5))

    # results_3d = processor.reconstruct_volume(data3d)
    # a = results_3d[4].data
    # np.save("input_numpy/a.npy", a)
    # b = np.load("input_numpy/a.npy")

