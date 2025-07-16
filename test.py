import numpy as np

from plot.containers_plot import PlotterContainer
from wavelet.seismic_wavelet_processor import SeismicWaveletProcessor
from attribute import rms

if __name__ == '__main__':

    npz = np.load('input_npy/yingxi_crop.npz', allow_pickle=True)
    seis3d = npz['data']
    meta = npz['meta'].item()

    slice_coord = 758
    seis2d =seis3d[slice_coord]

    plotter_seis = PlotterContainer.seis()
    plotter_band = PlotterContainer.band()
    plotter_rms = PlotterContainer.rms()

    processor = SeismicWaveletProcessor(dt=0.008, level_count=10, levels=[1,2])
    # results = processor.reconstruct_slice(seis2d)

    # min 50, max 100, dt 0.008, level_count 10, level 1
    results = processor.extract_frequency_bands(seis2d, 50, 100)

    plotter_seis.plot_2d_with_meta(seis2d, meta, "inline", slice_coord)
    for band in results.values():
        plotter_band.plot_2d_with_meta(band.data, meta, "inline", slice_coord)
        plotter_rms.plot_2d_with_meta(rms.rms_filter(band.data), meta, "inline", slice_coord)


