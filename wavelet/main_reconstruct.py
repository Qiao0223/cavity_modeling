import numpy as np

from plot.containers_plot import PlotterContainer
from wavelet.seismic_wavelet_processor import SeismicWaveletProcessor
from attribute import rms

if __name__ == '__main__':

    npz = np.load(r'C:\Work\sunjie\Python\cavity_modeling\data\input_npy\fuyuan3_crop.npz', allow_pickle=True)
    seis3d = npz['data']
    meta = npz['meta'].item()

    slice_coord = 500
    seis2d =seis3d[slice_coord]

    plotter_seis = PlotterContainer.seis()
    plotter_band = PlotterContainer.band()

    # yingxi_crop dt = 0.008, level_count=10, level = 1, wavelet_param = 6
    num = 8
    plotter_seis.plot_2d_with_meta(seis2d,  meta, "inline", slice_coord)

    # 只重构并画 2D slice
    processor = SeismicWaveletProcessor(
        dt=0.008,
        scales=np.arange(start=0.5, stop=10, step=0.2),
    )

    results2d = processor.reconstruct_slice(seis2d)
    for lvl, band in results2d.items():
        plotter_band.plot_2d_with_meta(band.data, meta, "inline", slice_coord)

    # processor = SeismicWaveletProcessor(dt=0.008, level_count=10, levels=list([2]))
    # results = processor.reconstruct_volume(seis3d)
    # frequency = results.get(2).data
    # np.save("frequency.npy", frequency)
    # for result in results.values():
    #     plotter_band.plot_2d_with_meta(result.data, meta, "inline", slice_coord)
    #
    #
    # processor = SeismicWaveletProcessor(dt=0.008, level_count=num, levels=list(range(1,num+1)))
    # results_2 = processor.reconstruct_slice(seis2d)
    # for band in results_2.values():
    #     plotter_band.plot_2d_with_meta(band.data, meta, "inline", slice_coord)


