# containers.py
from dependency_injector import containers, providers
from plot.seismic_plot import SeismicPlotter

class PlotterContainer(containers.DeclarativeContainer):
    seis = providers.Singleton(SeismicPlotter, config={
        'clip': 'none', 'vmin': None, 'vmax': None,
        'lower_percentile': 1, 'upper_percentile': 99,
        'cmap': 'seismic'
    })

    band = providers.Singleton(SeismicPlotter, config={
        'clip': 'manual', 'vmin': -4, 'vmax': 4,
        'lower_percentile': 1, 'upper_percentile': 99,
        'cmap': 'seismic'
    })

    rms = providers.Singleton(SeismicPlotter, config={
        'clip': 'manual', 'vmin': 0.3, 'vmax': 1,
        'lower_percentile': 1, 'upper_percentile': 99,
        'cmap': 'viridis'
    })
