import numpy as np
from myio.my_segyio import MySegyio
from plot import containers_plot

if __name__ == '__main__':
    plotter_seis = containers_plot.PlotterContainer.seis()
    mysegyio = MySegyio()

    data3d = np.load(r"C:\Work\sunjie\Python\cavity_modeling\output_npy\converted_data_time.npy")
    seis3d = mysegyio.load_data_only(r"C:\Work\sunjie\Python\cavity_modeling\input_npy\yingxi_crop.npz")
    plotter_seis.quick_show3d(data3d, 1, 400)
    plotter_seis.quick_show3d(seis3d, 1, 400)
