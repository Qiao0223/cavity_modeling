import napari
import numpy as np


if __name__ == '__main__':
    npz = np.load(r'C:\Work\sunjie\Python\cavity_modeling\input_npy\yingxi_crop.npz', allow_pickle=True)
    data = npz['data']

    # 打开 napari 界面
    viewer = napari.view_image(data, name="Seismic", contrast_limits=[data.min(), data.max()])

    # 添加一个标注层（全零数组）
    labels = np.zeros_like(data, dtype=np.uint8)
    label_layer = viewer.add_labels(labels, name="Label")

    napari.run()

    # 标注完成后保存
    np.save("label.npy", label_layer.data)
