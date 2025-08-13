import napari
import numpy as np
import tifffile
import os

if __name__ == '__main__':
    npz = np.load(r'C:\Work\sunjie\Python\cavity_modeling\data\input_npy\yingxi_crop.npz', allow_pickle=True)
    data = npz['data']

    # 尝试加载已存在的标签文件
    label_path_tif = r"C:\Work\sunjie\Python\cavity_modeling\data\train\Labels.tif"
    label_path_npy = "label.npy"

    if os.path.exists(label_path_tif):
        labels = tifffile.imread(label_path_tif).astype(np.uint8)
        print("✅ 已加载现有的 label.tif")
    elif os.path.exists(label_path_npy):
        labels = np.load(label_path_npy).astype(np.uint8)
        print("✅ 已加载现有的 label.npy")
    else:
        labels = np.zeros_like(data, dtype=np.uint8)
        print("⚠️ 未找到标签文件，使用全零标签")

    # 打开 napari 界面
    viewer = napari.Viewer()
    viewer.add_image(data, name="Seismic", contrast_limits=[data.min(), data.max()])
    label_layer = viewer.add_labels(labels, name="Label")

    napari.run()

    # 标注完成后保存
    np.save(label_path_npy, label_layer.data)
    tifffile.imwrite(label_path_tif, label_layer.data.astype(np.uint8))
    print("💾 标签已保存为 label.npy 和 label.tif")
