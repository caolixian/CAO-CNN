import torch
import nibabel as nib
import numpy
import os

filepath = r'E:\Download\神经影像分析与疾病预测挑战赛初赛数据集\讯飞初赛\Train'

for path1, path2, path3 in os.walk(filepath):
    if path2 == []:
        for p in path3:
            data_path = os.path.join(path1, p)
            label = path1[len(filepath)+1:]

            img = nib.load(data_path)
            img_arr = img.get_fdata()
            print(label)
            print(img_arr.shape)
