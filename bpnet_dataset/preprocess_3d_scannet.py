#!/usr/bin/env python
"""
    File Name   :   s3g-preprocess_scannet
    date        :   16/4/2020
    Author      :   wenbo
    Email       :   huwenbodut@gmail.com
    Description :
                              _     _
                             ( |---/ )
                              ) . . (
________________________,--._(___Y___)_,--._______________________
                        `--'           `--'
"""

import glob
import multiprocessing as mp
import numpy as np
import plyfile
import torch

# Map relevant classes to {0,1,...,19}, and ignored classes to 255
remapper = np.ones(150) * (255)
for i, x in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]):
    remapper[x] = i


def f(fn):
    print("111")
    fn2 = fn[:-3] + 'labels.ply'
    a = plyfile.PlyData().read(fn)
    v = np.array([list(x) for x in a.elements[0]])
    coords = np.ascontiguousarray(v[:, :3])
    colors = np.ascontiguousarray(v[:, 3:6]) / 127.5 - 1
    a = plyfile.PlyData().read(fn2)
    w = remapper[np.array(a.elements[0]['label'])]
    torch.save((coords, colors, w), fn[:-4] + '.pth')
    print(fn, fn2)

'''
files = sorted(glob.glob('PATH_OF_TRAIN/*_vh_clean_2.ply'))
files2 = sorted(glob.glob('PATH_OF_TRAIN/*_vh_clean_2.labels.ply'))
assert len(files) == len(files2)
'''

# files = sorted(glob.glob('/qys/cuda10docker/BPNet-main/dataset/scannetv2/scene0113_00/scene0113_00_vh_clean_2.ply'))
# files2 = sorted(glob.glob('/qys/cuda10docker/BPNet-main/dataset/scannetv2/scene0113_00/scene0113_00_vh_clean_2.labels.ply'))
# assert len(files) == len(files2)

# p = mp.Pool(processes=mp.cpu_count())
# p.map(f, files)
# p.close()
# p.join()

# files = sorted(glob.glob('/qys/cuda10docker/BPNet-main/dataset/testtrain/*_vh_clean_2.ply'))
# files2 = sorted(glob.glob('/qys/cuda10docker/BPNet-main/dataset/testtrain/*_vh_clean_2.labels.ply'))
files = sorted(glob.glob('/home/vr717/Documents/qys/code/NSEPN/BPNet_qys/dataset/testtrain/*_vh_clean_2.ply'))
files2 = sorted(glob.glob('/home/vr717/Documents/qys/code/NSEPN/BPNet_qys/dataset/testtrain/*_vh_clean_2.labels.ply'))
assert len(files) == len(files2)
print("fuck")
p = mp.Pool(processes=mp.cpu_count())
p.map(f, files)
p.close()
p.join()
