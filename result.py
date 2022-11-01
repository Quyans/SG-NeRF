import torch
import numpy as np

from plyfile import PlyData, PlyElement


import torch
import numpy as np
resultPath = "/qys/cuda10docker/BPNet-main/Data/ScanNet/"

# pred = np.load(resultPath + "result/best/pred.npy")
# gt = np.load(resultPath + "result/best/gt.npy")
# # ply = np.load()
# print("0")
def float2color(zero2one):
    x = zero2one * 256 * 256 * 256
    r = x % 256
    g = ((x - r)/256 % 256)
    b = ((x - r - g * 256)/(256*256) % 256)
    r = round(float(r/256),2)
    g = round(float(g/256),2)
    b = round(float(b/256),2)
    return [r,g,b]

def Color_To_RGB(color):
    
    b = color / (256 * 256)
    g = (color - b * 256 * 256) / 256
    r = color - b * 256 * 256 - g * 256
    return [r,g,b]


colordict = {
    0:[151,223,137],
    1:[174,198,232],
    2:[31,120,180],
    3:[255,188,120],
    4:[188,189,35],
    5:[140,86,74],
    6:[255,152,151],
    7:[213,39,40],
    8:[196,176,213],
    9:[148,103,188],
    10:[196,156,148],
    11:[23,190,208],
    12:[247,183,210],
    13:[218,219,141],
    14:[254,127,14],
    15:[227,119,194],
    16:[158,218,229],
    17:[43,160,45],
    18:[112,128,144],
    19:[82,83,163],
    255:[255,255,170]    
}

# # 顏色值
# for i in pred:
#     rgb = colordict[i]

# pth = np.load(resultPath+"train/scene0241_00_vh_clean_2.pth")
# print("0")


import os
def main():
    
    # pth = torch.load("scene0241_00_vh_clean_2.pth")
    # points = torch.tensor(pth[0])
    # color = torch.tensor(pth[1])
    # gt = pth[2]

    basedir = "/home/vr717/Documents/qys/code/NSEPN_ori/NSEPN/checkpoints/scannet/scene024102_Semantic_step5_debug_640x480"
    pred = np.loadtxt(os.path.join(basedir,"predict_label_10001.txt"),delimiter=' ')
    label = pred[:,3]
    pred_colors = []
    for ind in range(len(label)):
        pred_colors.append(colordict[label[ind]])
    
    # for i in pred:
    #     if  i!=2:
    #         print(i)
    # pred =torch.t( torch.Tensor(pred))
    matrix =  torch.cat((torch.Tensor(pred[:,0:3]),torch.Tensor(pred_colors)),dim=1)
    np.savetxt(os.path.join(basedir,'predict_pointA.txt'), matrix, fmt='%f', delimiter=',')
    print(0)

if __name__ == '__main__':
    main()