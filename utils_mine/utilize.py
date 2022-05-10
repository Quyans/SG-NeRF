import math
import sys
import os
import pathlib
import shutil
import argparse
sys.path.append(os.path.join(pathlib.Path(__file__).parent.absolute(), '..'))
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image
from utils.util import to8b
import imageio
def merge_dataset(path1,path2):#path1:basepath,make path2 move to path1
    '''
    example
    path2 = '/home/slam/devdata/pointnerf/data_src/scannet/scans/scene0000_02/exported'
    path1 = '/home/slam/devdata/pointnerf/data_src/scannet/scans/scene0000_000102-T-blur/exported'
    merge_dataset(path1,path2)
    '''
    base = len(os.listdir(os.path.join(path1,'color')))
    imgsize = len(os.listdir(os.path.join(path2,'color')))
    print('base:',base,'imgsize:',imgsize)
    for i in tqdm(range(imgsize)):
            ori_color_path = os.path.join(path2,'color','{}.jpg'.format(i))
            ori_depth_path = os.path.join(path2,'depth', '{}.png'.format(i))
            ori_pose_path = os.path.join(path2,'pose', '{}.txt'.format(i))
            new_color_path = os.path.join(path1,'color','{}.jpg'.format(base+i))
            new_depth_path = os.path.join(path1,'depth', '{}.png'.format(base+i))
            new_pose_path = os.path.join(path1,'pose', '{}.txt'.format(base+i))
            shutil.copy(ori_color_path,new_color_path)
            shutil.copy(ori_depth_path, new_depth_path)
            shutil.copy(ori_pose_path, new_pose_path)

def cauc_RotationMatrix(alpha,beta,gamma):
    '''
    clockwise;
    外旋
    alpha:rotation around x axis
    betha: rotation around y axis
    gamma：rotation around z axis
    '''
    from math import cos,sin,pi
    alpha = alpha/180*pi
    beta = beta/180*pi
    gamma = gamma/180*pi
    Rx = np.array([
        [1,          0,           0],
        [0, cos(alpha), -sin(alpha)],
        [0, sin(alpha),  cos(alpha)]
    ])
    Ry = np.array([
        [cos(beta), 0,sin(beta)],
        [0,         1,        0],
        [-sin(beta),0,cos(beta)]
    ])
    Rz = np.array([
        [cos(gamma),-sin(gamma), 0],
        [sin(gamma), cos(gamma), 0],
        [         0,          0, 1]
    ])
    R = Rz@Ry@Rx
    return R

def cauc_transformationMatrix(rotationMatrix,posVector):
    posVector = np.array(posVector)
    res = np.concatenate([rotationMatrix,posVector[...,None]],axis = -1)
    tmp = np.array([0,0,0,1])
    res = np.concatenate([res,tmp[None,...]],axis = 0)
    return res
def visualize_dir_vector():
    from mpl_toolkits import mplot3d
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    from math import cos, sin, pi
    fig = plt.figure()
    # 创建3d绘图区域
    dir1 = np.array([-2,2,1])
    dir2 = np.array([1,2,1])
    dir1 = dir1/np.sqrt(np.sum(dir1*dir1))
    dir2 = dir2 / np.sqrt(np.sum(dir2 * dir2))
    dir1_proxy = np.array([dir1[0],dir1[1],0])
    dir2_proxy = np.array([dir2[0], dir2[1], 0])
    clockwise = 1 if torch.cross(torch.Tensor(dir1_proxy),torch.Tensor(dir2_proxy))[2]>0 else -1
    clockwiseangle = clockwise * torch.arccos(torch.sum(dir1_proxy*dir2_proxy/torch.norm(torch.Tensor(dir1_proxy))/torch.norm(torch.Tensor(dir2_proxy))))
    print(clockwiseangle)
    Rz = np.array([
        [cos(clockwiseangle),-sin(clockwiseangle), 0],
        [sin(clockwiseangle), cos(clockwiseangle), 0],
        [         0,          0, 1]
    ])
    rot_dir1 = Rz@dir1
    print(rot_dir1)

    ax = plt.axes(projection='3d')
    rot_x1 = np.linspace(0, rot_dir1[0], 100)
    rot_y1 = np.linspace(0, rot_dir1[1], 100)
    rot_z1 = np.linspace(0, rot_dir1[2], 100)
    ax.plot3D(rot_x1, rot_y1, rot_z1, 'black')

    x1 = np.linspace(0, dir1[0], 100)
    y1 = np.linspace(0, dir1[1], 100)
    z1 = np.linspace(0, dir1[2], 100)
    ax.plot3D(x1, y1, z1, 'red')
    x2 = np.linspace(0, dir2[0], 100)
    y2 = np.linspace(0, dir2[1], 100)
    z2 = np.linspace(0, dir2[2], 100)
    ax.plot3D(x2, y2, z2, 'blue')
    # #ax.plot_wireframe('X','Y','Z', color='black')
    # x3 = np.linspace(0, dir1_proxy[0], 100)
    # y3 = np.linspace(0, dir1_proxy[1], 100)
    # z3 = np.linspace(0, dir1_proxy[2], 100)
    # ax.plot3D(x3, y3, z3, 'green')
    # x4 = np.linspace(0, dir2_proxy[0], 100)
    # y4 = np.linspace(0, dir2_proxy[1], 100)
    # z4 = np.linspace(0, dir2_proxy[2], 100)
    # ax.plot3D(x4, y4, z4, 'yellow')

    ax.axis(xmin=-1,xmax=1,ymin=-1,ymax=1)
    plt.show()
if __name__=='__main__':

    visualize_dir_vector()
    # R = cauc_transformationMatrix(75,45,0)
    # print(R)