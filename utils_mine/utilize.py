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
if __name__=='__main__':
    pass
    # R = cauc_transformationMatrix(75,45,0)
    # print(R)