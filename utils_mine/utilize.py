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
if __name__=='__main__':

    path2 = '/home/slam/devdata/pointnerf/data_src/scannet/scans/scene0000_02/exported'
    path1 = '/home/slam/devdata/pointnerf/data_src/scannet/scans/scene0000_000102-T-blur/exported'
    merge_dataset(path1,path2)
    #merge_dataset(path1, path3)
    # delpath = os.path.join(path2,'pose')
    # filelist = os.listdir(delpath)
    # filelist = [i for i in filelist if int(i.split('.')[0])>=5919]
    # for file in filelist:
    #     os.remove(os.path.join(delpath,file))
    # print(filelist)

