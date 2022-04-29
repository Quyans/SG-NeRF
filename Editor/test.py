import torch.nn as nn
import sys
import os
import pathlib
import argparse
import open3d as o3d
import torch.cuda
from plyfile import PlyData, PlyElement
import numpy as np
sys.path.append(os.path.join(pathlib.Path(__file__).parent.absolute(), '..'))
from utils_mine.utilize import  *
#np.set_printoptions(suppress=True)  # 取消默认科学计数法，open3d无法读取科学计数法表示

import cv2
from tqdm import tqdm
from Editor.pointcloud import *
from Editor.checkpoints_controller import *
from Editor.pointcloud_editor import *
class Options:
    def __init__(self):
        self.opt = None
        self.parse()
    def parse(self):
        parser = argparse.ArgumentParser(description="Argparse of  point_editor")
        parser.add_argument('--checkpoints_root',
                            type=str,
                            default='/home/slam/devdata/pointnerf/checkpoints/scannet/13-scene0113-rotationinvariance',#/home/slam/devdata/pointnerf/checkpoints/scannet/scene000-T
                            help='root of checkpoints datasets')
        parser.add_argument('--gpu_ids',
                            type=str,
                            default='0',
                            help='gpu ids: e.g. 0  0,1,2, 0,2')

        self.opt = parser.parse_args()

        # print(self.opt.dataset_dir)

def main():
    sparse = Options()
    opt = sparse.opt
    print(opt)
    # '''
    # 测试从checkpoints转ply
    # '''
    # cpc = CheckpointsController(opt)
    # neural_pcd = cpc.load_checkpoints_as_nerualpcd()
    # neural_pcd.save_as_ply('origin_scene')
    # '''
    # 测试读ply:这一步中间，用mesh手抠一个物体，命名为sofa_meshlabpcd.ply~！~！~！~！~！~！~！~！
    # '''
    # scene_npcd = Neural_pointcloud(opt)
    # scene_npcd.load_from_ply('scene_origin')
    # '''
    # 测试从meshlab cropped后在带feature的点云中找
    # '''
    # object_mpcd = Meshlab_pointcloud(opt)
    # object_mpcd.load_from_meshlabfile("sofa1")
    # object_npcd = object_mpcd.meshlabpcd2neuralpcd(scene_npcd)
    # object_npcd.save_as_ply('sofa1')
    '''
    测试editor crop方法
    '''
    # scene_npcd = Neural_pointcloud(opt)
    # scene_npcd.load_from_ply('scene_origin')
    # object_npcd = Neural_pointcloud(opt)
    # object_npcd.load_from_ply('sofa1')
    # pce = PointCloudEditor(opt)
    # npcd_cropped = pce.crop_point_cloud(object_npcd,scene_npcd)
    # npcd_cropped.save_as_ply('nosofa1_scene')

    '''
    测试将neural point cloud 写回 checkpoints
    '''
    sofa_npcd = Neural_pointcloud(opt)
    sofa_npcd.load_from_ply('sofa1')#'nosofa'
    scene_npcd = Neural_pointcloud(opt)
    scene_npcd.load_from_ply('nosofa1_scene')  # 'nosofa'
    pce = PointCloudEditor(opt)
    for deg in (range(360)):
        transMatrix = cauc_transformationMatrix(cauc_RotationMatrix(0,0,deg),[0,0,0])
        transed_sofa = pce.translation_point_cloud_local(sofa_npcd,transMatrix)
        new_scene = pce.add_point_cloud(transed_sofa,scene_npcd)
        new_scene.save_as_ply('sofa_trans_scene'+str(deg))
        cpc = CheckpointsController(opt)
        cpc.save_checkpoints_from_neuralpcd(new_scene,'sofa-trans'+str(deg))
if __name__=="__main__":
    main()
    print('~finish~')