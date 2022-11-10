#!/usr/bin/env python
import sys
from collections import OrderedDict
from turtle import forward
import numpy as np

import torch
from torch import nn, softmax
import torch.nn.functional as F

from .unet_2d import ResUnet as model2D
from .unet_3d import mink_unet as model3D
import MinkowskiEngine as ME
from .bpm import Linking


# bpnet 
from MinkowskiEngine import SparseTensor, CoordsManager
import bpnet_dataset.augmentation_2d as t_2d
from bpnet_dataset.voxelizer import Voxelizer
from asyncio.log import logger
import random
import imageio
import math
from PIL import Image

def state_dict_remove_moudle(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # name = k[7:]  # remove 'module.' of dataparallel
        name = k.replace('module.', '')
        new_state_dict[name] = v
    return new_state_dict


def constructor3d(**kwargs):
    model = model3D(**kwargs)
    # model = model.cuda()
    return model


def constructor2d(**kwargs):
    model = model2D(**kwargs)
    # model = model.cuda()
    return model


# create camera intrinsics
def make_intrinsic(fx, fy, mx, my):
    intrinsic = np.eye(4)
    intrinsic[0][0] = fx
    intrinsic[1][1] = fy
    intrinsic[0][2] = mx
    intrinsic[1][2] = my
    return intrinsic
# create camera intrinsics
def adjust_intrinsic(intrinsic, intrinsic_image_dim, image_dim):
    if intrinsic_image_dim == image_dim:
        return intrinsic
    resize_width = int(math.floor(image_dim[1] * float(intrinsic_image_dim[0]) / float(intrinsic_image_dim[1])))
    intrinsic[0, 0] *= float(resize_width) / float(intrinsic_image_dim[0])
    intrinsic[1, 1] *= float(image_dim[1]) / float(intrinsic_image_dim[1])
    # account for cropping here
    intrinsic[0, 2] *= float(image_dim[0] - 1) / float(intrinsic_image_dim[0] - 1)
    intrinsic[1, 2] *= float(image_dim[1] - 1) / float(intrinsic_image_dim[1] - 1)
    return intrinsic
class LinkCreator(object):
    def __init__(self, fx=577.870605, fy=577.870605, mx=319.5, my=239.5, image_dim=(320, 240), voxelSize=0.05):
        self.intricsic = make_intrinsic(fx=fx, fy=fy, mx=mx, my=my)
        self.intricsic = adjust_intrinsic(self.intricsic, intrinsic_image_dim=[640, 480], image_dim=image_dim)
        self.imageDim = image_dim
        self.voxel_size = voxelSize

    def computeLinking(self, camera_to_world, coords, depth):
        """
        :param camera_to_world: 4 x 4
        :param coords: N x 3 format
        :param depth: H x W format
        :return: linking, N x 3 format, (H,W,mask)
        """
        link = np.zeros((3, coords.shape[0]), dtype=np.int)
        coordsNew = np.concatenate([coords, np.ones([coords.shape[0], 1])], axis=1).T
        assert coordsNew.shape[0] == 4, "[!] Shape error"

        world_to_camera = np.linalg.inv(camera_to_world)
        p = np.matmul(world_to_camera, coordsNew)
        p[0] = (p[0] * self.intricsic[0][0]) / p[2] + self.intricsic[0][2]
        p[1] = (p[1] * self.intricsic[1][1]) / p[2] + self.intricsic[1][2]
        pi = np.round(p).astype(np.int)
        inside_mask = (pi[0] >= 0) * (pi[1] >= 0) \
                      * (pi[0] < self.imageDim[0]) * (pi[1] < self.imageDim[1])
        occlusion_mask = np.abs(depth[pi[1][inside_mask], pi[0][inside_mask]]
                                - p[2][inside_mask]) <= self.voxel_size
        inside_mask[inside_mask == True] = occlusion_mask
        link[0][inside_mask] = pi[1][inside_mask]
        link[1][inside_mask] = pi[0][inside_mask]
        link[2][inside_mask] = 1

        return link.T

class BPNet(nn.Module):

    def __init__(self, cfg=None):
        super(BPNet, self).__init__()
        self.viewNum = cfg.viewNum

        voxelSize = 0.05
        self.SCALE_AUGMENTATION_BOUND = (0.9, 1.1)
        self.ROTATION_AUGMENTATION_BOUND = ((-np.pi / 64, np.pi / 64), (-np.pi / 64, np.pi / 64), (-np.pi,
                                                                                            np.pi))
        self.TRANSLATION_AUGMENTATION_RATIO_BOUND = ((-0.2, 0.2), (-0.2, 0.2), (0, 0))
        self.ELASTIC_DISTORT_PARAMS = ((0.2, 0.4), (0.8, 1.6))

        self.ROTATION_AXIS = 'z'
        self.LOCFEAT_IDX = 2
        
        # self.VIEW_NUM = 3 todo设置可变的
        # self.IMG_DIM = (320, 240)

        # 原本 use_augmentation = true
        self.IMG_DIM = cfg.img_wh
        self.voxelizer = Voxelizer(
            voxel_size=voxelSize,
            clip_bound=None,
            use_augmentation=False,
            scale_augmentation_bound=self.SCALE_AUGMENTATION_BOUND,
            rotation_augmentation_bound=self.ROTATION_AUGMENTATION_BOUND,
            translation_augmentation_ratio_bound=self.TRANSLATION_AUGMENTATION_RATIO_BOUND)
        self.data2D_path = []

         # 2d图片到3D图片的lable对齐
        self.remapper = np.ones(256) * 255
        for i, x in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]):
            self.remapper[x] = i
        self.linkCreator = LinkCreator(image_dim=self.IMG_DIM, voxelSize=voxelSize)
        
         # 2D AUG
        value_scale = 255
        mean = [0.485, 0.456, 0.406]
        mean = [item * value_scale for item in mean]
        std = [0.229, 0.224, 0.225]
        std = [item * value_scale for item in std]

        self.aug = cfg.aug
        if self.aug:
            self.transform_2d = t_2d.Compose([
                t_2d.RandomGaussianBlur(),
                t_2d.Crop([self.IMG_DIM[1] + 1, self.IMG_DIM[0] + 1], crop_type='rand', padding=mean,
                        ignore_label=255),
                t_2d.ToTensor(),
                t_2d.Normalize(mean=mean, std=std)])
        else:
            # 这里设置为center 而不是rand
            self.transform_2d = t_2d.Compose([
                # t_2d.Crop([self.IMG_DIM[1] + 1, self.IMG_DIM[0] + 1], crop_type='center', padding=mean,
                #         ignore_label=255),
                t_2d.ToTensor(),
                t_2d.Normalize(mean=mean, std=std)])



        # 2D
        net2d = constructor2d(layers=cfg.layers_2d, classes=cfg.classes)
        self.layer0_2d = net2d.layer0
        self.layer1_2d = net2d.layer1
        self.layer2_2d = net2d.layer2
        self.layer3_2d = net2d.layer3
        self.layer4_2d = net2d.layer4
        self.up4_2d = net2d.up4
        self.delayer4_2d = net2d.delayer4
        self.up3_2d = net2d.up3
        self.delayer3_2d = net2d.delayer3
        self.up2_2d = net2d.up2
        self.delayer2_2d = net2d.delayer2
        self.cls_2d = net2d.cls

        # 3D
        net3d = constructor3d(in_channels=3, out_channels=cfg.classes, D=3, arch=cfg.arch_3d)
        self.layer0_3d = nn.Sequential(net3d.conv0p1s1, net3d.bn0, net3d.relu)
        self.layer1_3d = nn.Sequential(net3d.conv1p1s2, net3d.bn1, net3d.relu, net3d.block1)
        self.layer2_3d = nn.Sequential(net3d.conv2p2s2, net3d.bn2, net3d.relu, net3d.block2)
        self.layer3_3d = nn.Sequential(net3d.conv3p4s2, net3d.bn3, net3d.relu, net3d.block3)
        self.layer4_3d = nn.Sequential(net3d.conv4p8s2, net3d.bn4, net3d.relu, net3d.block4)
        self.layer5_3d = nn.Sequential(net3d.convtr4p16s2, net3d.bntr4, net3d.relu)
        self.layer6_3d = nn.Sequential(net3d.block5, net3d.convtr5p8s2, net3d.bntr5, net3d.relu)
        self.layer7_3d = nn.Sequential(net3d.block6, net3d.convtr6p4s2, net3d.bntr6, net3d.relu)
        self.layer8_3d = nn.Sequential(net3d.block7, net3d.convtr7p2s2, net3d.bntr7, net3d.relu)
        self.layer9_3d = net3d.block8
        self.cls_3d = net3d.final

        # Linker
        self.linker_p2 = Linking(96, net3d.PLANES[6], viewNum=self.viewNum)
        self.linker_p3 = Linking(128, net3d.PLANES[5], viewNum=self.viewNum)
        self.linker_p4 = Linking(256, net3d.PLANES[4], viewNum=self.viewNum)
        self.linker_p5 = Linking(512, net3d.PLANES[3], viewNum=self.viewNum)


    def forward(self, sparse_3d, images, links):
        """
        images:BCHWV               
        """
        # 2D feature extract
        x_size = images.size()
        h, w = x_size[2], x_size[3]
        # data_2d = images.permute(0, 1, 2, 3).contiguous()  # VBCHW
        data_2d = images.permute(4, 0, 1, 2, 3).contiguous()  # VBCHW
        data_2d = data_2d.view(x_size[0] * x_size[4], x_size[1], x_size[2], x_size[3])
        
        x = self.layer0_2d(data_2d)  # 1/4
        x2 = self.layer1_2d(x)  # 1/4
        x3 = self.layer2_2d(x2)  # 1/8
        x4 = self.layer3_2d(x3)  # 1/16
        x5 = self.layer4_2d(x4)  # 1/32

        # 3D feature extract
        out_p1 = self.layer0_3d(sparse_3d)
        out_b1p2 = self.layer1_3d(out_p1)
        out_b2p4 = self.layer2_3d(out_b1p2)
        out_b3p8 = self.layer3_3d(out_b2p4)
        out_b4p16 = self.layer4_3d(out_b3p8)  # corresponding to FPN p5

        # Linking @ p5
        V_B, C, H, W = x5.shape
        links_current_level = links.clone()
        links_current_level[:, 1:3, :] = ((H - 1.) / (h - 1.) * links_current_level[:, 1:3, :].float()).int()
        # 这里有个随机性 在linker这里，out_b4p16不会变 x5不变 links_current_level会变
        fused_3d_p5, fused_2d_p5 = self.linker_p5(x5, out_b4p16, links_current_level, init_3d_data=sparse_3d)

        
        p4 = self.up4_2d(F.interpolate(fused_2d_p5, x4.shape[-2:], mode='bilinear', align_corners=True))
        p4 = torch.cat([p4, x4], dim=1)
        p4 = self.delayer4_2d(p4)
        feat_3d = self.layer5_3d(fused_3d_p5)  # corresponding to FPN p4

        # Linking @ p4
        V_B, C, H, W = p4.shape
        links_current_level = links.clone()
        links_current_level[:, 1:3, :] = ((H - 1.) / (h - 1.) * links_current_level[:, 1:3, :].float()).int()
        fused_3d_p4, fused_2d_p4 = self.linker_p4(p4, feat_3d, links_current_level, init_3d_data=sparse_3d)

        p3 = self.up3_2d(F.interpolate(fused_2d_p4, x3.shape[-2:], mode='bilinear', align_corners=True))
        p3 = torch.cat([p3, x3], dim=1)
        p3 = self.delayer3_2d(p3)
        feat_3d = self.layer6_3d(ME.cat(fused_3d_p4, out_b3p8))  # corresponding to FPN p3

        # Linking @ p3
        V_B, C, H, W = p3.shape
        links_current_level = links.clone()
        links_current_level[:, 1:3, :] = ((H - 1.) / (h - 1.) * links_current_level[:, 1:3, :].float()).int()
        fused_3d_p3, fused_2d_p3 = self.linker_p3(p3, feat_3d, links_current_level, init_3d_data=sparse_3d)

        p2 = self.up2_2d(F.interpolate(fused_2d_p3, x2.shape[-2:], mode='bilinear', align_corners=True))
        p2 = torch.cat([p2, x2], dim=1)
        p2 = self.delayer2_2d(p2)
        feat_3d = self.layer7_3d(ME.cat(fused_3d_p3, out_b2p4))  # corresponding to FPN p2

        # Linking @ p2
        V_B, C, H, W = p2.shape
        links_current_level = links.clone()
        links_current_level[:, 1:3, :] = ((H - 1.) / (h - 1.) * links_current_level[:, 1:3, :].float()).int()
        fused_3d_p2, fused_2d_p2 = self.linker_p2(p2, feat_3d, links_current_level, init_3d_data=sparse_3d)

        feat_3d = self.layer8_3d(ME.cat(fused_3d_p2, out_b1p2))      #84456,96

        # Res
        # pdb.set_trace()
        res_2d = self.cls_2d(fused_2d_p2)
        res_2d = F.interpolate(res_2d, size=(h, w), mode='bilinear', align_corners=True)
        V_B, C, H, W = res_2d.shape
        res_2d = res_2d.view(self.viewNum, -1, C, H, W).permute(1, 2, 3, 4, 0)

        res_3d_feat = self.layer9_3d(ME.cat(feat_3d, out_p1)) #18404,96
        res_3d = self.cls_3d(res_3d_feat) #18404,20    input:sparse_3d 18404 3

        # 将输出softmax到0~1里面  max概率小于0.5的有386个体素 共两万个   统计方法 np.sum((softmax3d.detach().max(1)[0]<0.5).cpu().numpy()!=False)
        softmax3d = torch.softmax(res_3d.F,dim=1)
        # temnum = (softmax3d.detach().max(1)[0]<0.5).cpu().numpy()
        
        return softmax3d, res_2d,res_3d_feat
    
    
    
    # def get_2d(self,train_id_paths, coords: np.ndarray):
    #     """
    #     :param      coords: Nx3
    #     :return:    imgs:   CxHxWxV Tensor
    #                 labels: HxWxV Tensor
    #                 links: Nx4xV(1,H,W,mask) Tensor
    #     """
    #     # 默认为False
    #     self.val_benchmark = False
    #     frames_path = train_id_paths[0]
    #     #frames_path 是這個場景的训练集所有图片 对于scannet241是100帧
    #     # print(room_id)
    #     partial = int(len(frames_path) / self.viewNum)
    #     imgs, labels, links = [], [], []
    #     for v in range(self.viewNum):
    #         if not self.val_benchmark:
    #             f = random.sample(frames_path[v * partial:v * partial + partial], k=1)[0][0]
    #         else:
    #             select_id = (v * partial+self.offset) % len(frames_path)
    #             # select_id = (v * partial+partial//2)
    #             f = frames_path[select_id]
    #         # pdb.set_trace()
    #         img = imageio.imread(f)
    #         label = imageio.imread(f.replace('color', 'label').replace('jpg', 'png'))
            
    #         # label = self.remapper[label] # 这里可以不用搞因为这里的语义label都已经处理过了，如果没有处理过需要用到这里
            
    #         depth = imageio.imread(f.replace('color', 'depth').replace('jpg', 'png')) / 1000.0  # convert to meter
    #         posePath = f.replace('color', 'pose').replace('.jpg', '.txt')
    #         pose = np.asarray(
    #             [[float(x[0]), float(x[1]), float(x[2]), float(x[3])] for x in
    #              (x.split(" ") for x in open(posePath).read().splitlines())]
    #         )
    #         # pdb.set_trace()
    #         link = np.ones([coords.shape[0], 4], dtype=np.int)
    #         link[:, 1:4] = self.linkCreator.computeLinking(pose, coords, depth)
    #         temimg = img
    #         temlabel =label
    #         img = self.transform_2d(img)
    #         imgs.append(img)
    #         # labels.append(label)
    #         links.append(link)

    #     imgs = torch.stack(imgs, dim=-1)
    #     # labels = torch.stack(labels, dim=-1)
    #     links = np.stack(links, axis=-1)
    #     links = torch.from_numpy(links)
    #     return imgs, links



    def get_2d(self,train_id_paths, coords: np.ndarray,image_path):
    # def get_2d(self, coords: np.ndarray,train_id_paths,image_path):
        """
        :param      coords: Nx3
        :return:    imgs:   CxHxWxV Tensor
                    labels: HxWxV Tensor
                    links: Nx4xV(1,H,W,mask) Tensor
        """
        # 默认为False
        self.val_benchmark = False
        frames_path = train_id_paths[0]
        # frames_path = ['/home/vr717/Documents/qys/code/NSEPN_ori/NSEPN/data_src/scannet/scans/scene046/scene0046_02/exported/color/0.jpg','/home/vr717/Documents/qys/code/NSEPN_ori/NSEPN/data_src/scannet/scans/scene046/scene0046_02/exported/color/5.jpg','/home/vr717/Documents/qys/code/NSEPN_ori/NSEPN/data_src/scannet/scans/scene046/scene0046_02/exported/color/5.jpg','/home/vr717/Documents/qys/code/NSEPN_ori/NSEPN/data_src/scannet/scans/scene046/scene0046_02/exported/color/5.jpg','/home/vr717/Documents/qys/code/NSEPN_ori/NSEPN/data_src/scannet/scans/scene046/scene0046_02/exported/color/5.jpg','/home/vr717/Documents/qys/code/NSEPN_ori/NSEPN/data_src/scannet/scans/scene046/scene0046_02/exported/color/5.jpg','/home/vr717/Documents/qys/code/NSEPN_ori/NSEPN/data_src/scannet/scans/scene046/scene0046_02/exported/color/5.jpg','/home/vr717/Documents/qys/code/NSEPN_ori/NSEPN/data_src/scannet/scans/scene046/scene0046_02/exported/color/5.jpg']
        # image_path = "/home/vr717/Documents/qys/code/NSEPN_ori/NSEPN/data_src/scannet/scans/scene046/scene0046_02/exported/color/1120.jpg"
        #frames_path 是這個場景的训练集所有图片 对于scannet241是100帧
        # print(room_id)
        partial = int(len(frames_path) / self.viewNum)
        imgs, labels, links = [], [], []
        for v in range(self.viewNum):

            # 这里三个f都是同一个文件
            if isinstance(image_path,list):
                image_pa = image_path[0]
            else:
                image_pa = image_path
            
            
            if tuple([image_pa,]) in frames_path[v * partial:v * partial + partial]:
                imgio = imageio.imread(image_pa)
                img = Image.open(image_pa)
                img = img.resize(self.IMG_DIM, Image.NEAREST)
                img = np.array(img, dtype='float32')
                # label = imageio.imread(image_pa.replace('color', 'label').replace('jpg', 'png'))
                label = Image.open(image_pa.replace('color', 'label').replace('jpg', 'png'))
                label = label.resize(self.IMG_DIM, Image.NEAREST)
                label = np.array(label, dtype='float32')
                # label = self.remapper[label] # 这里可以不用搞因为这里的语义label都已经处理过了，如果没有处理过需要用到这里
                # depth = imageio.imread(image_pa.replace('color', 'depth').replace('jpg', 'png')) / 1000.0  # convert to meter
                depth = Image.open(image_pa.replace('color', 'depth').replace('jpg', 'png'))
                depth = depth.resize(self.IMG_DIM, Image.NEAREST)
                depth = np.array(depth, dtype='float32') / 1000.0  # convert to meter

                posePath = image_pa.replace('color', 'pose').replace('.jpg', '.txt')
                pose = np.asarray(
                    [[float(x[0]), float(x[1]), float(x[2]), float(x[3])] for x in
                    (x.split(" ") for x in open(posePath).read().splitlines())]
                )
                # pdb.set_trace()
                link = np.ones([coords.shape[0], 4], dtype=np.int)
                link[:, 1:4] = self.linkCreator.computeLinking(pose, coords, depth)

                img = self.transform_2d(img)
                imgs.insert(0,img)
                # labels.append(label)
                links.insert(0,link)
                continue
            f = random.sample(frames_path[v * partial:v * partial + partial], k=1)[0][0]
            # if not self.val_benchmark:
            #     f = random.sample(frames_path[v * partial:v * partial + partial], k=1)[0][0]
            # else:
            #     select_id = (v * partial+self.offset) % len(frames_path)
                # select_id = (v * partial+partial//2)
                # f = frames_path[select_id]
            # pdb.set_trace()
            imgio = imageio.imread(image_pa)
            img = Image.open(image_pa)
            img = img.resize(self.IMG_DIM, Image.NEAREST)
            img = np.array(img, dtype='float32')
            # label = imageio.imread(image_pa.replace('color', 'label').replace('jpg', 'png'))
            label = Image.open(image_pa.replace('color', 'label').replace('jpg', 'png'))
            label = label.resize(self.IMG_DIM, Image.NEAREST)
            label = np.array(label, dtype='float32')
            # label = self.remapper[label] # 这里可以不用搞因为这里的语义label都已经处理过了，如果没有处理过需要用到这里
            # depth = imageio.imread(image_pa.replace('color', 'depth').replace('jpg', 'png')) / 1000.0  # convert to meter
            depth = Image.open(image_pa.replace('color', 'depth').replace('jpg', 'png'))
            depth = depth.resize(self.IMG_DIM, Image.NEAREST)
            depth = np.array(depth, dtype='float32') / 1000.0  # convert to meter

            posePath = image_pa.replace('color', 'pose').replace('.jpg', '.txt')
            pose = np.asarray(
                [[float(x[0]), float(x[1]), float(x[2]), float(x[3])] for x in
                (x.split(" ") for x in open(posePath).read().splitlines())]
            )
            # pdb.set_trace()
            link = np.ones([coords.shape[0], 4], dtype=np.int)
            link[:, 1:4] = self.linkCreator.computeLinking(pose, coords, depth)

            img = self.transform_2d(img)
            imgs.append(img)
            # labels.append(label)
            links.append(link)

        imgs = torch.stack(imgs, dim=-1)
        # labels = torch.stack(labels, dim=-1)
        links = np.stack(links, axis=-1)
        links = torch.from_numpy(links)
        return imgs, links


    # def train_bpnet(self,locs_in,feats_in):
    def train_bpnet(self,locs_in,feats_in,train_id_paths,image_path):
        
        # colors, links = self.get_2d(locs_in)
        colors, links = self.get_2d(train_id_paths, locs_in,image_path)
        
        locs = self.prevoxel_transforms(locs_in) if self.aug else locs_in
        locs, feats, labels_3d, inds_reconstruct, links = self.voxelizer.voxelize(locs, feats_in, link=links)
        coords = torch.from_numpy(locs).int()
        coords = torch.cat((torch.ones(coords.shape[0], 1, dtype=torch.int), coords), dim=1)
        # feats = torch.from_numpy(feats).float() / 127.5 - 1.
        feats = torch.from_numpy(feats).float() / 127.5 -1.
        # feats = torch.from_numpy(feats).float()

        # labels_3d = labels_in
        # labels_3d = torch.from_numpy(labels_3d).long()
        
        inds_reconstruct = torch.from_numpy(inds_reconstruct).long()
        
        colors = torch.stack([colors])
        # labels_2d = torch.stack([labels_2d])
        
        # 在这里需要对数据的进行一步预处理 对应scanNetCross.py 239行
        coords[:,0] *= 0
        links[:,0,:]*=0
        
        sinput = SparseTensor(feats.cuda(non_blocking=True), coords.cuda(non_blocking=True))
        # sinput = SparseTensor(feats,coords)
        colors, links = colors.cuda(non_blocking=True), links.cuda(non_blocking=True)
        # labels_3d, labels_2d = labels_3d.cuda(non_blocking=True), labels_2d.cuda(non_blocking=True)

        output_3d_prob, output_2d, point_inside_feat_ST =  self.forward(sinput, colors, links)
        output_3d_prob = output_3d_prob[inds_reconstruct, :]

        point_inside_feat = point_inside_feat_ST.F[inds_reconstruct,:]
        output_3d = output_3d_prob.detach().max(1)[1]
        # output_3d = output_3d.resize(output_3d.shape[0],1).cpu().numpy() #将数据从 [125988] -> [125988,1]
        output_2d = output_2d.detach().max(1)[1]
        output_2d = output_2d[0,:,:,0]
        output_2d = output_2d.reshape(1,self.IMG_DIM[1],self.IMG_DIM[0],1)

        # 返回的output_2d 应该是[240,320,1 ] 
        # output_3d  [122598]  0~19
        # output_3d_prob [122598,20]
        return output_3d,output_3d_prob,output_2d,point_inside_feat


