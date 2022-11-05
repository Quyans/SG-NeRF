import csv
import cv2
import re
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
def mapping_labels_from_Scannet2Nyu40(mapfile_path,label_path):
    '''
    paras:
        mapfile_path: path of 'scannetv2-labels.combined.tsv'
    '''
    with open(mapfile_path, mode="r", encoding="utf-8") as f:
        dictionary = dict()
        lines = [line.rstrip() for line in f][1:]
        dictionary[0] = 0
        for i in range(len(lines)):
            value = re.split(r'\t',lines[i])
            scannet_id = int(value[0])
            nyu40_id = int(value[4])
            dictionary[scannet_id] = nyu40_id
    for i in tqdm(range(733)):
        semantic_path = os.path.join(label_path, "label-filt/{}.png".format(i))
        gt_semantic_img = np.array(Image.open(semantic_path).convert(mode='I'))
        h,w = gt_semantic_img.shape[0],gt_semantic_img.shape[1]
        gt_semantic_img = gt_semantic_img.flatten()
        gt_semantic_img = np.array(list(map(lambda x:dictionary[x],gt_semantic_img)))
        gt_semantic_img = gt_semantic_img.reshape(h,w)
        save_path = os.path.join(label_path, "label-filt/{}.png".format(i))
        cv2.imwrite(save_path,gt_semantic_img)

if __name__=='__main__':
    mapping_labels_from_Scannet2Nyu40('/home/slam/devdata/NSEPN/data_src/scannet/scans/scannetv2-labels.combined.tsv',"/home/slam/devdata/NSEPN/data_src/scannet/scans/scene0113_00")
