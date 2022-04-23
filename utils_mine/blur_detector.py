import sys
import os
import pathlib
import argparse
sys.path.append(os.path.join(pathlib.Path(__file__).parent.absolute(), '..'))
import numpy as np
import cv2
from tqdm import tqdm
'''
README:
1. 处理的是scannet这个数据集，文件组织遵循point-nerf；
2. warning: 采取的是原数据集中直接删除blur图片的做法！！！！一定请备份数据集
3.手动的blur_img_list.txt放在exported下
'''
class Options:
    def __init__(self):
        self.opt = None
        self.parse()
    def parse(self):
        parser = argparse.ArgumentParser(description="Demo of argparse")
        parser.add_argument('--data_root',type=str, default='/home/slam/devdata/pointnerf/data_src/scannet/scans',help='root of dataset')
        parser.add_argument('--scan',type=str, default='scene0000_000102-T-blur',help='room which to scan')
        parser.add_argument('--auto_or_manual', type=str, default='1', help='0:auto to detect blur;1:manual to detect blur')
        parser.add_argument('--num_of_remove', type=str, default='150', help='set how may blur image to be remove if use automatic')
        self.opt = parser.parse_args()
        self.opt.dataset_dir = os.path.join(self.opt.data_root,self.opt.scan,'exported')
        # print(self.opt.dataset_dir)
class Dataset:
    def __init__(self,opt):
        self.auto_detect = (opt.auto_or_manual=='0')
        self.dataset_dir = opt.dataset_dir
        self.color_dir = os.path.join(self.dataset_dir,'color')
        self.depth_dir = os.path.join(self.dataset_dir,'depth')
        self.pose_dir = os.path.join(self.dataset_dir,'pose')
        self.num_of_remove = opt.num_of_remove
        self.color_path_list = [os.path.join(self.color_dir,f) for f in os.listdir(self.color_dir)]
        self.blur_id_list  = self.get_blur_id_list()
        self.all_id_list = list(range(len(self.color_path_list )))
        print('init down')
    def automatic_detect_blur(self):
        assert self.num_of_remove< len(self.color_path_list),'too much images to move! shrink the --num_of_remove'

        print('automatic detect blur image:')
        blur_score = []
        for img_path in tqdm(self.color_path_list):
            img = cv2.imread(img_path)
            gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            var =  cv2.Laplacian(gray_img, cv2.CV_64F).var()
            blur_score.append(var)
        blur_score = np.asarray(blur_score)
        blur_id_list = blur_score.argsort()[:self.num_of_remove] # 认为拉普拉斯算子的方差小的图片为blur图片
        return blur_id_list
    def get_blur_id_list(self):
        blur_id = []
        if(self.auto_detect):
            blur_id  = self.automatic_detect_blur()
        else:
            print(os.path.join(self.dataset_dir,'blur_img_list.txt'))
            assert os.path.exists(os.path.join(self.dataset_dir,'blur_img_list.txt')),"Choose Manual,but don't exists blur_img_list at {}".format(os.path.join(self.dataset_dir,'blur_img_list.txt'))
            print('Use manual blur_img_list')
            blur_id = np.loadtxt(os.path.join(self.dataset_dir,'blur_img_list.txt')).astype(np.int32)
        return blur_id

    def rename_dataset(self):
        # After remove blur images,we need to rename every sequence one by one
        self.not_blur_list = [id for id in self.all_id_list if id not in self.blur_id_list]

        for i in tqdm(range(len(self.not_blur_list))):
                ori_color_path = os.path.join(self.color_dir,'{}.jpg'.format(self.not_blur_list[i]))
                ori_depth_path = os.path.join(self.depth_dir, '{}.png'.format(self.not_blur_list[i]))
                ori_pose_path = os.path.join(self.pose_dir, '{}.txt'.format(self.not_blur_list[i]))
                new_color_path = os.path.join(self.color_dir,'{}.jpg'.format(i))
                new_depth_path = os.path.join(self.depth_dir, '{}.png'.format(i))
                new_pose_path = os.path.join(self.pose_dir, '{}.txt'.format(i))
                os.rename(ori_color_path,new_color_path)
                os.rename(ori_depth_path, new_depth_path)
                os.rename(ori_pose_path, new_pose_path)
    def remove_blur_data(self):
        print('data to remove:\n',self.blur_id_list)
        print('Removing blur data')
        self.blur_color_path_list = []
        self.blur_depth_path_list = []
        self.blur_pose_path_list = []
        blur_id_list = self.blur_id_list
        self.blur_id_list = []
        for itm in blur_id_list:
            if itm not in self.blur_id_list:
                self.blur_id_list.append(itm)
        assert len(self.blur_id_list) == len((set(self.blur_id_list))), 'Has repeatly itm,check it !'
        for id in tqdm(list(self.blur_id_list)):
            color_path = os.path.join(self.color_dir,'{}.jpg'.format(id))
            depth_path = os.path.join(self.depth_dir, '{}.png'.format(id))
            pose_path = os.path.join(self.pose_dir, '{}.txt'.format(id))
            os.remove(color_path)
            os.remove(depth_path)
            os.remove(pose_path)
        print('Remove blur data down')


def main():
    sparse = Options()
    opt = sparse.opt
    print(opt)
    ds = Dataset(opt)
    ds.remove_blur_data()
    ds.rename_dataset()
if __name__=="__main__":
    main()