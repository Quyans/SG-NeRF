import sys
import os
import pathlib
import argparse
sys.path.append(os.path.join(pathlib.Path(__file__).parent.absolute(), '..'))
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image
from utils.util import to8b
import imageio
'''
README:
1. 处理的是scannet这个数据集，文件组织遵循point-nerf；
'''
class Options:
    def __init__(self):
        self.opt = None
        self.parse()
    def parse(self):
        parser = argparse.ArgumentParser(description="Demo of argparse")
        parser.add_argument('--data_root',type=str, default='/home/slam/devdata/pointnerf/checkpoints/scannet/5-scene000-colmap_rmBlur4000',help='root of rendering result(It is probably in checkpoints file)')
        parser.add_argument('--unit',type=str, default='iter',choices=['iter','pose'],help='how to generate video,iter means show the fist pic every several iters,pose mean show the latest iter every camera poss')
        #parser.add_argument('--num_iter', type=int,default=10000,help='n iter you want to generator')
        parser.add_argument('--video_format', type=str, default='gif',choices=['mp4','gif','mov'],help='video format')
        parser.add_argument('--fps', type=int, default=20,help='frame per second of video')

        self.opt = parser.parse_args()

        # print(self.opt.dataset_dir)

class VideoGenerator:
    def __init__(self,opt):
        self.data_root = opt.data_root
        self.unit = opt.unit
        self.video_format = opt.video_format
        self.img_path_list = []
        self.fps = opt.fps
        dirs = [i for i in os.listdir(self.data_root) if i.startswith("test_")]
        assert len(dirs) > 0, "No rendering result is found"
        numlist = [int(i[5:]) for i in dirs]
        numlist.sort()
        if self.unit == 'pose':
            latestnum = str(numlist[-1])
            del numlist,dirs
            self.latest_dir = os.path.join(self.data_root,'test_'+latestnum,'images')
            print('latest directory:', self.latest_dir)
            img_path_list = [i for i in os.listdir(self.latest_dir) if i.endswith("-coarse_raycolor.png")]
            img_path_list.sort()
            self.img_path_list = [os.path.join(self.latest_dir,img_path) for img_path in img_path_list]
        else:#unit=='iter'
            self.img_path_list = [os.path.join(self.data_root,'test_'+str(i),'images','step-0000-coarse_raycolor.png') for i in numlist]
    def gen_video(self):
        img_lst = []
        for img_path in tqdm(self.img_path_list):
            image = np.asarray(Image.open(img_path))
            img_lst.append(image)
        video_save_path = os.path.join(self.data_root,'video_'+self.unit+'.'+self.video_format)
        print("Generating video...")
        imageio.mimwrite(video_save_path, img_lst, fps=self.fps)
        print("Generating video done at ",video_save_path)

def main():
    sparse = Options()
    opt = sparse.opt
    print(opt)
    vg = VideoGenerator(opt)
    vg.gen_video()

if __name__=="__main__":
    main()