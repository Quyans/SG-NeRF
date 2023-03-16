import sys
import os
import pathlib
sys.path.append(os.path.join(pathlib.Path(__file__).parent.absolute(), '..'))
import math
import torch
import numpy as np
import dearpygui.dearpygui as dpg
from scipy.spatial.transform import Rotation as R
import pickle

import random

from PIL import Image
from data import create_data_loader, create_dataset
from models import create_model
from utils.visualizer import Visualizer
from utils import format as fmt
import copy
import time
from options import TrainOptions
from utils import util
from render_vid import render_vid
from utils.visualizer import save_image
import imageio

import glob

import mitsuba
mitsuba.set_variant('scalar_rgb')
from mitsuba.core import ScalarTransform4f, AnimatedTransform
import numpy as np
# from frame import *

"""
可视化pointnerf的时候 将 semantic_guidance和 predict_semantic开关设为0 并且把 shading_feature_mlp_layer2_bpnet 设为0
SGNeRF 则相反

需要先copy path 然後save path 然後可以選擇genvid
然後渲染出視頻後可以在服務器使用 vlc xxx.mov 直接遠程播放視頻
"""


""" test poses
global_poses = [

[[-0.985777, -0.099129, -0.00645934, 1.22676],
 [-0.279312, 0.391849, -0.876607, 2.73202],
 [-0.0114557, -0.914236, -0.405019, 1.37772],
 [0, 0, 0, 1]],
[[-0.985773, -0.0991183, -0.00393758, 1.23166],
 [-0.281377, 0.391805, -0.875955, 2.7339],
 [-0.0124243, -0.914258, -0.404949, 1.378],
 [0, 0, 0, 1]],
[[-0.985764, -0.0991075, -0.00141895, 1.23656],
 [-0.283442, 0.391761, -0.875297, 2.73577],
 [-0.0133939, -0.91428, -0.404877, 1.37829],
 [0, 0, 0, 1]],
[[-0.985749, -0.0990965, 0.00109661, 1.24146],
 [-0.285508, 0.391717, -0.874633, 2.73765],
 [-0.0143645, -0.914302, -0.404801, 1.37857],
 [0, 0, 0, 1]],
[[-0.985729, -0.0990855, 0.003609, 1.24636],
 [-0.287576, 0.391672, -0.873963, 2.73952],
 [-0.0153361, -0.914323, -0.404722, 1.37885],
 [0, 0, 0, 1]],
]
global_poses = np.array(global_poses)
"""


def to8b(x): return (255*np.clip(x, 0, 1)).astype(np.uint8)

def read_image(filepath, dtype=None):
    image = np.asarray(Image.open(filepath))
    if dtype is not None and dtype==np.float32:
        image = (image / 255).astype(dtype)
    return image

def gen_video(output_dir, img_dir, name, steps):
    img_lst = []
    for i in range(steps):
        img_filepath = os.path.join(img_dir, '{}.png'.format(i))
        img_arry = read_image(img_filepath, dtype=np.float32)
        img_lst.append(img_arry)
    stacked_imgs = [to8b(img_arry) for img_arry in img_lst]
    filename = 'video_quality_{}.mov'.format( name)
    imageio.mimwrite(os.path.join(output_dir, filename), stacked_imgs, fps=10, quality=10)
    filename = 'video_{}.gif'.format( name)
    imageio.mimwrite(os.path.join(output_dir, filename), stacked_imgs, fps=5, format='GIF')

def convert(x, min_value, max_value, h, w):
    x = x.transpose(0, -1)[None]
    x = torch.nn.functional.interpolate(x, (w, h), mode='bilinear', align_corners=False)
    x = x.squeeze(0).transpose(0, -1).contiguous()
    # return ((x - min_value) / (max_value - min_value)).detach().clamp(0, 1).cpu().numpy()
    return x.numpy()

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

class Camera:
    def __init__(self, W, H, r=2, fovy=60):
        self.W = W
        self.H = H
        self.fovy = fovy # in degree
        self.up = np.array([0, 1, 0], dtype=np.float32)
        self.side = np.array([-1, 0, 0], dtype=np.float32)
        self.roll = np.array([0, 0, 1], dtype=np.float32)
        self.focal = 0
        # self.scalar = 1
        
        # self.T = np.array([[1, 0, 0, 0],
        #                    [0, -1, 0, 0],
        #                    [0, 0, -1, 0],
        #                    [0, 0, 0, 1]], dtype=np.float32)

        self.T = np.array([[-0.980738, -0.099445, -0.099445,  1.226757],
                           [-0.193549, 0.378740, -0.905039, 2.732021],
                           [0.026327, -0.920145, -0.390692, 1.377719],
                           [0, 0, 0, 1]], dtype=np.float32)

    def pose(self):
        return self.T
    

    def intrinsics(self,f=None):

        
        if f==None:
            # 变焦

            # np.radians 将角度转为弧度制 因为np.tan输入是弧度
            self.focal = self.H / (2 * np.tan(np.radians(self.fovy) / 2))
        else:
            # 固定焦距
            self.focal = f
        # self.dataset.focal = focal
        return np.array([self.focal, self.focal, self.W // 2, self.H // 2])

    
    
    def rotate_up_side(self, dx, dy):
        # rotate along camera up/side axis!
        rotvec_x = self.up * np.radians(-0.05 * dx)
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R.from_rotvec(rotvec_x).as_matrix()
        self.T = self.T @ T

        rotvec_y = self.side * np.radians(-0.05 * dy)
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R.from_rotvec(rotvec_y).as_matrix()
        self.T = self.T @ T
        
    def rotate_roll_left(self):
        rotvec_z = self.roll * np.radians(-1)
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R.from_rotvec(rotvec_z).as_matrix()
        self.T = self.T @ T
    
    def rotate_roll_right(self):
        rotvec_z = - self.roll * np.radians(-1)
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R.from_rotvec(rotvec_z).as_matrix()
        self.T = self.T @ T

    def move_forward(self):
        T = np.eye(4, dtype=np.float32)
        T[:3, 3] = self.roll * 0.1
        self.T = self.T @ T

    def move_backward(self):
        T = np.eye(4, dtype=np.float32)
        T[:3, 3] = - self.roll * 0.1
        self.T = self.T @ T

    def move_left(self):
        T = np.eye(4, dtype=np.float32)
        T[:3, 3] = self.side * 0.1
        self.T = self.T @ T

    def move_right(self):
        T = np.eye(4, dtype=np.float32)
        T[:3, 3] = - self.side * 0.1
        self.T = self.T @ T

    def scale(self, d):
        # print(d)
        self.fovy = self.fovy * (1 + d * 0.01) 

class NeRFGUI:
    def __init__(self, model,dataset,visualizer,opt):
        self.W = opt.img_wh[0]
        self.H = opt.img_wh[1]
        self.cam = Camera(self.W, self.H, fovy=45)

        self.model = model
        self.dataset = dataset
        self.visualizer = visualizer
        self.opt = opt

        self.render_buffer = np.zeros((self.W, self.H, 3), dtype=np.float32)
        self.need_update = True # camera moved, should reset accumulation
        self.render_rgblist = False #
        self.gen_vid = False # generate video
        self.spp = 1 # sample per pixel
        self.mode = 'image' # choose from ['image', 'depth', 'segments']
        self.inited = False

        self.dynamic_resolution = True
        self.downscale = 0.1

        self.save_camera_path = False
        # self.camera_path = global_poses
        self.camera_path = []
        self.set_cameras = [] #存放设置的camera

        dpg.create_context()
        self.register_dpg()
        self.teststep()


    def __del__(self):
        dpg.destroy_context()

    def prepare_scene(self, radiance_param):
        density_fn, all_fn = self.system.nerf_container(radiance_param)
        semantic_fn = self.system.network(all_fn, 1)

        self.density_fn = density_fn
        self.semantic_fn = semantic_fn
        self.rgb_fn = lambda x, d: self.system.nerf_container.rgb_fn(x, d, radiance_param)


    
    def teststep(self , test_steps=0, lpips=True):
        
        
        if self.gen_vid or self.render_rgblist:
            ckpt_dir = os.path.join(self.opt.checkpoints_dir, self.opt.ckpt_name,"gui") 
            camera_rgb_dir = os.path.join(ckpt_dir,"render_img")
            
            totalView = len([n for n in glob.glob(camera_rgb_dir + "/*.png") if os.path.isfile(n)])
            if self.render_rgblist:
                totalView = self.renderlist(camera_rgb_dir)
                print("==============================Finish render List==============================")
                self.render_rgblist = False

            if self.gen_vid:
                gen_video(ckpt_dir,camera_rgb_dir,"test",totalView)
                print("==============================Finish Generating Video==============================")
                self.gen_vid = False
        elif self.need_update:
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()

            model = self.model
            visualizer = self.visualizer
            opt = self.opt
           
            H = int(self.H * self.downscale)
            W = int(self.W * self.downscale)
            opt.img_wh = [W,H]

            # ori_img_shape = list(self.transform(img).shape)  # (4, h, w)
            # self.intrinsic[0, :] *= (self.width / ori_img_shape[2])
            # self.intrinsic[1, :] *= (self.height / ori_img_shape[1])

         
            if hasattr(opt,'focal'):
                # 指定相机焦距
                intrinsics = self.cam.intrinsics(opt.focal) * self.downscale 
                fx, fy,cx,cy = intrinsics
                intrinsics = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])
            else:
                intrinsics = self.cam.intrinsics() * self.downscale 
                fx, fy,cx,cy = intrinsics
                intrinsics = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])

            
            dataset = self.dataset
            dataset.img_wh = [W,H]
            # dataset.intrinsic[0, :] *= (W/ dataset.width)
            # dataset.intrinsic[1, :] *= (H/ dataset.height)
            # intrinsics = self.cam.intrinsics() * self.downscale 
            dataset.intrinsic = intrinsics
            dataset.focal = self.cam.focal
            dataset.height = H
            dataset.width = W
            
            total_num = dataset.total
            
            patch_size = opt.random_sample_size
            chunk_size = patch_size * patch_size
            
            

            # height = dataset.height
            # width = dataset.width
            visualizer.reset()
            count = 0
            pose = self.cam.pose()
            
            if self.save_camera_path:
                print("up pose:")
                self.camera_path.append(pose)
            
            data = dataset.gui_item(pose)
            raydir = data['raydir'].clone()
            pixel_idx = data['pixel_idx'].view(data['pixel_idx'].shape[0], -1, data['pixel_idx'].shape[3]).clone()
            edge_mask = torch.zeros([H, W], dtype=torch.bool)
            edge_mask[pixel_idx[0,...,1].to(torch.long), pixel_idx[0,...,0].to(torch.long)] = 1
            edge_mask=edge_mask.reshape(-1) > 0
            np_edge_mask=edge_mask.numpy().astype(bool)
            totalpixel = pixel_idx.shape[1]
            tmpgts = {}

            visuals = None
            stime = time.time()
            ray_masks = []
            for k in range(0, totalpixel, chunk_size):
                start = k
                end = min([k + chunk_size, totalpixel])
                data['raydir'] = raydir[:, start:end, :]
                data["pixel_idx"] = pixel_idx[:, start:end, :]
                model.set_input(data)

                model.test()
                curr_visuals = model.get_current_visuals(data=data)
                chunk_pixel_id = data["pixel_idx"].cpu().numpy().astype(np.int32)
                if visuals is None:
                    visuals = {}
                    for key, value in curr_visuals.items():
                        if value is None or key=="gt_image":
                            continue
                        chunk = value.cpu().numpy()
                        visuals[key] = np.zeros((H, W, 3)).astype(chunk.dtype)
                        visuals[key][chunk_pixel_id[0,...,1], chunk_pixel_id[0,...,0], :] = chunk
                else:
                    for key, value in curr_visuals.items():
                        if value is None or key=="gt_image":
                            continue
                        visuals[key][chunk_pixel_id[0,...,1], chunk_pixel_id[0,...,0], :] = value.cpu().numpy()
                if "ray_mask" in model.output and "ray_masked_coarse_raycolor" in opt.test_color_loss_items:
                    ray_masks.append(model.output["ray_mask"] > 0)
            if len(ray_masks) > 0:
                ray_masks = torch.cat(ray_masks, dim=1)

            ender.record()
            torch.cuda.synchronize()
            t = starter.elapsed_time(ender)

            if 'ray_masked_coarse_raycolor' in model.visual_names:
                visuals['ray_masked_coarse_raycolor'] = np.copy(visuals["coarse_raycolor"]).reshape(H, W, 3)
                print(visuals['ray_masked_coarse_raycolor'].shape, ray_masks.cpu().numpy().shape)
                visuals['ray_masked_coarse_raycolor'][ray_masks.view(H, W).cpu().numpy() <= 0,:] = 0.0
            if 'ray_depth_masked_coarse_raycolor' in model.visual_names:
                visuals['ray_depth_masked_coarse_raycolor'] = np.copy(visuals["coarse_raycolor"]).reshape(H, W, 3)
                visuals['ray_depth_masked_coarse_raycolor'][model.output["ray_depth_mask"][0].cpu().numpy() <= 0] = 0.0

            img_color  = torch.from_numpy(np.array(np.copy(visuals["coarse_raycolor"])))
            img_color = convert(img_color, 0, 1, self.H, self.W)
            
            # update dynamic resolution
            if self.dynamic_resolution:
                # max allowed infer time per-frame is 200 ms
                full_t = t / (self.downscale ** 2)
                downscale = min(1, max(1/10, math.sqrt(20 / full_t)))
                # if downscale > self.downscale * 1.2 or downscale < self.downscale * 0.8:
                self.downscale = downscale

            if self.need_update:
                    self.render_buffer = img_color
                    self.spp = 1
                    self.need_update = False
            else:
                self.render_buffer = (self.render_buffer * self.spp + img_color) / (self.spp + 1)
                self.spp += 1

            dpg.set_value("_log_infer_time", f'{t:.4f}ms ({int(1000/t)} FPS)')
            dpg.set_value("_log_resolution", f'{int(self.downscale * self.W)}x{int(self.downscale * self.H)}')
            dpg.set_value("_log_spp", self.spp)
            dpg.set_value("_texture", self.render_buffer)
            
            for key, value in visuals.items():
                if key in opt.visual_items:
                    visualizer.print_details("{}:{}".format(key, visuals[key].shape))
                    visuals[key] = visuals[key].reshape(H, W, 3)


            # print(" time used: {} s".format(time.time() - stime), " at ", visualizer.image_dir)
            # visualizer.display_current_results(visuals, 0, opt=opt)

    def renderlist(self,camera_rgb_dir):
        util.mkdir(camera_rgb_dir)

        model = self.model
        print(model.device)
        visualizer = copy.deepcopy(self.visualizer)
        opt = copy.deepcopy(self.opt)
    
        H = int(self.H)
        W = int(self.W)
        opt.img_wh = [W,H]

        # ori_img_shape = list(self.transform(img).shape)  # (4, h, w)
        # self.intrinsic[0, :] *= (self.width / ori_img_shape[2])
        # self.intrinsic[1, :] *= (self.height / ori_img_shape[1])

    
        if hasattr(opt,'focal'):
            # 指定相机焦距
            intrinsics = self.cam.intrinsics(opt.focal)
            fx, fy,cx,cy = intrinsics
            intrinsics = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])
        else:
            intrinsics = self.cam.intrinsics()
            fx, fy,cx,cy = intrinsics
            intrinsics = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])

        
        dataset = copy.deepcopy(self.dataset)
        dataset.img_wh = [W,H]
        # dataset.intrinsic[0, :] *= (W/ dataset.width)
        # dataset.intrinsic[1, :] *= (H/ dataset.height)
        # intrinsics = self.cam.intrinsics() * self.downscale 
        dataset.intrinsic = intrinsics
        dataset.focal = self.cam.focal
        dataset.height = H
        dataset.width = W

        patch_size = opt.random_sample_size
        chunk_size = patch_size * patch_size

        visualizer.reset()

        count = 0
        # pose_list = []
        pose_list = self.camera_path
  

        for i in range(len(pose_list)):
            pose = pose_list[i]
            print("pose:!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(pose)
            data = dataset.gui_item(pose)
            raydir = data['raydir'].clone()
            pixel_idx = data['pixel_idx'].view(data['pixel_idx'].shape[0], -1, data['pixel_idx'].shape[3]).clone()
            edge_mask = torch.zeros([H, W], dtype=torch.bool)
            edge_mask[pixel_idx[0,...,1].to(torch.long), pixel_idx[0,...,0].to(torch.long)] = 1
            edge_mask=edge_mask.reshape(-1) > 0
            np_edge_mask=edge_mask.numpy().astype(bool)
            totalpixel = pixel_idx.shape[1]

            visuals = None
            stime = time.time()
            ray_masks = []
            for k in range(0, totalpixel, chunk_size):
                start = k
                end = min([k + chunk_size, totalpixel])
                data['raydir'] = raydir[:, start:end, :]
                data["pixel_idx"] = pixel_idx[:, start:end, :]
                model.set_input(data)
                model.test()
                curr_visuals = model.get_current_visuals(data=data)
                chunk_pixel_id = data["pixel_idx"].cpu().numpy().astype(np.int32)
                if visuals is None:
                    visuals = {}
                    for key, value in curr_visuals.items():
                        if value is None or key=="gt_image":
                            continue
                        chunk = value.cpu().numpy()
                        visuals[key] = np.zeros((H, W, 3)).astype(chunk.dtype)
                        visuals[key][chunk_pixel_id[0,...,1], chunk_pixel_id[0,...,0], :] = chunk
                else:
                    for key, value in curr_visuals.items():
                        if value is None or key=="gt_image":
                            continue
                        visuals[key][chunk_pixel_id[0,...,1], chunk_pixel_id[0,...,0], :] = value.cpu().numpy()
                if "ray_mask" in model.output and "ray_masked_coarse_raycolor" in opt.test_color_loss_items:
                    ray_masks.append(model.output["ray_mask"] > 0)
            if len(ray_masks) > 0:
                ray_masks = torch.cat(ray_masks, dim=1)
            torch.cuda.synchronize()

            if 'ray_masked_coarse_raycolor' in model.visual_names:
                visuals['ray_masked_coarse_raycolor'] = np.copy(visuals["coarse_raycolor"]).reshape(H, W, 3)
                print(visuals['ray_masked_coarse_raycolor'].shape, ray_masks.cpu().numpy().shape)
                visuals['ray_masked_coarse_raycolor'][ray_masks.view(H, W).cpu().numpy() <= 0,:] = 0.0
            if 'ray_depth_masked_coarse_raycolor' in model.visual_names:
                visuals['ray_depth_masked_coarse_raycolor'] = np.copy(visuals["coarse_raycolor"]).reshape(H, W, 3)
                visuals['ray_depth_masked_coarse_raycolor'][model.output["ray_depth_mask"][0].cpu().numpy() <= 0] = 0.0

            img_color  = torch.from_numpy(np.array(np.copy(visuals["coarse_raycolor"])))
            img_color = convert(img_color, 0, 1, self.H, self.W)
            save_image(img_color,os.path.join(camera_rgb_dir,"{}.png".format(i)))
            print("genvid, num.{} in {}  time used: {} s".format(i,len(pose_list), time.time() - stime), " at ", camera_rgb_dir)
        
        totalView = len(self.camera_path)
        self.camera_path = []
        return totalView
        
    def register_dpg(self):

        ### register texture 

        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(self.W, self.H, self.render_buffer, format=dpg.mvFormat_Float_rgb, tag="_texture")

        ### register window

        # the rendered image, as the primary window
        with dpg.window(tag="_primary_window", width=self.W, height=self.H):

            # add the texture
            dpg.add_image("_texture")

        dpg.set_primary_window("_primary_window", True)

        # control window
        with dpg.window(label="Control", tag="_control_window", width=200, height=300):

            # button theme
            with dpg.theme() as theme_button:
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (23, 3, 18))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (51, 3, 47))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (83, 18, 83))
                    dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                    dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)

            # time             
            with dpg.group(horizontal=True):
                dpg.add_text("Infer time: ")
                dpg.add_text("no data", tag="_log_infer_time")
            
            with dpg.group(horizontal=True):
                dpg.add_text("SPP: ")
                dpg.add_text("1", tag="_log_spp")
            
            # rendering options
            with dpg.collapsing_header(label="Options", default_open=True):

                # dynamic rendering resolution
                with dpg.group(horizontal=True):

                    def callback_set_dynamic_resolution(sender, app_data):
                        if self.dynamic_resolution:
                            self.dynamic_resolution = False
                            self.downscale = 1
                        else:
                            self.dynamic_resolution = True
                        self.need_update = True

                    dpg.add_checkbox(label="dynamic resolution", default_value=self.dynamic_resolution, callback=callback_set_dynamic_resolution)
                    dpg.add_text(f"{self.W}x{self.H}", tag="_log_resolution")

                # mode combo
                def callback_change_mode(sender, app_data):
                    self.mode = app_data
                    self.need_update = True
                
                dpg.add_combo(('image', 'segments'), label='mode', default_value='segments', callback=callback_change_mode)
                
                def callback_open_checkpoint(sender, app_data):
                    # print('OK was clicked.')
                    # print("Sender: ", sender)
                    print("App Data: ", app_data)
                    try:
                        # pass
                        # print(torch.load(app_data['file_path_name'] + '.ckpt'))
                        self.system = NeRF2SemanticSystem.load_from_checkpoint(app_data['file_path_name'] + '.ckpt').to(device)
                        print(list(self.system.network.decoder.parameters()))
                    except:
                        print('open checkpoint failed.')
                
                def callback_open_scene(sender, app_data):
                    print("App Data: ", app_data)
                    if True:
                    # try:
                        params = torch.load(app_data['file_path_name'] + '.pth', map_location=device)['model']
                        # print(params)
                        for ignored_key in ['aabb_train', 'aabb_infer', 'density_bitfield', 'step_counter']:
                            del params[ignored_key]
                        for key in params.keys():
                            params[key] = params[key][None]
                        self.prepare_scene(params)
                        self.inited = True
                    # except:
                    #     print('open scene failed.')

                dpg.add_file_dialog(
                        directory_selector=False, show=False, callback=callback_open_checkpoint, tag="checkpoint", default_path='/home/huangchi/proj/Project-A/logs/main/lightning_logs/version_10/checkpoints/', default_filename='epoch=529-step=99640.ckpt')

                dpg.add_file_dialog(
                        directory_selector=False, show=False, callback=callback_open_scene, tag="scene", default_path='/data/hc/ScanNet_NERF_dataset/val/scene0019_00/', default_filename='ngp.pth')

                #TODO  完成checkpoint的替换
                # with dpg.group(horizontal=True):
                #     dpg.add_button(label="open checkpoint", callback=lambda: dpg.show_item("checkpoint"))
                #     dpg.add_button(label="open scene", callback=lambda: dpg.show_item("scene"))
                
                
                def add_camera_callback(sender):
                    pose = self.cam.pose()
                    print(pose)
                    self.set_cameras.append(pose)
                    # self.camera_path
                    
                def generate_path_callback(sender):
                    # for pose in self.camera_path:
                    #     print(pose)
                    c2w = self.set_cameras

                    fps=60
                    time_step = 0.5
                    ani=AnimatedTransform()
                    time_array = []
                    print(len(c2w))
                    print(c2w[0])

                    
                    for i in range(0,len(c2w)):
                        print("i is :",i)
                        time_array.append(time_step*i)
                        ani.append(time_array[i],ScalarTransform4f(c2w[i]))
                    
                    # 清空原本的camera_path
                    self.camera_path = []

                    for t in np.arange(time_array[0],time_array[-1],step=1./fps):
                        self.camera_path.append(ani.eval(t).matrix.numpy()) #转为numpy对象
                        # print(ani.eval(t))
                    
                    save_path()
                    print("============finush generating path,total {} setting cameras, generate to {} poses".format(len(c2w),len(self.camera_path)))


                with dpg.group(horizontal=True):
                    dpg.add_button(label="add camera", callback=add_camera_callback)
                    dpg.add_button(label="generate path", callback=generate_path_callback)
                
                def copy_path_callback(sender):
                    self.save_camera_path = True
                    print(self.cam.pose())
                    # print("pose,list",self.camera_path)

                def stop_copy_callback(sender):
                    self.save_camera_path = False


                def save_path():
                    # todo 兩種格式輸出pose
                    ckpt_dir = os.path.join(self.opt.checkpoints_dir, self.opt.ckpt_name,"gui") 
                    camera_path_dir = os.path.join(ckpt_dir,"render_path")
                    util.mkdir(camera_path_dir)
                    print(ckpt_dir)
                    
                    for i in range(len(self.camera_path)):
                        pose = self.camera_path[i]
                        print("save！！")
                        print(pose)
                        print(pose.shape)

                        np.savetxt(os.path.join(camera_path_dir,'{}.txt'.format(i)), pose, fmt='%f', delimiter=' ')

                def save_path_callback(sender):
                    save_path()
                    
                    
                    # reset_path()

                def reset_path():
                    self.save_camera_path = False
                    self.camera_path = []
                    print("succeed reset camera poses:", self.camera_path)

                def reset_path_callback(sender):
                    reset_path()
                
                with dpg.group(horizontal=True):
                    dpg.add_button(label="copy path", callback=copy_path_callback)
                    dpg.add_button(label="pause", callback=stop_copy_callback)
                    dpg.add_button(label="save", callback=save_path_callback)
                    dpg.add_button(label="reset", callback=reset_path_callback)
                
                def render_rgb_callback(sender):
                    self.render_rgblist = True
                def gen_vid_callback(sender):
                    #todo 
                    # pycuda + 多線程 會爆cuFuncSetBlockShape failed: invalid resource handle
                    #  例子https://www.oomake.com/question/1486019

                    # self.renderlist()
                    self.gen_vid = True
                    print(1)

                with dpg.group(horizontal=True):
                    dpg.add_button(label="renderRGB", callback=render_rgb_callback)            
                    dpg.add_button(label="genvid", callback=gen_vid_callback)
                

        ### register camera handler

        def callback_camera_drag_rotate(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.rotate_up_side(dx, dy)
            self.need_update = True

        def callback_key_press(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                return

            funcs = {
                    87: self.cam.move_forward, # w
                    65: self.cam.move_left, # a
                    83: self.cam.move_backward, # s
                    68: self.cam.move_right, # d
                    81: self.cam.rotate_roll_left, # q
                    69: self.cam.rotate_roll_right, # e
                }
            if app_data in funcs.keys():
                # print(app_data)
                funcs[app_data]()

                self.need_update = True

        def callback_mouse_wheel(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            # print(app_data)

            delta = app_data

            self.cam.scale(delta)
            self.need_update = True

        with dpg.handler_registry():
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Left, callback=callback_camera_drag_rotate)

            dpg.add_mouse_wheel_handler(callback=callback_mouse_wheel)
            dpg.add_key_press_handler(callback=callback_key_press)

        dpg.create_viewport(title='Project-A', width=self.W, height=self.H, resizable=False)



        ### global theme
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core)
        
        dpg.bind_item_theme("_primary_window", theme_no_padding)

        dpg.setup_dearpygui()
        dpg.show_viewport()

    def render(self):

        while dpg.is_dearpygui_running():
            # update texture every frame
            self.teststep()
            dpg.render_dearpygui_frame()


def get_latest_epoch(resume_dir):
    os.makedirs(resume_dir, exist_ok=True)
    str_epoch = [file.split("_")[0] for file in os.listdir(resume_dir) if file.endswith("_states.pth")]
    int_epoch = [int(i) for i in str_epoch]
    return None if len(int_epoch) == 0 else str_epoch[int_epoch.index(max(int_epoch))]


def main():

    # c2w = [
    #     [[1,2,3,2],
    #     [1,2,2,2],
    #     [1,2,1,1],
    #     [0,0,0,1]],

    #     [[1,0,1,0],
    #     [1,2,2,5],
    #     [1,3,1,4],
    #     [0,0,0,1]]
        
    # ]

    # fps=60
    # time_step = 0.5
    # ani=AnimatedTransform()
    # time_array = []
    # print(len(c2w))
    # print(c2w[0])

    
    # for i in range(0,len(c2w)):
    #     print("i is :",i)
    #     time_array.append(time_step*i)
    #     ani.append(time_array[i],ScalarTransform4f(c2w[i]))
    
    # # 清空原本的camera_path
    # camera_path = []

    # for t in np.arange(time_array[0],time_array[-1],step=1./fps):
    #     camera_path.append(ani.eval(t))
    #     # print(ani.eval(t))


    torch.backends.cudnn.benchmark = True

    opt = TrainOptions().parse()
    cur_device = torch.device('cuda:{}'.format(opt.gpu_ids[0]) if opt.
                              gpu_ids else torch.device('cpu'))
    print("opt.color_loss_items ", opt.color_loss_items)

    if opt.debug:
        torch.autograd.set_detect_anomaly(True)
        print(fmt.RED +
              '++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('Debug Mode')
        print(
            '++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++' +
            fmt.END)
    visualizer = Visualizer(opt)
    train_dataset = create_dataset(opt)
    # 这里有影响
    # opt.img_wh=[32,24]
    img_lst=None
    with torch.no_grad():
        print(opt.checkpoints_dir + opt.name + "/*_net_ray_marching.pth")

        resume_dir = os.path.join(opt.checkpoints_dir, opt.name)
        if opt.resume_iter == "best":
            opt.resume_iter = "latest"
        resume_iter = opt.resume_iter if opt.resume_iter != "latest" else get_latest_epoch(resume_dir)
        if resume_iter is None:
            visualizer.print_details("No previous checkpoints at iter {} !!", resume_iter)
            exit()
        else:
            opt.resume_iter = resume_iter
            visualizer.print_details('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            visualizer.print_details('test at {} iters'.format(opt.resume_iter))
            visualizer.print_details(f"Iter: {resume_iter}")
            visualizer.print_details('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        opt.mode = 2
        opt.load_points=1
        opt.resume_dir=resume_dir
        opt.resume_iter = resume_iter
        opt.is_train=True

    model = create_model(opt)
    model.setup(opt, train_len=len(train_dataset))

    # create test loader
    test_opt = copy.deepcopy(opt)
    test_opt.is_train = False
    test_opt.random_sample = 'no_crop'
    test_opt.random_sample_size = min(48, opt.random_sample_size)
    test_opt.batch_size = 1
    test_opt.n_threads = 0
    test_opt.prob = 0
    test_opt.split = "test"
    test_opt.fps=60
    visualizer.reset()

    fg_masks = None
    test_bg_info = None
        # if opt.vid > 0:
        #     render_dataset = create_render_dataset(test_opt, opt, resume_iter, test_num_step=opt.test_num_step)
    ############ initial test ###############
    with torch.no_grad():
        test_opt.nerf_splits = ["test"]
        test_opt.split = "test"
        test_opt.ckpt_name = opt.name
        test_opt.name = opt.name + "/test_{}".format(resume_iter)
        test_opt.test_num_step = opt.test_num_step
        
        test_dataset = create_dataset(test_opt)
        model.opt.is_train = 0
        model.opt.no_loss = 1
        # test_opt.img_wh=[32,24]
        model.eval()

        gui = NeRFGUI(model=model,visualizer=visualizer,dataset=test_dataset,opt=test_opt)
        gui.render()

        # test(model, test_dataset, test_opt, test_bg_info, test_steps=resume_iter)


if __name__ == '__main__':
    main()
       