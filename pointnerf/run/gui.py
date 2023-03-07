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

"""
可视化pointnerf的时候 将 semantic_guidance和 predict_semantic开关设为0 并且把 shading_feature_mlp_layer2_bpnet 设为0
SGNeRF 则相反

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
    filename = 'video_{}.mov'.format( name)
    imageio.mimwrite(os.path.join(output_dir, filename), stacked_imgs, fps=10, quality=10)
    filename = 'video_{}.gif'.format( name)
    imageio.mimwrite(os.path.join(output_dir, filename), stacked_imgs, fps=5, format='GIF')

def convert(x, min_value, max_value, h, w):
    x = x.transpose(0, -1)[None]
    x = torch.nn.functional.interpolate(x, (w, h), mode='bilinear', align_corners=False)
    x = x.squeeze(0).transpose(0, -1).contiguous()
    # return ((x - min_value) / (max_value - min_value)).detach().clamp(0, 1).cpu().numpy()
    return x.numpy()

def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    return torch.meshgrid(*args)

def get_rays(poses, intrinsics, H, W):
    ''' get rays
    Args:
        poses: [B, 4, 4], cam2world
        intrinsics: [4]
        H, W, N: int
        error_map: [B, 128 * 128], sample probability based on training error
    Returns:
        rays_o, rays_d: [B, N, 3]
        inds: [B, N]
    '''

    B = poses.shape[0]
    fx, fy, cx, cy = intrinsics

    i, j = custom_meshgrid(torch.linspace(0, W-1, W, device=device), torch.linspace(0, H-1, H, device=device)) # float
    i = i.t().reshape([1, H*W]).expand([B, H*W]) + 0.5
    j = j.t().reshape([1, H*W]).expand([B, H*W]) + 0.5

    zs = torch.ones_like(i)
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    directions = torch.stack((xs, ys, zs), dim=-1)
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)
    rays_d = directions @ poses[:, :3, :3].transpose(-1, -2) # (B, N, 3)

    rays_o = poses[..., :3, 3] # [B, 3]
    rays_o = rays_o[..., None, :].expand_as(rays_d) # [B, N, 3]

    return rays_o, rays_d

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

        # # 第一张的位姿
        # -0.980738 -0.099445-0.099445 1.226757
        # -0.193549 0.378740 -0.905039 2.732021
        # 0.026327 -0.920145 -0.390692 1.377719
        # 0.000000 0.000000 0.000000 1.000000

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
        self.gen_vid = False # generate video
        self.spp = 1 # sample per pixel
        self.mode = 'image' # choose from ['image', 'depth', 'segments']
        self.inited = False

        self.dynamic_resolution = True
        self.downscale = 0.1

        self.save_camera_path = False
        self.camera_path = []

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

    """
    # 用teststep方法，此方法废弃
    def test_step(self):
        if self.need_update and self.inited:
        
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()
            H = int(self.H * self.downscale)
            W = int(self.W * self.downscale)
            intrinsics = self.cam.intrinsics() * self.downscale 
            # print(self.cam.pose())
            pose = torch.from_numpy(self.cam.pose()).unsqueeze(0).to(device)

            rays_o, rays_d = get_rays(pose, intrinsics, H, W)

            rays_o, rays_d = rays_o.view(1, 1, -1, 3), rays_d.view(1, 1, -1, 3)

            # from render.simple_render import render
            density_fn = self.density_fn

            ray_size = rays_o.shape[2]
            chunk_size = 128 * 128 * 4

            if self.mode == 'image':
                feature_fn = self.rgb_fn
                imgs = []
                i = 0
                while i < ray_size:
                    chunk_size = (ray_size - i) if (i + chunk_size > ray_size) else chunk_size
                        
                    img = self.system.render(rays_o[:, :, i:i+chunk_size], rays_d[:, :, i:i+chunk_size], density_fn=density_fn, feature_fn=feature_fn, feature_fn_need_rays_d=True, **self.system.render_args, perturb=False, return_depth=False)
                        
                    imgs.append(img)
                    i += chunk_size

                if i == 0:
                    imgs = imgs[0]
                else:
                    imgs = torch.cat(imgs, dim=2)
                    
                img = imgs.reshape(H, W, 3)
                img = convert(img, 0, 1, self.H, self.W)
            elif self.mode == 'segments':
                feature_fn = self.semantic_fn
                imgs = []
                i = 0
                while i < ray_size:
                    chunk_size = (ray_size - i) if (i + chunk_size > ray_size) else chunk_size
                        
                    img = self.system.render(rays_o[:, :, i:i+chunk_size], rays_d[:, :, i:i+chunk_size], density_fn=density_fn, feature_fn=feature_fn, **self.system.render_args, perturb=False, return_depth=False)
                        
                    imgs.append(img)
                    i += chunk_size

                if i == 0:
                    imgs = imgs[0]
                else:
                    imgs = torch.cat(imgs, dim=2)

                imgs = label2color(imgs.argmax(-1))
                img = imgs.reshape(H, W, 3)
                
                img = convert(img, 0, 1, self.H, self.W)
                # elif self.mode == 'depth':
                #     depth = torch.zeros(B, H * W, device=device)
                #     depth[ids] = packed_sum(vw.view(-1) * rsts['t'].view(-1), hit_ray_pack_infos)
                #     depth = depth + (1 - vw_sum) * rsts['t'].max()
                #     depth = depth.reshape(H, W, 1)

                #     return convert(depth.repeat(1, 1, 3), depth.max(), 0, self.H, self.W)
                # elif self.mode == 'nablas':
                #     nablas = torch.zeros(B, H * W, 3, device=device)
                #     nablas[ids] = packed_sum(vw.view(-1,1) * rsts['nablas'].view(-1,3), hit_ray_pack_infos)
                #     nablas = nablas.reshape(H, W, 3)
                #     return convert(nablas, -1, 1, self.H, self.W)
                    

            ender.record()
            torch.cuda.synchronize()
            t = starter.elapsed_time(ender)

            # update dynamic resolution
            if self.dynamic_resolution:
                # max allowed infer time per-frame is 200 ms
                full_t = t / (self.downscale ** 2)
                downscale = min(1, max(1/4, math.sqrt(100 / full_t)))
                if downscale > self.downscale * 1.2 or downscale < self.downscale * 0.8:
                    self.downscale = downscale

            if self.need_update:
                self.render_buffer = img
                self.spp = 1
                self.need_update = False
            else:
                self.render_buffer = (self.render_buffer * self.spp + img) / (self.spp + 1)
                self.spp += 1

            dpg.set_value("_log_infer_time", f'{t:.4f}ms ({int(1000/t)} FPS)')
            dpg.set_value("_log_resolution", f'{int(self.downscale * self.W)}x{int(self.downscale * self.H)}')
            dpg.set_value("_log_spp", self.spp)
            dpg.set_value("_texture", self.render_buffer)
"""
    
    def teststep(self , test_steps=0, lpips=True):
        
        # if self.need_update and self.inited:
        if self.gen_vid:

            ckpt_dir = os.path.join(self.opt.checkpoints_dir, self.opt.ckpt_name,"gui") 
            camera_rgb_dir = os.path.join(ckpt_dir,"render_img")
            totalView = self.renderlist(camera_rgb_dir)

            
            gen_video(ckpt_dir,camera_rgb_dir,"test",totalView)
            self.gen_vid = False
        elif self.need_update and True:
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

                # if opt.bgmodel.endswith("plane"):
                #     img_lst, c2ws_lst, w2cs_lst, intrinsics_all, HDWD_lst, fg_masks, bg_ray_lst = bg_info
                #     if len(bg_ray_lst) > 0:
                #         bg_ray_all = bg_ray_lst[data["id"]]
                #         bg_idx = data["pixel_idx"].view(-1,2)
                #         bg_ray = bg_ray_all[:, bg_idx[:,1].long(), bg_idx[:,0].long(), :]
                #     else:
                #         xyz_world_sect_plane = mvs_utils.gen_bg_points(data)
                #         bg_ray, _ = model.set_bg(xyz_world_sect_plane, img_lst, c2ws_lst, w2cs_lst, intrinsics_all, HDWD_lst, data["plane_color"], fg_masks=fg_masks, vis=visualizer)
                #     data["bg_ray"] = bg_ray

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
        pose_list = self.camera_path
        # pose_list = [
        #     np.array([[-0.980738, -0.099445, -0.099445,  1.226757],
        #         [-0.193549, 0.378740, -0.905039, 2.732021],
        #         [0.026327, -0.920145, -0.390692, 1.377719],
        #         [0, 0, 0, 1]], dtype=np.float32),

        #     np.array([[-0.9, -0.09, -0.099445,  1.226757],
        #     [-0.193549, 0.378740, -0.905039, 1.732021],
        #     [0.026327, -0.920145, -0.390692, 1.577719],
        #     [0, 0, 0, 1]], dtype=np.float32)
        # ]

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


                with dpg.group(horizontal=True):

                    dpg.add_button(label="open checkpoint", callback=lambda: dpg.show_item("checkpoint"))

                    dpg.add_button(label="open scene", callback=lambda: dpg.show_item("scene"))
                
                
                def copy_path_callback(sender):
                    self.save_camera_path = True
                    print(self.cam.pose())
                    # print("pose,list",self.camera_path)

                def stop_copy_callback(sender):
                    self.save_camera_path = False

                def save_path_callback(sender):
                    # todo 兩種格式輸出pose

                    ckpt_dir = os.path.join(self.opt.checkpoints_dir, self.opt.ckpt_name,"gui") 
                    camera_path_dir = os.path.join(ckpt_dir,"render_path")
                    util.mkdir(camera_path_dir)
                    print(ckpt_dir)
                    
                    for i in range(len(self.camera_path)):
                        pose = self.camera_path[i]
                        np.savetxt(os.path.join(camera_path_dir,'{}.txt'.format(i)), pose, fmt='%f', delimiter=' ')
                    
                    # reset_path()

                def reset_path():
                    self.save_camera_path = False
                    self.camera_path = []
                    print("reset camera poses:", self.camera_path)

                def reset_path_callback(sender):
                    print("123")
                    reset_path()
                
                with dpg.group(horizontal=True):
                    dpg.add_button(label="copy path", callback=copy_path_callback)
                    dpg.add_button(label="pause", callback=stop_copy_callback)
                    dpg.add_button(label="save", callback=save_path_callback)
                    dpg.add_button(label="reset", callback=reset_path_callback)
                


                def gen_vid_callback(sender):
                    

                    #todo 
                    # pycuda + 多線程 會爆cuFuncSetBlockShape failed: invalid resource handle
                    #  例子https://www.oomake.com/question/1486019

                    # self.renderlist()
                    self.gen_vid = True
                    print(1)
                    
                
                dpg.add_button(label="genvid", callback=gen_vid_callback)

                # def callback_reset_object(sender, app_data):
                #     self.reset_object()
                #     self.need_update = True

                # dpg.add_button(label='reset_object', callback=callback_reset_object)

                # fov slider
                # def callback_set_fovy(sender, app_data):
                #     self.cam.fovy = app_data
                #     self.need_update = True

                # dpg.add_slider_int(label="FoV (vertical)", min_value=1, max_value=120, format="%d deg", default_value=self.cam.fovy, callback=callback_set_fovy)

                # # max_steps slider
                # def callback_set_max_steps(sender, app_data):
                #     self.opt.max_steps = app_data
                #     self.need_update = True

                # dpg.add_slider_int(label="max steps", min_value=1, max_value=1024, format="%d", default_value=self.opt.max_steps, callback=callback_set_max_steps)

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
    # # import utils
    # from main import NeRF2SemanticSystem
    # from utils import import_str, label2color, seed_everything, safe_log, get_args
    # # import utils
    # path = os.environ.get('CONFIG' ,'./configs/scannet_triplane.yaml')

    # args = get_args(path)
    # system = NeRF2SemanticSystem(args).to(device)

    # with torch.no_grad():
    #     gui = NeRFGUI(system)

    main()
       