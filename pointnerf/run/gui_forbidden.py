# implement by LXY lixinyang

import math
import torch
import numpy as np
import dearpygui.dearpygui as dpg
from scipy.spatial.transform import Rotation as R
import pickle
import os
import random



device = 'cuda:0'

def convert(x, min_value, max_value, h, w):
    x = x.transpose(0, -1)[None]
    x = torch.nn.functional.interpolate(x, (w, h), mode='bilinear', align_corners=False)
    x = x.squeeze(0).transpose(0, -1).contiguous()
    return ((x - min_value) / (max_value - min_value)).detach().clamp(0, 1).cpu().numpy()

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
        # self.scalar = 1
        
        self.T = np.array([[1, 0, 0, 0],
                           [0, -1, 0, 0],
                           [0, 0, -1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

    def pose(self):
        return self.T
    
    def intrinsics(self):
        focal = self.H / (2 * np.tan(np.radians(self.fovy) / 2))
        return np.array([focal, focal, self.W // 2, self.H // 2])
    
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
    def __init__(self, system):
        self.W = 500
        self.H = 300
        self.cam = Camera(self.W, self.H, fovy=45)

        self.system = system 
        
        self.render_buffer = np.zeros((self.W, self.H, 3), dtype=np.float32)
        self.need_update = True # camera moved, should reset accumulation
        self.spp = 1 # sample per pixel
        self.mode = 'segments' # choose from ['image', 'depth', 'segments']
        self.inited = False

        self.dynamic_resolution = True
        self.downscale = 1

        dpg.create_context()
        self.register_dpg()
        self.test_step()

    def __del__(self):
        dpg.destroy_context()

    def prepare_scene(self, radiance_param):
        density_fn, all_fn = self.system.nerf_container(radiance_param)
        semantic_fn = self.system.network(all_fn, 1)

        self.density_fn = density_fn
        self.semantic_fn = semantic_fn
        self.rgb_fn = lambda x, d: self.system.nerf_container.rgb_fn(x, d, radiance_param)

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
            # self.test_step()
            dpg.render_dearpygui_frame()

if __name__ == '__main__':
    # import utils
    from main import NeRF2SemanticSystem
    from utils import import_str, label2color, seed_everything, get_args
    # import utils
    path = os.environ.get('CONFIG' ,'./configs/scannet_triplane.yaml')

    args = get_args(path)
    system = NeRF2SemanticSystem(args).to(device)

    with torch.no_grad():
        gui = NeRFGUI(system)
        gui.render()