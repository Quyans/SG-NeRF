import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from ..helpers.networks import init_seq, positional_encoding
from utils.spherical import SphericalHarm_table as SphericalHarm
from ..helpers.geometrics import compute_world2local_dist




class PointAggregator(torch.nn.Module):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument(
            '--feature_init_method',
            type=str,
            default="rand",
            help='which agg model to use [feature_interp | graphconv | affine_mix]')

        parser.add_argument(
            '--which_agg_model',
            type=str,
            default="viewmlp",
            help='which agg model to use [viewmlp | nsvfmlp]')

        parser.add_argument(
            '--agg_distance_kernel',
            type=str,
            default="quadric",
            help='which agg model to use [quadric | linear | feat_intrp | harmonic_intrp]')

        parser.add_argument(
            '--sh_degree',
            type=int,
            default=4,
            help='degree of harmonics')

        parser.add_argument(
            '--sh_dist_func',
            type=str,
            default="sh_quadric",
            help='sh_quadric | sh_linear | passfunc')

        parser.add_argument(
            '--sh_act',
            type=str,
            default="sigmoid",
            help='sigmoid | tanh | passfunc')

        parser.add_argument(
            '--agg_axis_weight',
            type=float,
            nargs='+',
            default=None,
            help=
            '(1., 1., 1.)'
        )

        parser.add_argument(
            '--agg_dist_pers',
            type=int,
            default=1,
            help='use pers dist')

        parser.add_argument(
            '--apply_pnt_mask',
            type=int,
            default=1,
            help='use pers dist')

        parser.add_argument(
            '--modulator_concat',
            type=int,
            default=0,
            help='use pers dist')

        parser.add_argument(
            '--agg_intrp_order',
            type=int,
            default=0,
            help='interpolate first and feature mlp 0 | feature mlp then interpolate 1 | feature mlp color then interpolate 2')

        parser.add_argument(
            '--shading_feature_mlp_layer0',
            type=int,
            default=0,
            help='interp to agged features mlp num')

        parser.add_argument(
            '--shading_feature_mlp_layer1',
            type=int,
            default=2,
            help='interp to agged features mlp num')

        parser.add_argument(
            '--shading_feature_mlp_layer2',
            type=int,
            default=0,
            help='interp to agged features mlp num')

        parser.add_argument(
            '--shading_feature_mlp_layer3',
            type=int,
            default=0,
            help='interp to agged features mlp num')
        parser.add_argument(
            '--shading_feature_mlp_layer4',
            type=int,
            default=1,
            help='interp to agged features mlp num')
        parser.add_argument(
            '--shading_feature_mlp_linear',
            type=int,
            default=0,
            help='interp to agged features mlp num')
        parser.add_argument(
            '--shading_feature_num',
            type=int,
            default=256,
            help='agged shading feature channel num')

        parser.add_argument(
            '--point_hyper_dim',
            type=int,
            default=256,
            help='agged shading feature channel num')

        parser.add_argument(
            '--shading_alpha_mlp_layer',
            type=int,
            default=1,
            help='agged features to alpha mlp num')

        parser.add_argument(
            '--shading_color_mlp_layer',
            type=int,
            default=1,
            help='agged features to alpha mlp num')

        parser.add_argument(
            '--shading_color_channel_num',
            type=int,
            default=3,
            help='color channel num')

        parser.add_argument(
            '--num_feat_freqs',
            type=int,
            default=0,
            help='color channel num')

        parser.add_argument(
            '--num_hyperfeat_freqs',
            type=int,
            default=0,
            help='color channel num')

        parser.add_argument(
            '--dist_xyz_freq',
            type=int,
            default=2,
            help='color channel num')

        parser.add_argument(
            '--dist_xyz_deno',
            type=float,
            default=0,
            help='color channel num')

        parser.add_argument(
            '--weight_xyz_freq',
            type=int,
            default=2,
            help='color channel num')

        parser.add_argument(
            '--weight_feat_dim',
            type=int,
            default=8,
            help='color channel num')

        parser.add_argument(
            '--agg_weight_norm',
            type=int,
            default=1,
            help='normalize weight, sum as 1')

        parser.add_argument(
            '--view_ori',
            type=int,
            default=0,
            help='0 for pe+3 orignal channels')

        parser.add_argument(
            '--agg_feat_xyz_mode',
            type=str,
            default="None",
                help='which agg xyz mode to use [None not to use | world world xyz | pers perspective xyz ]')

        parser.add_argument(
            '--agg_alpha_xyz_mode',
            type=str,
            default="None",
                help='which agg xyz mode to use [None not to use | world world xyz | pers perspective xyz ]')

        parser.add_argument(
            '--agg_color_xyz_mode',
            type=str,
            default="None",
                help='which agg xyz mode to use [None not to use | world world xyz | pers perspective xyz ]')

        parser.add_argument(
            '--act_type',
            type=str,
            default="ReLU",
            # default="LeakyReLU",
            help='which agg xyz mode to use [None not to use | world world xyz | pers perspective xyz ]')

        parser.add_argument(
            '--act_super',
            type=int,
            default=1,
            # default="LeakyReLU",
            help='1 to use softplus and widden sigmoid for last activation')

        parser.add_argument('--predict_semantic',
                            type=int,
                            default=0,
                            help='if 0:donot use BPNet to predict semantic;1 use BPNet to predict semantic label')
        parser.add_argument('--layers_2d',
                            type=int,
                            default=34,
                            help='BPNet 2dUnet layers')
        parser.add_argument('--classes',
                            type=int,
                            default=20,
                            help='BPNet predict types')
        parser.add_argument('--arch_3d',
                            type=str,
                            default="MinkUNet18A",
                            help='BPNet arch_3d')     
        parser.add_argument('--bpnetweight',
                            type=str,
                            default="/qys/SG-NeRF/bpnetInitmodel/bpnet_5cm.pth.tar",
                            help='bpnet pretrained model weight'
        )

    def __init__(self, opt):

        super(PointAggregator, self).__init__()
        self.act = getattr(nn, opt.act_type, None)
        print("opt.act_type!!!!!!!!!", opt.act_type)
        self.point_hyper_dim=opt.point_hyper_dim if opt.point_hyper_dim < opt.point_features_dim else opt.point_features_dim#32

        block_init_lst = []
        if opt.agg_distance_kernel == "feat_intrp": #false
            feat_weight_block = []
            in_channels = 2 * opt.weight_xyz_freq * 3 + opt.weight_feat_dim
            out_channels = int(in_channels / 2)
            for i in range(2):
                feat_weight_block.append(nn.Linear(in_channels, out_channels))
                feat_weight_block.append(self.act(inplace=True))
                in_channels = out_channels
            feat_weight_block.append(nn.Linear(in_channels, 1))
            feat_weight_block.append(nn.Sigmoid())
            self.feat_weight_mlp = nn.Sequential(*feat_weight_block)
            block_init_lst.append(self.feat_weight_mlp)
        elif opt.agg_distance_kernel == "sh_intrp":  #false
            self.shcomp = SphericalHarm(opt.sh_degree)

        self.opt = opt
        self.dist_dim = (4 if self.opt.agg_dist_pers == 30 else 6) if self.opt.agg_dist_pers > 9 else 3
        self.dist_func = getattr(self, opt.agg_distance_kernel, None)
        assert self.dist_func is not None, "InterpAggregator doesn't have disance_kernel {} ".format(opt.agg_distance_kernel)

        self.axis_weight = None if opt.agg_axis_weight is None else torch.as_tensor(opt.agg_axis_weight, dtype=torch.float32, device="cuda")[None, None, None, None, :]

        self.num_freqs = opt.num_pos_freqs if opt.num_pos_freqs > 0 else 0
        self.num_viewdir_freqs = opt.num_viewdir_freqs if opt.num_viewdir_freqs > 0 else 0

        self.pnt_channels = (2 * self.num_freqs * 3) if self.num_freqs > 0 else 3
        self.viewdir_channels = (2 * self.num_viewdir_freqs * 3 + self.opt.view_ori * 3) if self.num_viewdir_freqs > 0 else 3

        self.which_agg_model = opt.which_agg_model.split("_")[0] if opt.which_agg_model.startswith("feathyper") else opt.which_agg_model
        getattr(self, self.which_agg_model+"_init", None)(opt, block_init_lst)

        self.density_super_act = torch.nn.Softplus()
        self.density_act = torch.nn.ReLU()
        self.color_act = torch.nn.Sigmoid()

    def raw2out_density(self, raw_density):
        # raw_density[35634,1]
        if self.opt.act_super > 0:#True
            # return self.density_act(raw_density - 1)  # according to mip nerf, to stablelize the training
            return self.density_super_act(raw_density - 1)  # according to mip nerf, to stablelize the training
        else:
            return self.density_act(raw_density)
    def raw2out_color(self, raw_color):
        color = self.color_act(raw_color)
        if self.opt.act_super > 0:#tRUE
            color = color * (1 + 2 * 0.001) - 0.001 # according to mip nerf, to stablelize the training
        return color


    def viewmlp_init(self, opt, block_init_lst):

        dist_xyz_dim = self.dist_dim if opt.dist_xyz_freq == 0 else 2 * abs(opt.dist_xyz_freq) * self.dist_dim
        in_channels = opt.point_features_dim + (0 if opt.agg_feat_xyz_mode == "None" else self.pnt_channels) - (opt.weight_feat_dim if opt.agg_distance_kernel in ["feat_intrp", "meta_intrp"] else 0) - (opt.sh_degree ** 2 if opt.agg_distance_kernel == "sh_intrp" else 0) - (7 if opt.agg_distance_kernel == "gau_intrp" else 0)
        in_channels += (2 * opt.num_feat_freqs * in_channels if opt.num_feat_freqs > 0 else 0) + (dist_xyz_dim if opt.agg_intrp_order > 0 else 0)
        
        if opt.shading_feature_mlp_layer1 > 0:#2
            out_channels = opt.shading_feature_num
            block1 = []
            for i in range(opt.shading_feature_mlp_layer1):
                block1.append(nn.Linear(in_channels, out_channels))
                block1.append(self.act(inplace=True))
                in_channels = out_channels
            self.block1 = nn.Sequential(*block1)
            block_init_lst.append(self.block1)
        else:
            self.block1 = self.passfunc


        if opt.shading_feature_mlp_layer2 > 0:
            in_channels = in_channels + (0 if opt.agg_feat_xyz_mode == "None" else self.pnt_channels) + (
                dist_xyz_dim if (opt.agg_intrp_order > 0 and opt.num_feat_freqs == 0) else 0)
            out_channels = opt.shading_feature_num
            block2 = []
            for i in range(opt.shading_feature_mlp_layer2):
                block2.append(nn.Linear(in_channels, out_channels))
                block2.append(self.act(inplace=True))
                in_channels = out_channels
            self.block2 = nn.Sequential(*block2)
            block_init_lst.append(self.block2)
        else:
            self.block2 = self.passfunc


        if opt.shading_feature_mlp_layer3 > 0:
            in_channels = in_channels + (3 if "1" in list(opt.point_color_mode) else 0) + (
                4 if "1" in list(opt.point_dir_mode) else 0)
            out_channels = opt.shading_feature_num
            block3 = []
            for i in range(opt.shading_feature_mlp_layer3):
                block3.append(nn.Linear(in_channels, out_channels))
                block3.append(self.act(inplace=True))
                in_channels = out_channels
            self.block3 = nn.Sequential(*block3)
            block_init_lst.append(self.block3)
        else:
            self.block3 = self.passfunc
        #layer4:rotatation careless
        if opt.shading_feature_mlp_layer4 > 0:
            in_channels = in_channels + 6*opt.num_feat_freqs+ (3 if "1" in list(opt.point_color_mode) else 0) + (96 if opt.predict_semantic==1 else 0)
            out_channels = opt.shading_feature_num
            block4 = []
            for i in range(opt.shading_feature_mlp_layer4):
                block4.append(nn.Linear(in_channels, out_channels))
                block4.append(self.act(inplace=True))
                in_channels = out_channels
            self.block4 = nn.Sequential(*block4)
            block_init_lst.append(self.block4)
        else:
            self.block4 = self.passfunc

        if opt.shading_feature_mlp_linear > 0:
            in_channels = in_channels
            out_channels = opt.shading_feature_num
            block_linear = []
            for i in range(opt.shading_feature_mlp_layer4):
                block_linear.append(nn.Linear(in_channels, out_channels))
                block_linear.append(self.act(inplace=True))
                in_channels = out_channels
            self.block_linear = nn.Sequential(*block_linear)
            block_init_lst.append(self.block_linear)
        else:
            self.block_linear = self.passfunc
        alpha_block = []
        in_channels = opt.shading_feature_num + (0 if opt.agg_alpha_xyz_mode == "None" else self.pnt_channels)
        out_channels = int(opt.shading_feature_num / 2)
        for i in range(opt.shading_alpha_mlp_layer - 1):
            alpha_block.append(nn.Linear(in_channels, out_channels))
            alpha_block.append(self.act(inplace=False))
            in_channels = out_channels
        alpha_block.append(nn.Linear(in_channels, 1))
        self.alpha_branch = nn.Sequential(*alpha_block)
        block_init_lst.append(self.alpha_branch)

        color_block = []
        in_channels = opt.shading_feature_num + self.viewdir_channels + (
            0 if opt.agg_color_xyz_mode == "None" else self.pnt_channels)
        out_channels = int(opt.shading_feature_num / 2)
        for i in range(opt.shading_color_mlp_layer - 1):
            color_block.append(nn.Linear(in_channels, out_channels))
            color_block.append(self.act(inplace=True))
            in_channels = out_channels
        color_block.append(nn.Linear(in_channels, 3))
        self.color_branch = nn.Sequential(*color_block)
        block_init_lst.append(self.color_branch)

        for m in block_init_lst:
            init_seq(m)
     

    def passfunc(self, input):
        return input


    def trilinear(self, embedding, dists, pnt_mask, vsize, grid_vox_sz, axis_weight=None):
        # dists: B * R * SR * K * 3
        # return B * R * SR * K
        dists = dists * pnt_mask[..., None]
        dists = dists / grid_vox_sz

        #  dist: [1, 797, 40, 8, 3];     pnt_mask: [1, 797, 40, 8]
        # dists = 1 + dists * torch.as_tensor([[1,1,1], [-1, 1, 1], [1, -1, 1], [1, 1, -1], [-1, 1, -1], [1, -1, -1], [-1, -1, 1], [-1, -1, -1]], dtype=torch.float32, device=dists.device).view(1, 1, 1, 8, 3)

        dists = 1 - torch.abs(dists)

        weights = pnt_mask * dists[..., 0] * dists[..., 1] * dists[..., 2]
        norm_weights = weights / torch.clamp(torch.sum(weights, dim=-1, keepdim=True), min=1e-8)

        # ijk = xyz.astype(np.int32)
        # i, j, k = ijk[:, 0], ijk[:, 1], ijk[:, 2]
        # V000 = data[i, j, k].astype(np.int32)
        # V100 = data[(i + 1), j, k].astype(np.int32)
        # V010 = data[i, (j + 1), k].astype(np.int32)
        # V001 = data[i, j, (k + 1)].astype(np.int32)
        # V101 = data[(i + 1), j, (k + 1)].astype(np.int32)
        # V011 = data[i, (j + 1), (k + 1)].astype(np.int32)
        # V110 = data[(i + 1), (j + 1), k].astype(np.int32)
        # V111 = data[(i + 1), (j + 1), (k + 1)].astype(np.int32)
        # xyz = xyz - ijk
        # x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        # Vxyz = (V000 * (1 - x) * (1 - y) * (1 - z)
        #         + V100 * x * (1 - y) * (1 - z) +
        #         + V010 * (1 - x) * y * (1 - z) +
        #         + V001 * (1 - x) * (1 - y) * z +
        #         + V101 * x * (1 - y) * z +
        #         + V011 * (1 - x) * y * z +
        #         + V110 * x * y * (1 - z) +
        #         + V111 * x * y * z)
        return norm_weights, embedding


    def avg(self, embedding, dists, pnt_mask, vsize, grid_vox_sz, axis_weight=None):
        # dists: B * channel* R * SR * K
        # return B * R * SR * K
        weights = pnt_mask * 1.0
        return weights, embedding


    def quadric(self, embedding, dists, pnt_mask, vsize, grid_vox_sz, axis_weight=None):
        # dists: B * channel* R * SR * K
        # return B * R * SR * K
        if axis_weight is None or (axis_weight[..., 0] == 1 and axis_weight[..., 1] == 1 and axis_weight[..., 2] ==1):
            weights = 1./ torch.clamp(torch.sum(torch.square(dists[..., :3]), dim=-1), min= 1e-8)
        else:
            weights = 1. / torch.clamp(torch.sum(torch.square(dists)* axis_weight, dim=-1), min=1e-8)
        weights = pnt_mask * weights
        return weights, embedding


    def numquadric(self, embedding, dists, pnt_mask, vsize, grid_vox_sz, axis_weight=None):
        # dists: B * channel* R * SR * K
        # return B * R * SR * K
        if axis_weight is None or (axis_weight[..., 0] == 1 and axis_weight[..., 1] == 1 and axis_weight[..., 2] ==1):
            weights = 1./ torch.clamp(torch.sum(torch.square(dists), dim=-1), min= 1e-8)
        else:
            weights = 1. / torch.clamp(torch.sum(torch.square(dists)* axis_weight, dim=-1), min=1e-8)
        weights = pnt_mask * weights
        return weights, embedding


    def linear(self, embedding, dists, pnt_mask, vsize, grid_vox_sz, axis_weight=None):
        # dists: B * R * SR * K * channel
        # return B * R * SR * K
        if axis_weight is None or (axis_weight[..., 0] == 1 and axis_weight[..., 2] ==1) :#True
            weights = 1. / torch.clamp(torch.norm(dists[..., :3], dim=-1), min= 1e-6)
        else:
            weights = 1. / torch.clamp(torch.sqrt(torch.sum(torch.square(dists[...,:2]), dim=-1)) * axis_weight[..., 0] + torch.abs(dists[...,2]) * axis_weight[..., 1], min= 1e-6)
        weights = pnt_mask * weights
        return weights, embedding


    def numlinear(self, embedding, dists, pnt_mask, vsize, grid_vox_sz, axis_weight=None):
        # dists: B * R * SR * K * channel
        # return B * R * SR * K
        if axis_weight is None or (axis_weight[..., 0] == 1 and axis_weight[..., 2] ==1) :
            weights = 1. / torch.clamp(torch.norm(dists, dim=-1), min= 1e-6)
        else:
            weights = 1. / torch.clamp(torch.sqrt(torch.sum(torch.square(dists[...,:2]), dim=-1)) * axis_weight[..., 0] + torch.abs(dists[...,2]) * axis_weight[..., 1], min= 1e-6)
        weights = pnt_mask * weights
        norm_weights = weights / torch.clamp(torch.sum(pnt_mask, dim=-1, keepdim=True), min=1)
        return norm_weights, embedding


    def sigmoid(self, input):
        return torch.sigmoid(input)


    def tanh(self, input):
        return torch.tanh(input)


    def sh_linear(self, dist_norm):
        return 1 / torch.clamp(dist_norm, min=1e-8)


    def sh_quadric(self, dist_norm):
        return 1 / torch.clamp(torch.square(dist_norm), min=1e-8)


    def sh_intrp(self, embedding, dists, pnt_mask, vsize, grid_vox_sz, axis_weight=None):
        # dists: B * R * SR * K * channel
        dist_norm = torch.linalg.norm(dists, dim=-1)
        dist_dirs = dists / torch.clamp(dist_norm[...,None], min=1e-8)
        shall = self.shcomp.sh_all(dist_dirs, filp_dir=False).view(dists.shape[:-1]+(self.shcomp.total_deg ** 2,))
        sh_coefs = embedding[..., :self.shcomp.total_deg ** 2]
        # shall: [1, 816, 24, 32, 16], sh_coefs: [1, 816, 24, 32, 16], pnt_mask: [1, 816, 24, 32]
        # debug: weights = pnt_mask * torch.sum(shall, dim=-1)
        # weights = pnt_mask * torch.sum(shall * getattr(self, self.opt.sh_act, None)(sh_coefs), dim=-1) * getattr(self, self.opt.sh_dist_func, None)(dist_norm)
        weights = pnt_mask * torch.sum(getattr(self, self.opt.sh_act, None)(shall * sh_coefs), dim=-1) * getattr(self, self.opt.sh_dist_func, None)(dist_norm) # changed
        return weights, embedding[..., self.shcomp.total_deg ** 2:]


    def gau_intrp(self, embedding, dists, pnt_mask, vsize, grid_vox_sz, axis_weight=None):
        # dists: B * R * SR * K * channel
        # dist: [1, 752, 40, 32, 3]
        B, R, SR, K, _ = dists.shape
        scale = torch.abs(embedding[..., 0]) #
        radii = vsize[2] * 20 * torch.sigmoid(embedding[..., 1:4])
        rotations = torch.clamp(embedding[..., 4:7], max=np.pi / 4, min=-np.pi / 4)
        gau_dist = compute_world2local_dist(dists, radii, rotations)[..., 0]
        # print("gau_dist", gau_dist.shape)
        weights = pnt_mask * scale * torch.exp(-0.5 * torch.sum(torch.square(gau_dist), dim=-1))
        # print("gau_dist", gau_dist.shape, gau_dist[0, 0])
        # print("weights", weights.shape, weights[0, 0, 0])
        return weights, embedding[..., 7:]


    def viewmlp(self, sampled_color,sampled_label_embedding, sampled_Rw2c, sampled_dir, sampled_conf, sampled_embedding, sampled_xyz_pers, sampled_xyz, sample_pnt_mask, sample_loc, sample_loc_w, sample_ray_dirs, vsize, weight, pnt_mask_flat, pts, viewdirs, total_len, ray_valid, in_shape, dists):
        # print("sampled_Rw2c", sampled_Rw2c.shape, sampled_xyz.shape)
        # assert sampled_Rw2c.dim() == 2
        B, R, SR, K, _ = dists.shape#B 1 R:784 K:8 SR:24
        sampled_Rw2c = sampled_Rw2c.transpose(-1, -2)
        uni_w2c = sampled_Rw2c.dim() == 2#True
        if not uni_w2c:#False
            sampled_Rw2c_ray = sampled_Rw2c[:,:,:,0,:,:].view(-1, 3, 3)
            sampled_Rw2c = sampled_Rw2c.reshape(-1, 3, 3)[pnt_mask_flat, :, :]
        pts_ray, pts_pnt = None, None
        if self.opt.agg_feat_xyz_mode != "None" or self.opt.agg_alpha_xyz_mode != "None" or self.opt.agg_color_xyz_mode != "None":#False
            if self.num_freqs > 0:
                pts = positional_encoding(pts, self.num_freqs)
            pts_ray = pts[ray_valid, :]
            if self.opt.agg_feat_xyz_mode != "None" and self.opt.agg_intrp_order > 0:
                pts_pnt = pts[..., None, :].repeat(1, K, 1).view(-1, pts.shape[-1])
                if self.opt.apply_pnt_mask > 0:
                    pts_pnt=pts_pnt[pnt_mask_flat, :]
        viewdirs = viewdirs @ sampled_Rw2c if uni_w2c else (viewdirs[..., None, :] @ sampled_Rw2c_ray).squeeze(-2)#换左右手系[18816,3]
        if self.num_viewdir_freqs > 0:#True，进行PE操作，viewdir做4级的傅里叶-
            viewdirs = positional_encoding(viewdirs, self.num_viewdir_freqs, ori=True)#[18816,27]
            ori_viewdirs, viewdirs = viewdirs[..., :3], viewdirs[..., 3:]
            #[18816，3]

        viewdirs = viewdirs[ray_valid, :]
        #attention
        if self.opt.agg_intrp_order == 0:#False
            feat = torch.sum(sampled_embedding * weight[..., None], dim=-2)
            feat = feat.view([-1, feat.shape[-1]])[ray_valid, :]
            if self.opt.num_feat_freqs > 0:
                feat = torch.cat([feat, positional_encoding(feat, self.opt.num_feat_freqs)], dim=-1)
            pts = pts_ray
        else:#True
            dists_flat = dists.view(-1, dists.shape[-1])#dists:[1,784,24,8,6],colmap的input点云到query的点的距离，目前只有[...，:3]有用[150528,6]
            if self.opt.apply_pnt_mask > 0:#True
                dists_flat = dists_flat[pnt_mask_flat, :]
            dists_flat /= (# 不变
                1.0 if self.opt.dist_xyz_deno == 0. else float(self.opt.dist_xyz_deno * np.linalg.norm(vsize)))
            dists_flat[..., :3] = dists_flat[..., :3] @ sampled_Rw2c if uni_w2c else (dists_flat[..., None, :3] @ sampled_Rw2c).squeeze(-2)#不变
            if self.opt.dist_xyz_freq != 0:
                # print(dists.dtype, (self.opt.dist_xyz_deno * np.linalg.norm(vsize)).dtype, dists_flat.dtype)
                dists_flat = positional_encoding(dists_flat, self.opt.dist_xyz_freq)
            feat= sampled_embedding.view(-1, sampled_embedding.shape[-1])
            # feat[150528,32],feature? 一个32D的点云feature；150528=1*28*28*24*8

            if self.opt.apply_pnt_mask > 0:
                feat = feat[pnt_mask_flat, :]#[35634,32]

            if self.opt.num_feat_freqs > 0:
                feat = torch.cat([feat, positional_encoding(feat, self.opt.num_feat_freqs)], dim=-1)#feat[48739,224],为什么feature也要做PE？
            feat = torch.cat([feat, dists_flat], dim=-1)#feat[48739,284],284=224+60;3d的dist和32d的feature都被PE了
            weight = weight.view(B * R * SR, K, 1)#[18816,8,1]
            pts = pts_pnt

        # used_point_embedding = feat[..., : self.opt.point_features_dim]

        if self.opt.agg_feat_xyz_mode != "None":#False
            feat = torch.cat([feat, pts], dim=-1)
        # print("feat",feat.shape) # 501
        feat = self.block1(feat)#Equation 3 ,Neural Network F 。input[28162,284]->output[28162,256]

        if self.opt.shading_feature_mlp_layer2>0:#Flase
            if self.opt.agg_feat_xyz_mode != "None":
                feat = torch.cat([feat, pts], dim=-1)
            if self.opt.agg_intrp_order > 0:
                feat = torch.cat([feat, dists_flat], dim=-1)
            feat = self.block2(feat)

        if self.opt.shading_feature_mlp_linear>0:
            feat = self.block_linear(feat)
        feat_branch = feat
        if self.opt.shading_feature_mlp_layer3>0:#True,2
            if sampled_color is not None:
                sampled_color = sampled_color.view(-1, sampled_color.shape[-1])#[150528,3]
                if self.opt.apply_pnt_mask > 0:
                    sampled_color = sampled_color[pnt_mask_flat, :]
                feat = torch.cat([feat, sampled_color], dim=-1)#[35634,256+3]
            if sampled_dir is not None:#True
                sampled_dir = sampled_dir.view(-1, sampled_dir.shape[-1])
                if self.opt.apply_pnt_mask > 0:
                    sampled_dir = sampled_dir[pnt_mask_flat, :]
                    sampled_dir = sampled_dir @ sampled_Rw2c if uni_w2c else (sampled_dir[..., None, :] @ sampled_Rw2c).squeeze(-2)
                ori_viewdirs = ori_viewdirs[..., None, :].repeat(1, K, 1).view(-1, ori_viewdirs.shape[-1])
                if self.opt.apply_pnt_mask > 0:
                    ori_viewdirs = ori_viewdirs[pnt_mask_flat, :]
                feat = torch.cat([feat, sampled_dir - ori_viewdirs, torch.sum(sampled_dir*ori_viewdirs, dim=-1, keepdim=True)], dim=-1)
            feat = self.block3(feat)#[35634,256]


        if self.opt.shading_feature_mlp_layer4>0:#implement by my-self
            if sampled_color is not None:
                sampled_color = sampled_color.view(-1, sampled_color.shape[-1])  # [150528,3]
                if self.opt.apply_pnt_mask > 0:
                    sampled_color = sampled_color[pnt_mask_flat, :]
                feat = torch.cat([feat, sampled_color], dim=-1)  # [35634,256+3]
            if sampled_dir is not None:  # True
                sampled_dir = sampled_dir.view(-1, sampled_dir.shape[-1])
                if self.opt.apply_pnt_mask > 0:
                    sampled_dir = sampled_dir[pnt_mask_flat, :]
                    sampled_dir = sampled_dir @ sampled_Rw2c if uni_w2c else (
                                sampled_dir[..., None, :] @ sampled_Rw2c).squeeze(-2)
                ori_viewdirs = ori_viewdirs[..., None, :].repeat(1, K, 1).view(-1, ori_viewdirs.shape[-1])
                if self.opt.apply_pnt_mask > 0:
                    ori_viewdirs = ori_viewdirs[pnt_mask_flat, :]

                proxy_sampled_dir = sampled_dir[:,:2]#[ptr,2]
                proxy_ori_viewdirs = ori_viewdirs[:, :2]#[ptr,2]
                proxz_sampled_dir = sampled_dir[:, ::2]  # [ptr,2]
                proxz_ori_viewdirs = ori_viewdirs[:, ::2]  # [ptr,2]proxy_sampled_dir = sampled_dir[:,:2]#[ptr,2]
                proyz_sampled_dir = sampled_dir[:, 1:]  # [ptr,2]
                proyz_ori_viewdirs = ori_viewdirs[:, 1:]  # [ptr,2]proxy_sampled_dir = sampled_dir[:,:2]#[ptr,2]
                theta = torch.sum(proxy_sampled_dir*proxy_ori_viewdirs,dim = -1)/torch.norm(proxy_sampled_dir,dim=-1)/torch.norm(proxy_ori_viewdirs,dim=-1)#[37410]
                clockwise_theta_msk = torch.where(proxy_sampled_dir[:,0]*proxy_ori_viewdirs[:,1]-proxy_sampled_dir[:,1]*proxy_ori_viewdirs[:,1]>0,1,-1)
                theta = clockwise_theta_msk*theta#[ptr]
                row = torch.sum(proxz_sampled_dir * proxz_ori_viewdirs, dim=-1) / torch.norm(proxz_sampled_dir,dim=-1) / torch.norm(proxz_ori_viewdirs, dim=-1)  # [37410]
                clockwise_row_msk = torch.where(proxz_sampled_dir[:, 0] * proxz_ori_viewdirs[:, 1] - proxz_sampled_dir[:, 1] * proxz_ori_viewdirs[:,1] > 0, 1, -1)
                row = clockwise_row_msk * row  # [ptr]
                fai = torch.sum(proyz_sampled_dir * proyz_ori_viewdirs, dim=-1) / torch.norm(proyz_sampled_dir,dim=-1) / torch.norm(proyz_ori_viewdirs, dim=-1) # [37410]
                clockwise_fai_msk = torch.where(proyz_sampled_dir[:, 0] * proyz_ori_viewdirs[:, 1] - proyz_sampled_dir[:, 1] * proyz_ori_viewdirs[:,1] > 0, 1, -1)
                fai = clockwise_fai_msk * fai  # [ptr]
                row_theta_fai_feat = torch.cat([row[...,None],theta[...,None],fai[...,None]], dim=-1)#18
                row_theta_fai_feat = positional_encoding(row_theta_fai_feat,self.opt.num_feat_freqs)#18
                feat = torch.cat([feat, row_theta_fai_feat],dim= -1)
            if sampled_label_embedding is not None:
                sampled_label_embedding = sampled_label_embedding.view(-1, sampled_label_embedding.shape[-1])
                if self.opt.apply_pnt_mask > 0:
                    sampled_label_embedding = sampled_label_embedding[pnt_mask_flat, :]
                feat = torch.cat([feat, sampled_label_embedding], dim=-1)  # [35634,256+3]
            
            feat = self.block4(feat)  # [35634,256+18+96]


        if self.opt.agg_intrp_order == 1:#False

            if self.opt.apply_pnt_mask > 0:
                feat_holder = torch.zeros([B * R * SR * K, feat.shape[-1]], dtype=torch.float32, device=feat.device)
                feat_holder[pnt_mask_flat, :] = feat
            else:
                feat_holder = feat
            feat = feat_holder.view(B * R * SR, K, feat_holder.shape[-1])
            feat = torch.sum(feat * weight, dim=-2).view([-1, feat.shape[-1]])[ray_valid, :]


            alpha_in = feat_branch
            if self.opt.agg_alpha_xyz_mode != "None":
                alpha_in = torch.cat([alpha_in, pts], dim=-1)

            alpha = self.raw2out_density(self.alpha_branch(alpha_in))

            color_in = feat
            if self.opt.agg_color_xyz_mode != "None":
                color_in = torch.cat([color_in, pts], dim=-1)

            color_in = torch.cat([color_in, viewdirs], dim=-1)
            color_output = self.raw2out_color(self.color_branch(color_in))

            # print("color_output", torch.sum(color_output), color_output.grad)

            output = torch.cat([alpha, color_output], dim=-1)

        elif self.opt.agg_intrp_order == 2:#True
            alpha_in = feat_branch#[35634,256]
            if self.opt.agg_alpha_xyz_mode != "None":
                alpha_in = torch.cat([alpha_in, pts], dim=-1)
            alpha = self.raw2out_density(self.alpha_branch(alpha_in))
            # print(alpha_in.shape, alpha_in)

            if self.opt.apply_pnt_mask > 0:
                alpha_holder = torch.zeros([B * R * SR * K, alpha.shape[-1]], dtype=torch.float32, device=alpha.device)
                alpha_holder[pnt_mask_flat, :] = alpha
            else:
                alpha_holder = alpha
            alpha = alpha_holder.view(B * R * SR, K, alpha_holder.shape[-1])#[18816,8,1]
            alpha = torch.sum(alpha * weight, dim=-2).view([-1, alpha.shape[-1]])[ray_valid, :]#equation 4 # alpha:[4890,1]

            # print("alpha", alpha.shape)
            # alpha_placeholder = torch.zeros([total_len, 1], dtype=torch.float32,
            #                                 device=alpha.device)
            # alpha_placeholder[ray_valid] = alpha


            if self.opt.apply_pnt_mask > 0:#True
                feat_holder = torch.zeros([B * R * SR * K, feat.shape[-1]], dtype=torch.float32, device=feat.device)
                feat_holder[pnt_mask_flat, :] = feat
            else:
                feat_holder = feat
            feat = feat_holder.view(B * R * SR, K, feat_holder.shape[-1])
            feat = torch.sum(feat * weight, dim=-2).view([-1, feat.shape[-1]])[ray_valid, :]#[4890,256]

            color_in = feat#[4871,256]
            if self.opt.agg_color_xyz_mode != "None":#False
                color_in = torch.cat([color_in, pts], dim=-1)

            color_in = torch.cat([color_in, viewdirs], dim=-1)
            color_output = self.raw2out_color(self.color_branch(color_in)) #[4890,3]
            # color_output = torch.sigmoid(color_output)
            # output_placeholder = torch.cat([alpha, color_output], dim=-1)
            output = torch.cat([alpha, color_output], dim=-1)#[10301,4]rgb+alpha

            # print("output_placeholder", output_placeholder.shape)
        output_placeholder = torch.zeros([total_len, self.opt.shading_color_channel_num + 1], dtype=torch.float32, device=output.device)
        output_placeholder[ray_valid] = output
        #output_placeholder[18816,4],18816=1*28*28*24
        return output_placeholder, None
    def print_point(self, dists, sample_loc_w, sampled_xyz, sample_loc, sampled_xyz_pers, sample_pnt_mask):

        # for i in range(dists.shape[0]):
        #     filepath = "./dists.txt"
        #     filepath1 = "./dists10.txt"
        #     filepath2 = "./dists20.txt"
        #     filepath3 = "./dists30.txt"
        #     filepath4 = "./dists40.txt"
        #     dists_cpu = dists.detach().cpu().numpy()
        #     np.savetxt(filepath1, dists_cpu[i, 80, 0, ...].reshape(-1, 3), delimiter=";")
        #     np.savetxt(filepath2, dists_cpu[i, 80, 3, ...].reshape(-1, 3), delimiter=";")
        #     np.savetxt(filepath3, dists_cpu[i, 80, 6, ...].reshape(-1, 3), delimiter=";")
        #     np.savetxt(filepath4, dists_cpu[i, 80, 9, ...].reshape(-1, 3), delimiter=";")
        #     dists_cpu = dists[i,...][torch.any(sample_pnt_mask, dim=-1)[i,...], :].detach().cpu().numpy()
        #     np.savetxt(filepath, dists_cpu.reshape(-1, 3), delimiter=";")

        for i in range(sample_loc_w.shape[0]):
            filepath = "./sample_loc_w.txt"
            filepath1 = "./sample_loc_w10.txt"
            filepath2 = "./sample_loc_w20.txt"
            filepath3 = "./sample_loc_w30.txt"
            filepath4 = "./sample_loc_w40.txt"
            sample_loc_w_cpu = sample_loc_w.detach().cpu().numpy()
            np.savetxt(filepath1, sample_loc_w_cpu[i, 80, 0, ...].reshape(-1, 3), delimiter=";")
            np.savetxt(filepath2, sample_loc_w_cpu[i, 80, 3, ...].reshape(-1, 3), delimiter=";")
            np.savetxt(filepath3, sample_loc_w_cpu[i, 80, 6, ...].reshape(-1, 3), delimiter=";")
            np.savetxt(filepath4, sample_loc_w_cpu[i, 80, 9, ...].reshape(-1, 3), delimiter=";")
            sample_loc_w_cpu = sample_loc_w[i,...][torch.any(sample_pnt_mask, dim=-1)[i,...], :].detach().cpu().numpy()
            np.savetxt(filepath, sample_loc_w_cpu.reshape(-1, 3), delimiter=";")


        for i in range(sampled_xyz.shape[0]):
            sampled_xyz_cpu = sampled_xyz.detach().cpu().numpy()
            filepath = "./sampled_xyz.txt"
            filepath1 = "./sampled_xyz10.txt"
            filepath2 = "./sampled_xyz20.txt"
            filepath3 = "./sampled_xyz30.txt"
            filepath4 = "./sampled_xyz40.txt"
            np.savetxt(filepath1, sampled_xyz_cpu[i, 80, 0, ...].reshape(-1, 3), delimiter=";")
            np.savetxt(filepath2, sampled_xyz_cpu[i, 80, 3, ...].reshape(-1, 3), delimiter=";")
            np.savetxt(filepath3, sampled_xyz_cpu[i, 80, 6, ...].reshape(-1, 3), delimiter=";")
            np.savetxt(filepath4, sampled_xyz_cpu[i, 80, 9, ...].reshape(-1, 3), delimiter=";")
            np.savetxt(filepath, sampled_xyz_cpu[i, ...].reshape(-1, 3), delimiter=";")

        for i in range(sample_loc.shape[0]):
            filepath1 = "./sample_loc10.txt"
            filepath2 = "./sample_loc20.txt"
            filepath3 = "./sample_loc30.txt"
            filepath4 = "./sample_loc40.txt"
            filepath = "./sample_loc.txt"
            sample_loc_cpu =sample_loc.detach().cpu().numpy()

            np.savetxt(filepath1, sample_loc_cpu[i, 80, 0, ...].reshape(-1, 3), delimiter=";")
            np.savetxt(filepath2, sample_loc_cpu[i, 80, 3, ...].reshape(-1, 3), delimiter=";")
            np.savetxt(filepath3, sample_loc_cpu[i, 80, 6, ...].reshape(-1, 3), delimiter=";")
            np.savetxt(filepath4, sample_loc_cpu[i, 80, 9, ...].reshape(-1, 3), delimiter=";")
            np.savetxt(filepath, sample_loc[i, ...][torch.any(sample_pnt_mask, dim=-1)[i,...], :].reshape(-1, 3).detach().cpu().numpy(), delimiter=";")

        for i in range(sampled_xyz_pers.shape[0]):
            filepath1 = "./sampled_xyz_pers10.txt"
            filepath2 = "./sampled_xyz_pers20.txt"
            filepath3 = "./sampled_xyz_pers30.txt"
            filepath4 = "./sampled_xyz_pers40.txt"
            filepath = "./sampled_xyz_pers.txt"
            sampled_xyz_pers_cpu = sampled_xyz_pers.detach().cpu().numpy()

            np.savetxt(filepath1, sampled_xyz_pers_cpu[i, 80, 0, ...].reshape(-1, 3), delimiter=";")
            np.savetxt(filepath2, sampled_xyz_pers_cpu[i, 80, 3, ...].reshape(-1, 3), delimiter=";")
            np.savetxt(filepath3, sampled_xyz_pers_cpu[i, 80, 6, ...].reshape(-1, 3), delimiter=";")
            np.savetxt(filepath4, sampled_xyz_pers_cpu[i, 80, 9, ...].reshape(-1, 3), delimiter=";")

            np.savetxt(filepath, sampled_xyz_pers_cpu[i, ...].reshape(-1, 3), delimiter=";")
        print("saved sampled points and shading points")
        exit()


    def gradiant_clamp(self, sampled_conf, min=0.0001, max=1):
        diff = sampled_conf - torch.clamp(sampled_conf, min=min, max=max)
        return sampled_conf - diff.detach()


    def forward(self, sampled_color,sampled_label_embedding,  sampled_Rw2c, sampled_dir, sampled_conf, sampled_embedding, sampled_xyz_pers, sampled_xyz, sample_pnt_mask, sample_loc, sample_loc_w, sample_ray_dirs, vsize, grid_vox_sz):
        # return B * R * SR * channel
        '''
        sampled_color [1,784,24,8,3]
        sampled_Rw2c ones(3)
        sampled_dir [1,784,24,8,3]
        sampled_conf[1.784.24.8.1]
        sampled_embedding[1,784,24,8,32]
        sampled_xyz_pers[1,784,24,8,3]
        sampled_xyz[1,784,24,8,3]
        sample_pnt_mask[1,784,24,8]
        sample_loc[1,784,24,3]
        sample_loc_w[1,784,24,3]
        sample_ray_dirs[1,784,24,3]
        vsize[0.008 0.008 0.008]
        grid_vox_sz = 0
        '''
        ray_valid = torch.any(sample_pnt_mask, dim=-1).view(-1)
        total_len = len(ray_valid)#18816：784*24
        in_shape = sample_loc_w.shape
        if total_len == 0 or torch.sum(ray_valid) == 0:#False
            # print("skip since no valid ray, total_len:", total_len, torch.sum(ray_valid))
            return torch.zeros(in_shape[:-1] + (self.opt.shading_color_channel_num + 1,), device=ray_valid.device, dtype=torch.float32), ray_valid.view(in_shape[:-1]), None, None

        if self.opt.agg_dist_pers < 0: #False
            dists = sample_loc_w[..., None, :]
        elif self.opt.agg_dist_pers == 0: #False
            dists = sampled_xyz - sample_loc_w[..., None, :]
        elif self.opt.agg_dist_pers == 1: #False
            dists = sampled_xyz_pers - sample_loc[..., None, :]
        elif self.opt.agg_dist_pers == 2: #False
            if sampled_xyz_pers.shape[1] > 0:
                xdist = sampled_xyz_pers[..., 0] * sampled_xyz_pers[..., 2] - sample_loc[:, :, :, None, 0] * sample_loc[:, :, :, None, 2]
                ydist = sampled_xyz_pers[..., 1] * sampled_xyz_pers[..., 2] - sample_loc[:, :, :, None, 1] * sample_loc[:, :, :, None, 2]
                zdist = sampled_xyz_pers[..., 2] - sample_loc[:, :, :, None, 2]
                dists = torch.stack([xdist, ydist, zdist], dim=-1)
            else:
                B, R, SR, K, _ = sampled_xyz_pers.shape
                dists = torch.zeros([B, R, SR, K, 3], device=sampled_xyz_pers.device, dtype=sampled_xyz_pers.dtype)

        elif self.opt.agg_dist_pers == 10:#False

            if sampled_xyz_pers.shape[1] > 0:
                dists = sampled_xyz_pers - sample_loc[..., None, :]
                dists = torch.cat([sampled_xyz - sample_loc_w[..., None, :], dists], dim=-1)
            else:
                B, R, SR, K, _ = sampled_xyz_pers.shape
                dists = torch.zeros([B, R, SR, K, 6], device=sampled_xyz_pers.device, dtype=sampled_xyz_pers.dtype)
        #Attention1!!!!
        elif self.opt.agg_dist_pers == 20:#True!!!!!!
                #sampled_xyz_pers：[1,784,24,8,3]
            if sampled_xyz_pers.shape[1] > 0:#True!!!!!!为什么做一个Z的操作，归一化吗？
                xdist = sampled_xyz_pers[..., 0] * sampled_xyz_pers[..., 2] - sample_loc[:, :, :, None, 0] * sample_loc[:, :, :, None, 2]#[1,784,24,8]
                ydist = sampled_xyz_pers[..., 1] * sampled_xyz_pers[..., 2] - sample_loc[:, :, :, None, 1] * sample_loc[:, :, :, None, 2]#[1,784,24,8]
                zdist = sampled_xyz_pers[..., 2] - sample_loc[:, :, :, None, 2]#[1,784,24,8]
                dists = torch.stack([xdist, ydist, zdist], dim=-1)#[1,784,24,8,3],这段dists不知道干啥的，
                # dists = torch.cat([sampled_xyz - sample_loc_w[..., None, :], dists], dim=-1)
                dists = torch.cat([sampled_xyz - sample_loc_w[..., None, :], dists], dim=-1)#前一段dist确实是dist，后一段dist不知道干啥的，要*z;但也无所谓，没用到，只用到了[:3]
            else:
                B, R, SR, K, _ = sampled_xyz_pers.shape
                dists = torch.zeros([B, R, SR, K, 6], device=sampled_xyz_pers.device, dtype=sampled_xyz_pers.dtype)

        elif self.opt.agg_dist_pers == 30:

            if sampled_xyz_pers.shape[1] > 0:
                w_dists = sampled_xyz - sample_loc_w[..., None, :]
                dists = torch.cat([torch.sum(w_dists*sample_ray_dirs[..., None, :], dim=-1, keepdim=True), dists], dim=-1)
            else:
                B, R, SR, K, _ = sampled_xyz_pers.shape
                dists = torch.zeros([B, R, SR, K, 4], device=sampled_xyz_pers.device, dtype=sampled_xyz_pers.dtype)
        else:
            print("illegal agg_dist_pers code: ", agg_dist_pers)
            exit()
        # self.print_point(dists, sample_loc_w, sampled_xyz, sample_loc, sampled_xyz_pers, sample_pnt_mask)
        #dist_func:linear；weight[1,784,24,8],paper eq.4中的wi
        weight, sampled_embedding = self.dist_func(sampled_embedding, dists, sample_pnt_mask, vsize, grid_vox_sz, axis_weight=self.axis_weight)
        #weight[1，784，24，8]；sampled_embedding[1,784,24,8,32]
        #并对weight做归一化
        if self.opt.agg_weight_norm > 0 and self.opt.agg_distance_kernel != "trilinear" and not self.opt.agg_distance_kernel.startswith("num"):#True
            weight = weight / torch.clamp(torch.sum(weight, dim=-1, keepdim=True), min=1e-8)
        pnt_mask_flat = sample_pnt_mask.view(-1)#sample_pnt_mask：[1,784,24,8]
        pts = sample_loc_w.view(-1, sample_loc_w.shape[-1])#[18816,3]
        viewdirs = sample_ray_dirs.view(-1, sample_ray_dirs.shape[-1])#[18816,3]
        conf_coefficient = 1
        if sampled_conf is not None:#True
            conf_coefficient = self.gradiant_clamp(sampled_conf[..., 0], min=0.0001, max=1)#[1,784,24,8],all are 1
        #put data into nerual network at this line
        output, _ = getattr(self, self.which_agg_model, None)(sampled_color,sampled_label_embedding, sampled_Rw2c, sampled_dir, sampled_conf, sampled_embedding, sampled_xyz_pers, sampled_xyz, sample_pnt_mask, sample_loc, sample_loc_w, sample_ray_dirs, vsize, weight * conf_coefficient, pnt_mask_flat, pts, viewdirs, total_len, ray_valid, in_shape, dists)
        #output[18816,4]
        if (self.opt.sparse_loss_weight <=0) and ("conf_coefficient" not in self.opt.zero_one_loss_items) and self.opt.prob == 0:#False
            weight, conf_coefficient = None, None
        return output.view(in_shape[:-1] + (self.opt.shading_color_channel_num + 1,)), ray_valid.view(in_shape[:-1]), weight, conf_coefficient
