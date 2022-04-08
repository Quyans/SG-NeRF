import os
import numpy as np
from numpy import dot
from math import sqrt
import pycuda
from pycuda.compiler import SourceModule
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
import matplotlib.pyplot as plt
import torch
import pickle
import time
from models.rendering.diff_ray_marching import near_far_linear_ray_generation, near_far_disparity_linear_ray_generation

from data.load_blender import load_blender_data

# X = torch.cuda.FloatTensor(8)


class Holder(pycuda.driver.PointerHolderBase):
    def __init__(self, t):
        super(Holder, self).__init__()
        self.t = t
        self.gpudata = t.data_ptr()

    def get_pointer(self):
        return self.t.data_ptr()

class lighting_fast_querier():

    def __init__(self, device, opt):

        print("querier device", device, device.index)
        self.gpu = device.index
        self.opt = opt
        drv.init()
        # self.device = drv.Device(gpu)
        self.ctx = drv.Device(self.gpu).make_context()
        self.claim_occ, self.map_coor2occ, self.fill_occ2pnts, self.mask_raypos, self.get_shadingloc, self.query_along_ray = self.build_cuda()
        self.inverse = self.opt.inverse
        self.count=0

    def clean_up(self):
        self.ctx.pop()

    def get_hyperparameters(self, vsize_np, point_xyz_w_tensor, ranges=None):
        '''
        vsize_np:[0.008, 0.008, 0.008]
        :return:
        '''
        min_xyz, max_xyz = torch.min(point_xyz_w_tensor, dim=-2)[0][0], torch.max(point_xyz_w_tensor, dim=-2)[0][0]
        vscale_np = np.array(self.opt.vscale, dtype=np.int32)#[2,2,2]
        scaled_vsize_np = (vsize_np * vscale_np).astype(np.float32)#[0.008, 0.008, 0.008]
        if ranges is not None:#not none :[-10.0, -10.0, -10.0, 10.0, 10.0, 10.0]
            # print("min_xyz", min_xyz.shape)
            # print("max_xyz", max_xyz.shape)
            # print("ranges", ranges)
            min_xyz, max_xyz = torch.max(torch.stack([min_xyz, torch.as_tensor(ranges[:3], dtype=torch.float32, device=min_xyz.device)], dim=0), dim=0)[0], torch.min(torch.stack([max_xyz, torch.as_tensor(ranges[3:], dtype=torch.float32,  device=min_xyz.device)], dim=0), dim=0)[0]
        min_xyz = min_xyz - torch.as_tensor(scaled_vsize_np * self.opt.kernel_size / 2, device=min_xyz.device, dtype=torch.float32)#kernel_size[3,3,3]
        max_xyz = max_xyz + torch.as_tensor(scaled_vsize_np * self.opt.kernel_size / 2, device=min_xyz.device, dtype=torch.float32)#kernel_size[3,3,3]

        ranges_np = torch.cat([min_xyz, max_xyz], dim=-1).cpu().numpy().astype(np.float32)
        # print("ranges_np",ranges_np)
        vdim_np = (max_xyz - min_xyz).cpu().numpy() / vsize_np

        scaled_vdim_np = np.ceil(vdim_np / vscale_np).astype(np.int32)
        ranges_gpu, scaled_vsize_gpu, scaled_vdim_gpu, vscale_gpu, kernel_size_gpu, query_size_gpu = np_to_gpuarray(
            ranges_np, scaled_vsize_np, scaled_vdim_np, vscale_np, np.asarray(self.opt.kernel_size, dtype=np.int32),
            np.asarray(self.opt.query_size, dtype=np.int32))

        radius_limit_np, depth_limit_np = self.opt.radius_limit_scale * max(vsize_np[0], vsize_np[1]), self.opt.depth_limit_scale * vsize_np[2]
        return np.asarray(radius_limit_np).astype(np.float32), np.asarray(depth_limit_np).astype(np.float32), ranges_np, vsize_np, vdim_np, scaled_vsize_np, scaled_vdim_np, vscale_np, ranges_gpu, scaled_vsize_gpu, scaled_vdim_gpu, vscale_gpu, kernel_size_gpu, query_size_gpu


    def query_points(self, pixel_idx_tensor, point_xyz_pers_tensor, point_xyz_w_tensor, actual_numpoints_tensor, h, w, intrinsic, near_depth, far_depth, ray_dirs_tensor, cam_pos_tensor, cam_rot_tensor):
        near_depth, far_depth = np.asarray(near_depth).item() , np.asarray(far_depth).item()#0.1，8
        radius_limit_np, depth_limit_np, ranges_np, vsize_np, vdim_np, scaled_vsize_np, scaled_vdim_np, vscale_np, range_gpu, scaled_vsize_gpu, scaled_vdim_gpu, vscale_gpu, kernel_size_gpu, query_size_gpu = self.get_hyperparameters(self.opt.vsize, point_xyz_w_tensor, ranges=self.opt.ranges)
        # print("self.opt.ranges", self.opt.ranges, range_gpu, ray_dirs_tensor)
        if self.opt.inverse > 0:
            raypos_tensor, _, _, _ = near_far_disparity_linear_ray_generation(cam_pos_tensor, ray_dirs_tensor, self.opt.z_depth_dim, near=near_depth, far=far_depth, jitter=0.3 if self.opt.is_train > 0 else 0.)
        else:
            # raypos_tensor[1,784,400,3]->世界坐标系下的，需要采样的点的坐标
            raypos_tensor, _, _, _ = near_far_linear_ray_generation(cam_pos_tensor, ray_dirs_tensor, self.opt.z_depth_dim, near=near_depth, far=far_depth, jitter=0.3 if self.opt.is_train > 0 else 0.)#将28*28个像素坐标转化成了camera坐标系下的3D坐标
        #sample_pidx_tensor[1,784,24,8]每个像素(784)，需要采样的每个query点(24)的点云中临近8点
        #sample_loc_w_tensor[1,784,24,3],init:all-0，存某个pixel需要query的点的坐标
        #ray_mask_tensor[1,784]true or false，存放不需要采集的像素的msk
        sample_pidx_tensor, sample_loc_w_tensor, ray_mask_tensor = self.query_grid_point_index(h, w, pixel_idx_tensor, raypos_tensor, point_xyz_w_tensor, actual_numpoints_tensor, kernel_size_gpu, query_size_gpu, self.opt.SR, self.opt.K, ranges_np, scaled_vsize_np, scaled_vdim_np, vscale_np, self.opt.max_o, self.opt.P, radius_limit_np, depth_limit_np, range_gpu, scaled_vsize_gpu, scaled_vdim_gpu, vscale_gpu, ray_dirs_tensor, cam_pos_tensor, kMaxThreadsPerBlock=self.opt.gpu_maxthr)
        sample_ray_dirs_tensor = torch.masked_select(ray_dirs_tensor, ray_mask_tensor[..., None]>0).reshape(ray_dirs_tensor.shape[0],-1,3)[...,None,:].expand(-1, -1, self.opt.SR, -1).contiguous()
        #sample_pidx_tensor[1,784,24,8]每个像素(784)，需要采样的每个query点(24)的点云中临近8点
        #elf.w2pers(sample_loc_w_tensor, cam_rot_tensor, cam_pos_tensor) sample_loc_w_tensor转了坐标系
        #sample_loc_w_tensor[1,784,24,3],init:all-0，存某个pixel需要query的点的坐标
        #sample_ray_dirs_tensor[1,784,24,3]方向？
        #ray_mask_tensor[1,784]true or false，存放不需要采集的像素的msk
        #vsize_np[0.008 0.008 0.008]
        #ranges_np[-1.6265 -1.9573 -3.2914 3.868 4.070 2.417]
        return sample_pidx_tensor, self.w2pers(sample_loc_w_tensor, cam_rot_tensor, cam_pos_tensor), sample_loc_w_tensor, sample_ray_dirs_tensor, ray_mask_tensor, vsize_np, ranges_np


    def w2pers(self, point_xyz_w, camrotc2w, campos):
        #     point_xyz_pers    B X M X 3
        xyz_w_shift = point_xyz_w - campos[:, None, :]
        xyz_c = torch.sum(xyz_w_shift[..., None,:] * torch.transpose(camrotc2w, 1, 2)[:, None, None,...], dim=-1)
        z_pers = xyz_c[..., 2]
        x_pers = xyz_c[..., 0] / xyz_c[..., 2]
        y_pers = xyz_c[..., 1] / xyz_c[..., 2]
        return torch.stack([x_pers, y_pers, z_pers], dim=-1)


    def build_cuda(self):

        mod = SourceModule(
            """
            #define KN  """ + str(self.opt.K)
            + """ 
            #include <cuda.h>
            #include <cuda_runtime.h>
            #include <algorithm>
            #include <vector>
            #include <stdio.h>
            #include <math.h>
            #include <stdlib.h>
            #include <curand_kernel.h>
            namespace cuda {          
    
                static __device__ inline uint8_t atomicAdd(uint8_t *address, uint8_t val) {
                    size_t offset = (size_t)address & 3;
                    uint32_t *address_as_ui = (uint32_t *)(address - offset);
                    uint32_t old = *address_as_ui;
                    uint32_t shift = offset * 8;
                    uint32_t old_byte;
                    uint32_t newval;
                    uint32_t assumed;
    
                    do {
                      assumed = old;
                      old_byte = (old >> shift) & 0xff;
                      // preserve size in initial cast. Casting directly to uint32_t pads
                      // negative signed values with 1's (e.g. signed -1 = unsigned ~0).
                      newval = static_cast<uint8_t>(val + old_byte);
                      newval = (old & ~(0x000000ff << shift)) | (newval << shift);
                      old = atomicCAS(address_as_ui, assumed, newval);
                    } while (assumed != old);
                    return __byte_perm(old, 0, offset);   // need validate
                }
    
                static __device__ inline char atomicAdd(char* address, char val) {
                    // offset, in bytes, of the char* address within the 32-bit address of the space that overlaps it
                    size_t long_address_modulo = (size_t) address & 3;
                    // the 32-bit address that overlaps the same memory
                    auto* base_address = (unsigned int*) ((char*) address - long_address_modulo);
                    // A 0x3210 selector in __byte_perm will simply select all four bytes in the first argument in the same order.
                    // The "4" signifies the position where the first byte of the second argument will end up in the output.
                    unsigned int selectors[] = {0x3214, 0x3240, 0x3410, 0x4210};
                    // for selecting bytes within a 32-bit chunk that correspond to the char* address (relative to base_address)
                    unsigned int selector = selectors[long_address_modulo];
                    unsigned int long_old, long_assumed, long_val, replacement;
    
                    long_old = *base_address;
    
                    do {
                        long_assumed = long_old;
                        // replace bits in long_old that pertain to the char address with those from val
                        long_val = __byte_perm(long_old, 0, long_address_modulo) + val;
                        replacement = __byte_perm(long_old, long_val, selector);
                        long_old = atomicCAS(base_address, long_assumed, replacement);
                    } while (long_old != long_assumed);
                    return __byte_perm(long_old, 0, long_address_modulo);
                }            
    
                static __device__ inline int8_t atomicAdd(int8_t *address, int8_t val) {
                    return (int8_t)cuda::atomicAdd((char*)address, (char)val);
                }
    
                static __device__ inline short atomicAdd(short* address, short val)
                {
    
                    unsigned int *base_address = (unsigned int *)((size_t)address & ~2);
    
                    unsigned int long_val = ((size_t)address & 2) ? ((unsigned int)val << 16) : (unsigned short)val;
    
                    unsigned int long_old = ::atomicAdd(base_address, long_val);
    
                    if((size_t)address & 2) {
                        return (short)(long_old >> 16);
                    } else {
    
                        unsigned int overflow = ((long_old & 0xffff) + long_val) & 0xffff0000;
    
                        if (overflow)
    
                            atomicSub(base_address, overflow);
    
                        return (short)(long_old & 0xffff);
                    }
                }
    
                static __device__ float cas(double *addr, double compare, double val) {
                   unsigned long long int *address_as_ull = (unsigned long long int *) addr;
                   return __longlong_as_double(atomicCAS(address_as_ull,
                                                 __double_as_longlong(compare),
                                                 __double_as_longlong(val)));
                }
    
                static __device__ float cas(float *addr, float compare, float val) {
                    unsigned int *address_as_uint = (unsigned int *) addr;
                    return __uint_as_float(atomicCAS(address_as_uint,
                                           __float_as_uint(compare),
                                           __float_as_uint(val)));
                }
    
    
    
                static __device__ inline uint8_t atomicCAS(uint8_t * const address, uint8_t const compare, uint8_t const value)
                {
                    uint8_t const longAddressModulo = reinterpret_cast< size_t >( address ) & 0x3;
                    uint32_t *const baseAddress  = reinterpret_cast< uint32_t * >( address - longAddressModulo );
                    uint32_t constexpr byteSelection[] = { 0x3214, 0x3240, 0x3410, 0x4210 }; // The byte position we work on is '4'.
                    uint32_t const byteSelector = byteSelection[ longAddressModulo ];
                    uint32_t const longCompare = compare;
                    uint32_t const longValue = value;
                    uint32_t longOldValue = * baseAddress;
                    uint32_t longAssumed;
                    uint8_t oldValue;
                    do {
                        // Select bytes from the old value and new value to construct a 32-bit value to use.
                        uint32_t const replacement = __byte_perm( longOldValue, longValue,   byteSelector );
                        uint32_t const comparison  = __byte_perm( longOldValue, longCompare, byteSelector );
    
                        longAssumed  = longOldValue;
                        // Use 32-bit atomicCAS() to try and set the 8-bits we care about.
                        longOldValue = ::atomicCAS( baseAddress, comparison, replacement );
                        // Grab the 8-bit portion we care about from the old value at address.
                        oldValue     = ( longOldValue >> ( 8 * longAddressModulo )) & 0xFF;
                    } while ( compare == oldValue and longAssumed != longOldValue ); // Repeat until other three 8-bit values stabilize.
                    return oldValue;
                }
            }
    
            extern "C" {
                __global__ void claim_occ(
                    const float* in_data,   // B * N * 3 #[1,4242263,3],点云input数据
                    const int* in_actual_numpoints, // B #value = 4242263
                    const int B,//1
                    const int N,//4244263
                    const float *d_coord_shift,     // 3#[3],values = [-1.6265191,-1.9573721,-3.291426]取的是整个boudingbox的三个最小值，作为体素场的原点，即体素场坐标皆为正数
                    const float *d_voxel_size,      // 3#[3],values = [0.016,0.016,0.016]:单个voxel的size
                    const int *d_grid_size,       // 3#[3],values = [344,377,357]:voxel‘s dim for every axises
                    const int grid_size_vol,    //#46298616 = 344*377*357 整个的voxel的数量
                    const int max_o, //#一次性取的最多的voxel数量
                    int* occ_idx, // B, all 0 , 
                    int *coor_2_occ,  // B * 400 * 400 * 400, all -1#outputs,[1,344,377,357]初始值都是-1,如果遍历到某一voxel变成0
                    int *occ_2_coor,  // B * max_o * 3, all -1 #outputs,[1,610000,3]遍历到的体素场的坐标
                    unsigned long seconds
                ) {
                    int index = blockIdx.x * blockDim.x + threadIdx.x; // index of gpu thread
                    int i_batch = index / N;  // index of batch
                    if (i_batch >= B) { return; }
                    int i_pt = index - N * i_batch;
                    if (i_pt < in_actual_numpoints[i_batch]) {
                        int coor[3];
                        const float *p_pt = in_data + index * 3;
                        //d_voxel_size:voxel格子的大小,scannet取的0.016
                        coor[0] = (int) floor((p_pt[0] - d_coord_shift[0]) / d_voxel_size[0]);
                        coor[1] = (int) floor((p_pt[1] - d_coord_shift[1]) / d_voxel_size[1]);
                        coor[2] = (int) floor((p_pt[2] - d_coord_shift[2]) / d_voxel_size[2]);
                        // printf("p_pt %f %f %f %f; ", p_pt[2], d_coord_shift[2], d_coord_shift[0], d_coord_shift[1]);
                        if (coor[0] < 0 || coor[0] >= d_grid_size[0] || coor[1] < 0 || coor[1] >= d_grid_size[1] || coor[2] < 0 || coor[2] >= d_grid_size[2]) { return; }
                        int coor_indx_b = i_batch * grid_size_vol + coor[0] * (d_grid_size[1] * d_grid_size[2]) + coor[1] * d_grid_size[2] + coor[2];
                        
                        int voxel_idx = coor_2_occ[coor_indx_b];//被占用的voxel value from -1 to 0
                        if (voxel_idx == -1) {  // found an empty voxel
                            int old_voxel_num = atomicCAS(
                                    &coor_2_occ[coor_indx_b],
                                    -1, 0
                            );
                            if (old_voxel_num == -1) {
                                // CAS -> old val, if old val is -1//如果之前是空的，证明该线程抢占到了这个voxel，只有这个线程可以做counter+1操作
                                // if we get -1, this thread is the one who obtain a new voxel
                                // so only this thread should do the increase operator below
                                int tmp = atomicAdd(occ_idx+i_batch, 1); // increase the counter, return old counter；occ_idx：目前已经处理的体素个数
                                 // increase the counter, return old counter,max_o：一次处理体素的上限
                                if (tmp < max_o) {
                                    int coord_inds = (i_batch * max_o + tmp) * 3;
                                    occ_2_coor[coord_inds] = coor[0];//记录要处理的体素的坐标
                                    occ_2_coor[coord_inds + 1] = coor[1];
                                    occ_2_coor[coord_inds + 2] = coor[2];
                                } else {
                                    curandState state;
                                    curand_init(index+2*seconds, 0, 0, &state);
                                    int insrtidx = ceilf(curand_uniform(&state) * (tmp+1)) - 1;
                                    if(insrtidx < max_o){
                                        int coord_inds = (i_batch * max_o + insrtidx) * 3;
                                        occ_2_coor[coord_inds] = coor[0];
                                        occ_2_coor[coord_inds + 1] = coor[1];
                                        occ_2_coor[coord_inds + 2] = coor[2];
                                    }
                                }
                            }
                        }
                    }
                }
                //对每一个体素场坐标，查看kernel_size内所有体素激活（coor_occ即置1，否则是0）,这些都是我们后面要搞的体素
                __global__ void map_coor2occ(
                    const int B, //1
                    const int *d_grid_size,       // 3 [3],values = [344 377 357]
                    const int *kernel_size,       // 3 [3],values = [3 3 3]
                    const int grid_size_vol,      // #46298616 = 344*377*357 整个的voxel的数量
                    const int max_o,                //#一次性取的最多的voxel数量
                    int* occ_idx, // B, all -1 #[1]初始值为遍历到的点云idx
                    int *coor_occ,  // B * 400 * 400 * 400 #[1,344,377,357]初始时皆为0
                    int *coor_2_occ,  // B * 400 * 400 * 400#[1,344,377,357]初始时皆为-1；存index？
                    int *occ_2_coor  // B * max_o * 3 #[1,610000,3]体素场坐标
                ) {
                    int index = blockIdx.x * blockDim.x + threadIdx.x; // index of gpu thread
                    int i_batch = index / max_o;  // index of batch
                    if (i_batch >= B) { return; }
                    int i_pt = index - max_o * i_batch;
                    if (i_pt < occ_idx[i_batch] && i_pt < max_o) {
                        int coor[3];//取出体素场坐标放入其中
                        coor[0] = occ_2_coor[index*3];
                        if (coor[0] < 0) { return; }
                        coor[1] = occ_2_coor[index*3+1];
                        coor[2] = occ_2_coor[index*3+2];
                        
                        int coor_indx_b = i_batch * grid_size_vol + coor[0] * (d_grid_size[1] * d_grid_size[2]) + coor[1] * d_grid_size[2] + coor[2];
                        coor_2_occ[coor_indx_b] = i_pt;
                        // printf("kernel_size[0] %d", kernel_size[0]);
                        for (int coor_x = max(0, coor[0] - kernel_size[0] / 2) ; coor_x < min(d_grid_size[0], coor[0] + (kernel_size[0] + 1) / 2); coor_x++)    {
                            for (int coor_y = max(0, coor[1] - kernel_size[1] / 2) ; coor_y < min(d_grid_size[1], coor[1] + (kernel_size[1] + 1) / 2); coor_y++)   {
                                for (int coor_z = max(0, coor[2] - kernel_size[2] / 2) ; coor_z < min(d_grid_size[2], coor[2] + (kernel_size[2] + 1) / 2); coor_z++) {
                                    coor_indx_b = i_batch * grid_size_vol + coor_x * (d_grid_size[1] * d_grid_size[2]) + coor_y * d_grid_size[2] + coor_z;
                                    if (coor_occ[coor_indx_b] > 0) { continue; }
                                    atomicCAS(coor_occ + coor_indx_b, 0, 1);
                                }
                            }
                        }   
                    }
                }
                
                __global__ void fill_occ2pnts(
                    const float* in_data,   // B * N * 3#[1,4242263,3],点云input数据
                    const int* in_actual_numpoints, // B 
                    const int B,//1
                    const int N,//4242263
                    const int P,// 26 超参
                    const float *d_coord_shift,     // 3#[-1.6265,-1.9573,-3.2914]
                    const float *d_voxel_size,      // 3#[0.0016 0.0016 0.0016]
                    const int *d_grid_size,       // 3# [344 377 357]
                    const int grid_size_vol,// 46298616 = 344*377*357
                    const int max_o,//# 610000
                    int *coor_2_occ,  // B * 400 * 400 * 400, all -1#[1,344,377,357]初始时皆为-1；存index的值
                    int *occ_2_pnts,  // B * max_o * P, all -1#[1,610000,26]未曾出现过，初始化皆为-1
                    int *occ_numpnts,  // B * max_o, all 0 #[1,610000]未曾出现过，初始化皆为 0 
                    unsigned long seconds
                ) {
                    int index = blockIdx.x * blockDim.x + threadIdx.x; // index of gpu thread
                    int i_batch = index / N;  // index of batch
                    if (i_batch >= B) { return; }
                    int i_pt = index - N * i_batch;
                    if (i_pt < in_actual_numpoints[i_batch]) {
                        int coor[3];
                        const float *p_pt = in_data + index * 3;
                        coor[0] = (int) floor((p_pt[0] - d_coord_shift[0]) / d_voxel_size[0]);//计算点云所在的体素场坐标
                        coor[1] = (int) floor((p_pt[1] - d_coord_shift[1]) / d_voxel_size[1]);
                        coor[2] = (int) floor((p_pt[2] - d_coord_shift[2]) / d_voxel_size[2]);
                        if (coor[0] < 0 || coor[0] >= d_grid_size[0] || coor[1] < 0 || coor[1] >= d_grid_size[1] || coor[2] < 0 || coor[2] >= d_grid_size[2]) { return; }
                        int coor_indx_b = i_batch * grid_size_vol + coor[0] * (d_grid_size[1] * d_grid_size[2]) + coor[1] * d_grid_size[2] + coor[2];
                        
                        int voxel_idx = coor_2_occ[coor_indx_b];
                        if (voxel_idx > 0) {  // found an claimed coor2occ
                            int occ_indx_b = i_batch * max_o + voxel_idx;
                            int tmp = atomicAdd(occ_numpnts + occ_indx_b, 1); // increase the counter, return old counter
                            if (tmp < P) {
                                occ_2_pnts[occ_indx_b * P + tmp] = i_pt;
                            } else {
                                curandState state;
                                curand_init(index+2*seconds, 0, 0, &state);
                                int insrtidx = ceilf(curand_uniform(&state) * (tmp+1)) - 1;
                                if(insrtidx < P){
                                    occ_2_pnts[occ_indx_b * P + insrtidx] = i_pt;
                                }
                            }
                        }
                    }
                }
                
                            
                __global__ void mask_raypos(
                    float *raypos,    // # [1, 784, 400, 3]要query的点的世界坐标
                    int *coor_occ,    // # [1, 344, 377, 357]算是一个mask 1 or 0，这个体素场如果在kernelsize范围内，则1, else 0
                    const int B,       // # 1
                    const int R,       // 784
                    const int D,       // # 400
                    const int grid_size_vol,//# 46298616
                    const float *d_coord_shift,     // # [-1.6265191, -1.9573721 -3.291426]
                    const int *d_grid_size,       // #[344 377 357]
                    const float *d_voxel_size,      // # [0.016 0.016 0.016]
                    int *raypos_mask    // #[1,784,400];all 0 init
                ) {
                    int index = blockIdx.x * blockDim.x + threadIdx.x; // index of gpu thread
                    int i_batch = index / (R * D);  // index of batch
                    if (i_batch >= B) { return; }
                    int coor[3];
                    coor[0] = (int) floor((raypos[index*3] - d_coord_shift[0]) / d_voxel_size[0]);
                    coor[1] = (int) floor((raypos[index*3+1] - d_coord_shift[1]) / d_voxel_size[1]);
                    coor[2] = (int) floor((raypos[index*3+2] - d_coord_shift[2]) / d_voxel_size[2]);
                    // printf(" %f %f %f;", raypos[index*3], raypos[index*3+1], raypos[index*3+2]);
                    if ((coor[0] >= 0) && (coor[0] < d_grid_size[0]) && (coor[1] >= 0) && (coor[1] < d_grid_size[1]) && (coor[2] >= 0) && (coor[2] < d_grid_size[2])) { 
                        int coor_indx_b = i_batch * grid_size_vol + coor[0] * (d_grid_size[1] * d_grid_size[2]) + coor[1] * d_grid_size[2] + coor[2];
                        raypos_mask[index] = coor_occ[coor_indx_b];
                    }
                }
                
        
                __global__ void get_shadingloc(
                    const float *raypos,    // # [1, 784, 400, 3]要采样要query的点的世界坐标
                    const int *raypos_mask,    // #[1,784,400],沿光线这是第几个被采样的点-1 -1 0 1 -1 -1 1 2 3 4 5 -1 -1 -1 ...
                    const int B,       // 1
                    const int R,       // 784
                    const int D,       // 400
                    const int SR,       // 24 一条ray一次性最多采样的点数
                    float *sample_loc,       // #[1,784,24,3],init:all-0，存某个pixel需要采样的点的坐标
                    int *sample_loc_mask       // [1,784,24],init:all-0 sample_loc的msk
                ) {
                    int index = blockIdx.x * blockDim.x + threadIdx.x; // index of gpu thread
                    int i_batch = index / (R * D);  // index of batch
                    if (i_batch >= B) { return; }
                    int temp = raypos_mask[index]; // temp：mask的值，raypos_mask：#[1,784,400],沿光线这是第几个被采样的点-1 -1 0 1 -1 -1 1 2 3 4 5 -1 -1 -1 ...
                    if (temp >= 0) {
                        int r = (index - i_batch * R * D) / D; // deside which ray(0~783)
                        int loc_inds = i_batch * R * SR + r * SR + temp;
                        sample_loc[loc_inds * 3] = raypos[index * 3];
                        sample_loc[loc_inds * 3 + 1] = raypos[index * 3 + 1];
                        sample_loc[loc_inds * 3 + 2] = raypos[index * 3 + 2];
                        sample_loc_mask[loc_inds] = 1;
                    }
                }
                
                
                __global__ void query_neigh_along_ray_layered(
                    const float* in_data,   // #point cloud data input:[1,4242263,3]
                    const int B,            //1
                    const int SR,               // num. samples along each ray 24
                    const int R,               // e.g., 784
                    const int max_o,            // 610000
                    const int P,                //26 超参；P：一个体素中最多的点云数量
                    const int K,                // max num.  neighbors
                    const int grid_size_vol,    //46298616 = a*b*c
                    const float radius_limit2,  //radius_limit_np = 0.032  radius_limit2 = 0.032**2
                    const float *d_coord_shift,     // 3[-1.6265191,-1.9573721,-3.291426]
                    const int *d_grid_size,         //[344 377 357]
                    const float *d_voxel_size,      // 3[0.016 0.016 0.016]
                    const int *kernel_size,         // [3 3 3]
                    const int *occ_numpnts,    // B * max_o = 610000
                    const int *occ_2_pnts,            // B * max_o * P [1,610000,26]某个occ中每个点云的index
                    const int *coor_2_occ,      // B * 400 * 400 * 400  [1,344,377,357]：init=-1；存放的是occ的index(0~610000)
                    const float *sample_loc,       // B * R * SR * 3 [1,784,24,3],init:all-0，存某个pixel需要采样的点的坐标
                    const int *sample_loc_mask,       // B * R * SR  [1,784,24],init:all-0，存sample_loc_tensor的msk
                    int *sample_pidx,       // B * R * SR * K [1,784,24,8]init-all -1;8，即K，一个查询点的max num.  neighbors；返回每个像素（784个）所要query的每个点（24个）的真实邻居点（最临近的8点）的pid
                    unsigned long seconds,
                    const int NN
                ) {
                    int index =  blockIdx.x * blockDim.x + threadIdx.x; // index of gpu thread
                    int i_batch = index / (R * SR);  // index of batch
                    if (i_batch >= B || sample_loc_mask[index] <= 0) { return; }
                    float centerx = sample_loc[index * 3];
                    float centery = sample_loc[index * 3 + 1];
                    float centerz = sample_loc[index * 3 + 2];
                    int frustx = (int) floor((centerx - d_coord_shift[0]) / d_voxel_size[0]);//这个点云所在的体素场的坐标frustx,y,z
                    int frusty = (int) floor((centery - d_coord_shift[1]) / d_voxel_size[1]);
                    int frustz = (int) floor((centerz - d_coord_shift[2]) / d_voxel_size[2]);
                                        
                    centerx = sample_loc[index * 3];
                    centery = sample_loc[index * 3 + 1];
                    centerz = sample_loc[index * 3 + 2];
                                        
                    int kid = 0, far_ind = 0, coor_z, coor_y, coor_x;
                    float far2 = 0.0;
                    float xyz2Buffer[KN];
                    //按照kernel_size在occ内遍历[344 377 357]
                    for (int layer = 0; layer < (kernel_size[0]+1)/2; layer++){                        
                        for (int x = max(-frustx, -layer); x < min(d_grid_size[0] - frustx, layer + 1); x++) {
                            coor_x = frustx + x;
                            for (int y = max(-frusty, -layer); y < min(d_grid_size[1] - frusty, layer + 1); y++) {
                                coor_y = frusty + y;
                                for (int z =  max(-frustz, -layer); z < min(d_grid_size[2] - frustz, layer + 1); z++) {
                                    coor_z = z + frustz;
                                    if (max(abs(z), max(abs(x), abs(y))) != layer) continue;
                                    int coor_indx_b = i_batch * grid_size_vol + coor_x * (d_grid_size[1] * d_grid_size[2]) + coor_y * d_grid_size[2] + coor_z;
                                    int occ_indx = coor_2_occ[coor_indx_b] + i_batch * max_o;//[1,344,377,357]，从中取出occ的index(0-610000)
                                    if (occ_indx >= 0) {//if occ_indx = -1则里面没点
                                        for (int g = 0; g < min(P, occ_numpnts[occ_indx]); g++) {//occ_numpnts:某个occ中点云的数量，P超参：最大的点云数量
                                            int pidx = occ_2_pnts[occ_indx * P + g];//occ_2_pnts[1,610000,26]，从每个occ中取出每个点的index
                                            float x_v = (in_data[pidx*3]-centerx);//in_data[pidx*3]：点云坐标；centerx:query点的坐标
                                            float y_v = (in_data[pidx*3 + 1]-centery);
                                            float z_v = (in_data[pidx*3 + 2]-centerz);
                                            float xyz2 = x_v * x_v + y_v * y_v + z_v * z_v;//点到query点距离**2
                                            if ((radius_limit2 == 0 || xyz2 <= radius_limit2)){//如果是在radius_limit2的范围内
                                                if (kid++ < K) {//K:max num.  neighbors;kid:current num neighbors
                                                    sample_pidx[index * K + kid - 1] = pidx;//sample_pidx存相应的pidx
                                                    xyz2Buffer[kid-1] = xyz2;//缓存xyz距离
                                                    if (xyz2 > far2){
                                                        far2 = xyz2;//存储最远点的距离和index
                                                        far_ind = kid - 1;
                                                    }
                                                } else {//如果已经采集满了，去替换掉最远的那个点，即这8个点已经是离query点最近的点
                                                    if (xyz2 < far2) {
                                                        sample_pidx[index * K + far_ind] = pidx;
                                                        xyz2Buffer[far_ind] = xyz2;
                                                        far2 = xyz2;
                                                        for (int i = 0; i < K; i++) {
                                                            if (xyz2Buffer[i] > far2) {
                                                                far2 = xyz2Buffer[i];
                                                                far_ind = i;
                                                            }
                                                        }
                                                    } 
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        if (kid >= K) break;
                    }
                }
            }
        """, no_extern_c=True)
        claim_occ = mod.get_function("claim_occ")
        map_coor2occ = mod.get_function("map_coor2occ")
        fill_occ2pnts = mod.get_function("fill_occ2pnts")
        mask_raypos = mod.get_function("mask_raypos")
        get_shadingloc = mod.get_function("get_shadingloc")
        query_along_ray = mod.get_function("query_neigh_along_ray_layered") if self.opt.NN > 0 else mod.get_function("query_rand_along_ray")
        return claim_occ, map_coor2occ, fill_occ2pnts, mask_raypos, get_shadingloc, query_along_ray


    def switch_pixel_id(self, pixel_idx_tensor, h):
        pixel_id = torch.cat([pixel_idx_tensor[..., 0:1], h - 1 - pixel_idx_tensor[..., 1:2]], dim=-1)
        # print("pixel_id", pixel_id.shape, torch.min(pixel_id, dim=-2)[0], torch.max(pixel_id, dim=-2)[0])
        return pixel_id


    def build_occ_vox(self, point_xyz_w_tensor, actual_numpoints_tensor, B, N, P, max_o, scaled_vdim_np, kMaxThreadsPerBlock, gridSize, scaled_vsize_gpu, scaled_vdim_gpu, kernel_size_gpu, grid_size_vol, d_coord_shift):
        #scaled_vdim_np：体素场的dim，[334,377,357]
        device = point_xyz_w_tensor.device
        coor_occ_tensor = torch.zeros([B, scaled_vdim_np[0], scaled_vdim_np[1], scaled_vdim_np[2]], dtype=torch.int32, device=device)#scaled_vdim_np:[B，344,377,357]
        occ_2_pnts_tensor = torch.full([B, max_o, P], -1, dtype=torch.int32, device=device)#max_o:610000,P:2
        occ_2_coor_tensor = torch.full([B, max_o, 3], -1, dtype=torch.int32, device=device)
        occ_numpnts_tensor = torch.zeros([B, max_o], dtype=torch.int32, device=device)
        coor_2_occ_tensor = torch.full([B, scaled_vdim_np[0], scaled_vdim_np[1], scaled_vdim_np[2]], -1, dtype=torch.int32, device=device)
        occ_idx_tensor = torch.zeros([B], dtype=torch.int32, device=device)
        seconds = time.time()
        '''
        对站位场进行初始化
        '''
        self.claim_occ(
            Holder(point_xyz_w_tensor),#[1,4242263,3],点云input数据
            Holder(actual_numpoints_tensor),#value = 4242263
            np.int32(B),#1
            np.int32(N),#4242263
            d_coord_shift,#[3],values = [-1.6265191,-1.9573721,-3.291426]取的是整个boudingbox的三个最小值，作为体素场的原点，即体素场坐标皆为正数
            scaled_vsize_gpu,#[3],values = [0.016,0.016,0.016]:单个voxel的size
            scaled_vdim_gpu,#[3],values = [344,377,357]:voxel‘s dim for every axises
            np.int32(grid_size_vol),#46298616 = 344*377*357 整个的voxel的数量
            np.int32(max_o),#一次性取的最多的voxel数量
            Holder(occ_idx_tensor),#outputs, B ,初始值都为0 , values = [599947]，目前正在遍历的idx
            Holder(coor_2_occ_tensor),#outputs,[1,344,377,357]初始值都是-1,如果遍历到某一voxel变成0
            Holder(occ_2_coor_tensor),#outputs,[1,610000,3]初始值皆为-1，而后变成遍历到的体素场的坐标
            np.uint64(seconds),
            block=(kMaxThreadsPerBlock, 1, 1), grid=(gridSize, 1))
        # torch.cuda.synchronize()
        coor_2_occ_tensor = torch.full([B, scaled_vdim_np[0], scaled_vdim_np[1], scaled_vdim_np[2]], -1,
                                       dtype=torch.int32, device=device)
        # 对每一个体素场坐标，查看kernel_size内所有体素激活（coor_occ即置1，否则是0）, 这些都是我们后面要搞的体素
        gridSize = int((B * max_o + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock)#4143
        self.map_coor2occ(
            np.int32(B),#1
            scaled_vdim_gpu,#[3],values = [344 377 357]
            kernel_size_gpu,#[3],values = [3 3 3],超参，搜索的半径范围
            np.int32(grid_size_vol),#46298616 = 344*377*357 整个的voxel的数量
            np.int32(max_o),#一次性取的最多的voxel数量
            Holder(occ_idx_tensor),#[1]初始值为遍历到的点云idx
            Holder(coor_occ_tensor),#[1,344,377,357]初始时皆为0,如果是在体素场坐标(occ_2_coor)的kernel_size内的值，则置1
            Holder(coor_2_occ_tensor),#[1,344,377,357]初始时皆为-1；之后存的index值,采样点的索引
            Holder(occ_2_coor_tensor),#[1,610000,3]体素场坐标，此函数中未改变
            block=(kMaxThreadsPerBlock, 1, 1), grid=(gridSize, 1))
        # torch.cuda.synchronize()
        seconds = time.time()
        gridSize = int((B * N + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock)#4143
        #目前推测该函数：遍历每个点云(point_xyz_w_tensor)；我们之前采样了max_o个体素，把每个occ里面的点云
        self.fill_occ2pnts(
            Holder(point_xyz_w_tensor),#[1,4242263,3],点云input数据
            Holder(actual_numpoints_tensor),#values = [4242263]
            np.int32(B),# 1
            np.int32(N),# 4242263
            np.int32(P),#26 超参；P：一个体素中最多的点云数量
            d_coord_shift,#[-1.6265,-1.9573,-3.2914]
            scaled_vsize_gpu,#[0.0016 0.0016 0.0016]
            scaled_vdim_gpu,# [344 377 357]
            np.int32(grid_size_vol),# 46298616 = 344*377*357
            np.int32(max_o), # 610000
            Holder(coor_2_occ_tensor),#[1,344,377,357]初始时皆为-1；看起来应该存的index值？
            Holder(occ_2_pnts_tensor),#[1,610000,26]未曾出现过，初始化皆为-1;
            Holder(occ_numpnts_tensor),#[1,610000]未曾出现过，初始化皆为 0;
            np.uint64(seconds),
            block=(kMaxThreadsPerBlock, 1, 1), grid=(gridSize, 1))
        '''
        coor_occ_tensor:torch.Size([1, 344, 377, 357]) :1 or 0，这个体素场如果在kernelsize范围内，则1, else 0
        occ_2_coor_tensor:torch.Size([1, 610000, 3]) max_o个采样(OCC)的体素场的坐标
        coor_2_occ_tensor:torch.Size([1, 344, 377, 357]) ：init=-1；存放的是occ的index
        occ_idx_tensor:torch.Size([1]) 
        occ_numpnts_tensor:torch.Size([1, 610000]) #某个occ中点云数量
        occ_2_pnts_tensor:torch.Size([1, 610000, 26])#某个occ中每个点云的index
        '''
        return coor_occ_tensor, occ_2_coor_tensor, coor_2_occ_tensor, occ_idx_tensor, occ_numpnts_tensor, occ_2_pnts_tensor


    def query_grid_point_index(self, h, w, pixel_idx_tensor, raypos_tensor, point_xyz_w_tensor, actual_numpoints_tensor, kernel_size_gpu, query_size_gpu, SR, K, ranges_np, scaled_vsize_np, scaled_vdim_np, vscale_np, max_o, P, radius_limit_np, depth_limit_np, ranges_gpu, scaled_vsize_gpu, scaled_vdim_gpu, vscale_gpu, ray_dirs_tensor, cam_pos_tensor, kMaxThreadsPerBlock = 1024):
        #h, w, pixel_idx_tensor, raypos_tensor, point_xyz_w_tensor, actual_numpoints_tensor, kernel_size_gpu, query_size_gpu, self.opt.SR, self.opt.K, ranges_np, scaled_vsize_np, scaled_vdim_np, vscale_np, self.opt.max_o, self.opt.P, radius_limit_np, depth_limit_np, range_gpu, scaled_vsize_gpu, scaled_vdim_gpu, vscale_gpu, ray_dirs_tensor, cam_pos_tensor, kMaxThreadsPerBlock=self.opt.gpu_maxthr
        #scaled_vdim_np:[344,377,357]
        #raypos_tensor[1,784,400,3]<---->将28*28个像素坐标转化成了camera坐标系下的3D坐标;28*28=784
        #scaled_vsize_gpu[3]:volume的size
        #scaled_vdim_gpu[3]:volumn的维度
        #d_coord_shift：体素场的中心点（体素场的坐标全为正数，相当于是range[:3]的坐标）
        device = point_xyz_w_tensor.device
        B, N = point_xyz_w_tensor.shape[0], point_xyz_w_tensor.shape[1]#batch,n(num of point,4242266)
        pixel_size = scaled_vdim_np[0] * scaled_vdim_np[1]
        grid_size_vol = pixel_size * scaled_vdim_np[2]#总体素数量
        d_coord_shift = ranges_gpu[:3]
        R, D = raypos_tensor.shape[1], raypos_tensor.shape[2]#R:784个pixel;D:400-sample数量
        R = pixel_idx_tensor.reshape(B, -1, 2).shape[1]
        gridSize = int((B * N + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock)
        coor_occ_tensor, occ_2_coor_tensor, coor_2_occ_tensor, occ_idx_tensor, occ_numpnts_tensor, occ_2_pnts_tensor = self.build_occ_vox(point_xyz_w_tensor, actual_numpoints_tensor, B, N, P, max_o, scaled_vdim_np, kMaxThreadsPerBlock, gridSize, scaled_vsize_gpu, scaled_vdim_gpu, query_size_gpu, grid_size_vol, d_coord_shift)
        '''
        coor_occ_tensor:torch.Size([1, 344, 377, 357]) :1 or 0，这个体素场如果在kernelsize范围内，则1, else 0
        occ_2_coor_tensor:torch.Size([1, 610000, 3]) max_o个采样(OCC)的体素场的坐标
        coor_2_occ_tensor:torch.Size([1, 344, 377, 357]) ：init=-1；存放的是occ的index(0~610000)
        occ_idx_tensor:torch.Size([1]) 
        occ_numpnts_tensor:torch.Size([1, 610000]) #某个occ中点云数量
        occ_2_pnts_tensor:torch.Size([1, 610000, 26])#某个occ中每个点云的index
        '''
        # torch.cuda.synchronize()
        # print("coor_occ_tensor", torch.min(coor_occ_tensor), torch.max(coor_occ_tensor), torch.min(occ_2_coor_tensor), torch.max(occ_2_coor_tensor), torch.min(coor_2_occ_tensor), torch.max(coor_2_occ_tensor), torch.min(occ_idx_tensor), torch.max(occ_idx_tensor), torch.min(occ_numpnts_tensor), torch.max(occ_numpnts_tensor), torch.min(occ_2_pnts_tensor), torch.max(occ_2_pnts_tensor), occ_2_pnts_tensor.shape)
        # print("occ_numpnts_tensor", torch.sum(occ_numpnts_tensor > 0), ranges_np)
        # vis_vox(ranges_np, scaled_vsize_np, coor_2_occ_tensor)

        raypos_mask_tensor = torch.zeros([B, R, D], dtype=torch.int32, device=device)#[1,784,400];all 0 init 784采样的像素数，400每个ray中sample的点数
        gridSize = int((B * R * D + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock)
        #mask一下，哪些附近是有点的，哪些是没点的
        self.mask_raypos(
            Holder(raypos_tensor),  # [1, 784, 400, 3]要query的点的世界坐标
            Holder(coor_occ_tensor),  # [1, 344, 377, 357]算是一个mask 1 or 0，这个体素场如果在kernelsize范围内，则1, else 0
            np.int32(B),# 1
            np.int32(R),# 784
            np.int32(D),# 400
            np.int32(grid_size_vol),# 46298616
            d_coord_shift,# [-1.6265191, -1.9573721 -3.291426]
            scaled_vdim_gpu, #[344 377 357]
            scaled_vsize_gpu,# [0.016 0.016 0.016]
            Holder(raypos_mask_tensor),#[1,784,400];all 0 init
            block=(kMaxThreadsPerBlock, 1, 1), grid=(gridSize, 1)
        )
        # torch.cuda.synchronize()
        # print("raypos_mask_tensor", raypos_mask_tensor.shape, torch.sum(coor_occ_tensor), torch.sum(raypos_mask_tensor))
        # save_points(raypos_tensor.reshape(-1, 3), "./", "rawraypos_pnts")
        # raypos_masked = torch.masked_select(raypos_tensor, raypos_mask_tensor[..., None] > 0)
        # save_points(raypos_masked.reshape(-1, 3), "./", "raypos_pnts")
        #如果有的射线上sample到的全是空白，则没必要要了
        ray_mask_tensor = torch.max(raypos_mask_tensor, dim=-1)[0] > 0 # B, R
        R = torch.max(torch.sum(ray_mask_tensor.to(torch.int32))).cpu().numpy()
        sample_loc_tensor = torch.zeros([B, R, SR, 3], dtype=torch.float32, device=device)#[1,784,24,3]
        sample_pidx_tensor = torch.full([B, R, SR, K], -1, dtype=torch.int32, device=device)#[1,784,24,8]
        if R > 0:#True
            raypos_tensor = torch.masked_select(raypos_tensor, ray_mask_tensor[..., None, None].expand(-1, -1, D, 3)).reshape(B, R, D, 3)
            raypos_mask_tensor = torch.masked_select(raypos_mask_tensor, ray_mask_tensor[..., None].expand(-1, -1, D)).reshape(B, R, D)
            # print("R", R, raypos_tensor.shape, raypos_mask_tensor.shape)
            #raypos_mask_tensor：[1,784,400] raypos_maskcum[1,784,400];对每个ray的采样点的msk累加，截止到这个采样点，这个ray之前有多少需要处理的点
            raypos_maskcum = torch.cumsum(raypos_mask_tensor, dim=-1).to(torch.int32)#[1,784,400],按ray做mask的累加
            raypos_mask_tensor = (raypos_mask_tensor * raypos_maskcum * (raypos_maskcum <= SR)) - 1#-1 -1 0 1 -1 -1 0 1 2 3 4 5 -1 -1 ...SR，一条ray一次性最多采样的点数
            sample_loc_mask_tensor = torch.zeros([B, R, SR], dtype=torch.int32, device=device)#[1,784,24]
            self.get_shadingloc(
                Holder(raypos_tensor),  # [1, 784, 400, 3]要采样要query的点的世界坐标
                Holder(raypos_mask_tensor),#[1,784,400],沿光线这是第几个被采样的点-1 -1 0 1 -1 -1 1 2 3 4 5 -1 -1 -1 ...
                np.int32(B),# 1
                np.int32(R),#784
                np.int32(D),# 400
                np.int32(SR),# 24SR，一条ray一次性最多采样的点数
                Holder(sample_loc_tensor),#[1,784,24,3],init:all-0，存某个pixel需要采样的点的坐标
                Holder(sample_loc_mask_tensor),#[1,784,24],init:all-0，存sample_loc_tensor的msk
                block=(kMaxThreadsPerBlock, 1, 1), grid=(gridSize, 1)
            )

            # torch.cuda.synchronize()
            # print("shadingloc_mask_tensor", torch.sum(sample_loc_mask_tensor, dim=-1), torch.sum(torch.sum(sample_loc_mask_tensor, dim=-1) > 0), torch.sum(sample_loc_mask_tensor > 0))
            # shadingloc_masked = torch.masked_select(sample_loc_tensor, sample_loc_mask_tensor[..., None] > 0)
            # save_points(shadingloc_masked.reshape(-1, 3), "./", "shading_pnts{}".format(self.count))

            seconds = time.time()
            gridSize = int((B * R * SR + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock)
            self.query_along_ray(
                Holder(point_xyz_w_tensor),#point cloud data input:[1,4242263,3]
                np.int32(B),# 1
                np.int32(SR),# 24
                np.int32(R),# 784
                np.int32(max_o),#610000
                np.int32(P),#26 超参；P：一个体素中最多的点云数量
                np.int32(K),# 8:num.  neighbors
                np.int32(grid_size_vol),#46298616 = a*b*c
                np.float32(radius_limit_np ** 2),#radius_limit_np = 0.032
                d_coord_shift,#[-1.6265191,-1.9573721,-3.291426]
                scaled_vdim_gpu,#[344 377 357]
                scaled_vsize_gpu,#[0.016 0.016 0.016]
                kernel_size_gpu,#[3 3 3]
                Holder(occ_numpnts_tensor),#[1,610000]，某个occ中pnts数量
                Holder(occ_2_pnts_tensor),#[1,610000,26]某个occ中每个点云的index
                Holder(coor_2_occ_tensor),#[1,344,377,357]：init=-1；存放的是occ的index(0~610000)
                Holder(sample_loc_tensor),#[1,784,24,3],init:all-0，存某个pixel需要query的点的坐标
                Holder(sample_loc_mask_tensor),#[1,784,24],init:all-0，存sample_loc_tensor的msk
                Holder(sample_pidx_tensor),#B * R * SR * K [1,784,24,8]init-all -1;8，即K，一个查询点的max num.  neighbors；返回每个像素（784个）所要query的每个点（24个）的真实邻居点（最临近的8点）的pid
                np.uint64(seconds),
                np.int32(self.opt.NN),
                block=(kMaxThreadsPerBlock, 1, 1), grid=(gridSize, 1))
            # torch.cuda.synchronize()
            # print("point_xyz_w_tensor",point_xyz_w_tensor.shape)
            # queried_masked = point_xyz_w_tensor[0][sample_pidx_tensor.reshape(-1).to(torch.int64), :]
            # save_points(queried_masked.reshape(-1, 3), "./", "queried_pnts{}".format(self.count))
            # print("valid ray",  torch.sum(torch.sum(sample_loc_mask_tensor, dim=-1) > 0))
            masked_valid_ray = torch.sum(sample_pidx_tensor.view(B, R, -1) >= 0, dim=-1) > 0
            #[1,784]True;masked_valid_ray存完全没有点的ray，即不需要参加渲染的
            R = torch.max(torch.sum(masked_valid_ray.to(torch.int32), dim=-1)).cpu().numpy()
            #[784]R means ray
            ray_mask_tensor.masked_scatter_(ray_mask_tensor, masked_valid_ray)#ray_mask_tensor[1,784]-True.all()
            sample_pidx_tensor = torch.masked_select(sample_pidx_tensor, masked_valid_ray[..., None, None].expand(-1, -1, SR, K)).reshape(B, R, SR, K)
            sample_loc_tensor = torch.masked_select(sample_loc_tensor, masked_valid_ray[..., None, None].expand(-1, -1, SR, 3)).reshape(B, R, SR, 3)
        #sample_pidx_tensor[1,784,24,8]每个像素(784)，需要采样的每个query点(24)的点云中临近8点
        #sample_loc_tensor[1,784,24,3],init:all-0，存某个pixel需要query的点的坐标
        #ray_mask_tensor[1,784]true or false，存放不需要采集的像素的msk
        return sample_pidx_tensor, sample_loc_tensor, ray_mask_tensor.to(torch.int8)
def load_pnts(point_path, point_num):
    with open(point_path, 'rb') as f:
        print("point_file_path################", point_path)
        all_infos = pickle.load(f)
        point_xyz = all_infos["point_xyz"]
    print(len(point_xyz), point_xyz.dtype, np.mean(point_xyz, axis=0), np.min(point_xyz, axis=0),
          np.max(point_xyz, axis=0))
    np.random.shuffle(point_xyz)
    return point_xyz[:min(len(point_xyz), point_num), :]


def np_to_gpuarray(*args):
    result = []
    for x in args:
        if isinstance(x, np.ndarray):
            result.append(pycuda.gpuarray.to_gpu(x))
        else:
            print("trans",x)
    return result


def save_points(xyz, dir, filename):
    if xyz.ndim < 3:
        xyz = xyz[None, ...]
    filename = "{}.txt".format(filename)
    os.makedirs(dir, exist_ok=True)
    filepath = os.path.join(dir, filename)
    print("save at {}".format(filepath))
    if torch.is_tensor(xyz):
        np.savetxt(filepath, xyz.cpu().reshape(-1, xyz.shape[-1]), delimiter=";")
    else:
        np.savetxt(filepath, xyz.reshape(-1, xyz.shape[-1]), delimiter=";")



def try_build(ranges, vsize, vdim, vscale, max_o, P, kernel_size, SR, K, pixel_idx, obj,
              radius_limit, depth_limit, near_depth, far_depth, shading_count, split=["train"], imgidx=0, gpu=0, NN=2):
    # point_path = os.path.join(point_dir, point_file)
    # point_xyz = load_pnts(point_path, 819200000)  # 81920   233872
    point_xyz = load_init_points(obj)
    imgs, poses, _, hwf, _, intrinsic = load_blender_data(
        os.path.expandvars("${nrDataRoot}") + "/nerf/nerf_synthetic/{}".format(obj), split, half_res=False, testskip=1)
    H, W, focal = hwf
    intrinsic =  np.array([[focal, 0, W / 2], [0, focal, H / 2], [0, 0, 1]])
    plt.figure()
    plt.imshow(imgs[imgidx])
    point_xyz_w_tensor = torch.as_tensor(point_xyz, device="cuda:{}".format(gpu))[None,...]
    print("point_xyz_w_tensor", point_xyz_w_tensor[0].shape, torch.min(point_xyz_w_tensor[0], dim=0)[0], torch.max(point_xyz_w_tensor[0], dim=0)[0])
    # plt.show()
    actual_numpoints_tensor = torch.ones([1], device=point_xyz_w_tensor.device, dtype=torch.int32) * len(point_xyz_w_tensor[0])
    # range_gpu, vsize_gpu, vdim_gpu, vscale_gpu, kernel_size_gpu = np_to_gpuarray(ranges, scaled_vsize, scaled_vdim, vscale, kernel_size)
    pixel_idx_tensor = torch.as_tensor(pixel_idx, device=point_xyz_w_tensor.device, dtype=torch.int32)[None, ...]
    c2w = poses[0]
    print("c2w", c2w.shape, pixel_idx.shape)
    from data.data_utils import get_dtu_raydir
    cam_pos, camrot = c2w[:3, 3], c2w[:3, :3]
    ray_dirs_tensor, cam_pos_tensor = torch.as_tensor(get_dtu_raydir(pixel_idx, intrinsic, camrot, True), device=pixel_idx_tensor.device, dtype=torch.float32), torch.as_tensor(cam_pos, device=pixel_idx_tensor.device, dtype=torch.float32)

    from collections import namedtuple
    opt_construct = namedtuple('opt', 'inverse vsize vscale kernel_size radius_limit_scale depth_limit_scale max_o P SR K gpu_maxthr NN ranges z_depth_dim')
    opt = opt_construct(inverse=0, vscale=vscale, vsize=vsize, kernel_size=kernel_size, radius_limit_scale=0, depth_limit_scale=0, max_o=max_o, P=P, SR=SR, K=K, gpu_maxthr=1024, NN=NN, ranges=ranges, z_depth_dim=400)

    querier = lighting_fast_querier(point_xyz_w_tensor.device, opt)
    print("actual_numpoints_tensor", actual_numpoints_tensor)
    querier.query_points(pixel_idx_tensor, None, point_xyz_w_tensor, actual_numpoints_tensor, H, W, intrinsic, near_depth, far_depth, ray_dirs_tensor[None, ...], cam_pos_tensor[None, ...])



def w2img(point_xyz, transform_matrix, focal):
    camrot = transform_matrix[:3, :3]  # world 2 cam
    campos = transform_matrix[:3, 3]  #
    point_xyz_shift = point_xyz - campos[None, :]
    # xyz = np.sum(point_xyz_shift[:,None,:] * camrot.T, axis=-1)
    xyz = np.sum(camrot[None, ...] * point_xyz_shift[:, :, None], axis=-2)
    # print(xyz.shape, np.sum(camrot[None, None, ...] * point_xyz_shift[:,:,None], axis=-2).shape)
    xper = xyz[:, 0] / -xyz[:, 2]
    yper = xyz[:, 1] / xyz[:, 2]
    x_pixel = np.round(xper * focal + 400).astype(np.int32)
    y_pixel = np.round(yper * focal + 400).astype(np.int32)
    print("focal", focal, np.tan(.5 * 0.6911112070083618))
    print("pixel xmax xmin:", np.max(x_pixel), np.min(x_pixel), "pixel ymax ymin:", np.max(y_pixel), np.min(y_pixel))
    print("per xmax xmin:", np.max(xper), np.min(xper), "per ymax ymin:", np.max(yper), np.min(yper), "per zmax zmin:",
          np.max(xyz[:, 2]), np.min(xyz[:, 2]))
    print("min perx", -400 / focal, "max perx", 400 / focal)
    background = np.ones([800, 800, 3], dtype=np.float32)
    background[y_pixel, x_pixel, :] = .2

    plt.figure()
    plt.imshow(background)

    return np.stack([xper, yper, -xyz[:, 2]], axis=-1)


def render_mask_pers_points(queried_point_xyz, vsize, ranges, w, h):
    pixel_xy_inds = np.floor((queried_point_xyz[:, :2] - ranges[None, :2]) / vsize[None, :2]).astype(np.int32)
    print(pixel_xy_inds.shape)
    y_pixel, x_pixel = pixel_xy_inds[:, 1], pixel_xy_inds[:, 0]
    background = np.ones([h, w, 3], dtype=np.float32)
    background[y_pixel, x_pixel, :] = .5
    plt.figure()
    plt.imshow(background)


def save_mask_pers_points(queried_point_xyz, vsize, ranges, w, h):
    pixel_xy_inds = np.floor((queried_point_xyz[:, :2] - ranges[None, :2]) / vsize[None, :2]).astype(np.int32)
    print(pixel_xy_inds.shape)
    y_pixel, x_pixel = pixel_xy_inds[:, 1], pixel_xy_inds[:, 0]
    background = np.ones([h, w, 3], dtype=np.float32)
    background[y_pixel, x_pixel, :] = .5
    image_dir = os.path.join(self.opt.checkpoints_dir, opt.name, 'images')
    image_file = os.path.join(image_dir)


def render_pixel_mask(pixel_xy_inds, w, h):
    y_pixel, x_pixel = pixel_xy_inds[0, :, 1], pixel_xy_inds[0, :, 0]
    background = np.ones([h, w, 3], dtype=np.float32)
    background[y_pixel, x_pixel, :] = .0
    plt.figure()
    plt.imshow(background)

def vis_vox(ranges_np, scaled_vsize_np, coor_2_occ_tensor):
    print("ranges_np", ranges_np, scaled_vsize_np)
    mask = coor_2_occ_tensor.cpu().numpy() > 0
    xdim, ydim, zdim = coor_2_occ_tensor.shape[1:]
    x_ = np.arange(0, xdim)
    y_ = np.arange(0, ydim)
    z_ = np.arange(0, zdim)
    x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')
    xyz = np.stack([x,y,z], axis=-1).reshape(-1,3).astype(np.float32)
    xyz = ranges_np[None, :3] + (xyz + 0.5) * scaled_vsize_np[None, :]
    xyz = xyz[mask.reshape(-1)]
    save_points(xyz, "./", "occ_xyz")
    print(xyz.shape)

def save_queried_points(point_xyz_tensor, point_xyz_pers_tensor, sample_pidx_tensor, pixel_idx_tensor, pixel_idx_cur_tensor, vdim, vsize, ranges):
    B, R, SR, K = sample_pidx_tensor.shape
    # pixel_inds = torch.as_tensor([3210, 3217,3218,3219,3220, 3221,3222,3223,3224,3225,3226,3227,3228,3229,3230, 3231,3232,3233,3234,3235, 3236,3237,3238,3239,3240], device=sample_pidx_tensor.device, dtype=torch.int64)
    point_inds = sample_pidx_tensor[0, :, :, :]
    # point_inds = sample_pidx_tensor[0, pixel_inds, :, :]
    mask = point_inds > -1
    point_inds = torch.masked_select(point_inds, mask).to(torch.int64)
    queried_point_xyz_tensor = point_xyz_tensor[0, point_inds, :]
    queried_point_xyz = queried_point_xyz_tensor.cpu().numpy()
    print("queried_point_xyz.shape", B, R, SR, K, point_inds.shape, queried_point_xyz_tensor.shape,
          queried_point_xyz.shape)
    print("pixel_idx_cur_tensor", pixel_idx_cur_tensor.shape)
    render_pixel_mask(pixel_idx_cur_tensor.cpu().numpy(), vdim[0], vdim[1])

    render_mask_pers_points(point_xyz_pers_tensor[0, point_inds, :].cpu().numpy(), vsize, ranges, vdim[0], vdim[1])

    plt.show()

def load_init_points(scan, data_dir="/home/xharlie/user_space/data/nrData/nerf/nerf_synthetic_colmap"):
    points_path = os.path.join(data_dir, scan, "colmap_results/dense/fused.ply")
    # points_path = os.path.join(self.data_dir, self.scan, "exported/pcd_te_1_vs_0.01_jit.ply")
    assert os.path.exists(points_path)
    from plyfile import PlyData, PlyElement
    plydata = PlyData.read(points_path)
    # plydata (PlyProperty('x', 'double'), PlyProperty('y', 'double'), PlyProperty('z', 'double'), PlyProperty('nx', 'double'), PlyProperty('ny', 'double'), PlyProperty('nz', 'double'), PlyProperty('red', 'uchar'), PlyProperty('green', 'uchar'), PlyProperty('blue', 'uchar'))
    print("plydata", plydata.elements[0])
    x,y,z=torch.as_tensor(plydata.elements[0].data["x"].astype(np.float32), device="cuda", dtype=torch.float32), torch.as_tensor(plydata.elements[0].data["y"].astype(np.float32), device="cuda", dtype=torch.float32), torch.as_tensor(plydata.elements[0].data["z"].astype(np.float32), device="cuda", dtype=torch.float32)
    points_xyz = torch.stack([x,y,z], dim=-1).to(torch.float32)
    return points_xyz

if __name__ == "__main__":
    obj = "lego"
    # point_file = "{}.pkl".format(obj)
    # point_dir = os.path.expandvars("${nrDataRoot}/nerf/nerf_synthetic_points/")
    r = 0.36000002589322094
    ranges = np.array([-1., -1.3, -1.2, 1., 1.3, 1.2], dtype=np.float32)
    vdim = np.array([400, 400, 400], dtype=np.int32)
    # vsize = np.array([2 * r / vdim[0], 2 * r / vdim[1], 4. / vdim[2]], dtype=np.float32)
    vsize = np.array([0.005, 0.005, 0.005], dtype=np.float32)
    vscale = np.array([2, 2, 2], dtype=np.int32)
    SR = 24
    P = 128
    K = 8
    NN = 2
    ray_num = 2048
    kernel_size = np.array([5, 5, 5], dtype=np.int32)
    radius_limit = 0  # r / 400 * 5 #r / 400 * 5
    depth_limit = 0  # 4. / 400 * 1.5 # r / 400 * 2
    max_o = 500000
    near_depth, far_depth = 2., 6.
    shading_count = 400

    xrange = np.arange(0, 800, 1, dtype=np.int32)
    yrange = np.arange(0, 800, 1, dtype=np.int32)
    xv, yv = np.meshgrid(xrange, yrange, sparse=False, indexing='ij')
    inds = np.arange(len(xv.reshape(-1)), dtype=np.int32)
    np.random.shuffle(inds)
    inds = inds[:ray_num, ...]
    pixel_idx = np.stack([xv, yv], axis=-1).reshape(-1, 2)[inds]  # 20000 * 2
    gpu = 0
    imgidx = 3
    split = ["train"]

    if gpu < 0:
        import pycuda.autoinit
    else:
        drv.init()
        dev1 = drv.Device(gpu)
        ctx1 = dev1.make_context()
    try_build(ranges, vsize, vdim, vscale, max_o, P, kernel_size, SR, K, pixel_idx, obj,
              radius_limit, depth_limit, near_depth, far_depth, shading_count, split=split, imgidx=imgidx, gpu=0, NN=NN)