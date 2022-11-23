import torch


import os
basedir = "/qys/SG-NeRF/checkpoints/scannet/scene046_00_Semantic_640480step50_block2bpnet_laybpnet=1"


data8 = torch.load(os.path.join(basedir,"20000_net_ray_marching.pth"))
data9 = torch.load(os.path.join(basedir,"30000_net_ray_marching.pth"))
# data10 = torch.load(os.path.join(basedir,"100000_net_ray_marching.pth"))

for i in list(data8.keys()):
    print(i)

print(5)

# bpnet_points_embedding

# bpnet.layer0_2d.0.weight

# bpnet.layer3_2d.0.weight