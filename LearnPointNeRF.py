from plyfile import PlyData, PlyElement
import numpy as np
import torch

# points_path = "/home/slam/devdata/pointnerf/data_src/scannet/scans/scene0000_00/exported/pcd.ply"
# plydata = PlyData.read(points_path)
# x,y,z=torch.as_tensor(plydata.elements[0].data["x"].astype(np.float32), device="cuda", dtype=torch.float32), torch.as_tensor(plydata.elements[0].data["y"].astype(np.float32), device="cuda", dtype=torch.float32), torch.as_tensor(plydata.elements[0].data["z"].astype(np.float32), device="cuda", dtype=torch.float32)
# r,g,b=torch.as_tensor(plydata.elements[0].data["red"].astype(np.float32), device="cuda", dtype=torch.float32), torch.as_tensor(plydata.elements[0].data["green"].astype(np.float32), device="cuda", dtype=torch.float32), torch.as_tensor(plydata.elements[0].data["blue"].astype(np.float32), device="cuda", dtype=torch.float32)
# points_xyz = torch.stack([x,y,z,r,g,b], dim=-1)
# print(points_xyz.shape)

from data.scannet_ft_dataset import ScannetFtDataset
