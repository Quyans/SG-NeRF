import torch
marching = torch.load('checkpoints/scannet/scene241/0_net_ray_marching.pth')
for key,value in marching.items():
    print(key)
    print(value.shape)
    '''
    aggregator.block1
    .0.weight
    torch.Size([256, 284])
    aggregator.block1
    .0.bias
    torch.Size([256])
    aggregator.block1
    .2.weight
    torch.Size([256, 256])
    aggregator.block1
    .2.bias
    torch.Size([256])
    aggregator.block3
    .0.weight
    torch.Size([256, 263])
    aggregator.block3
    .0.bias
    torch.Size([256])
    aggregator.block3
    .2.weight
    torch.Size([256, 256])
    aggregator.block3
    .2.bias
    torch.Size([256])
    aggregator.alpha_branch
    .0.weight
    torch.Size([1, 256])
    aggregator.alpha_branch
    .0.bias
    torch.Size([1])
    aggregator.color_branch
    .0.weight
    torch.Size([128, 280])
    aggregator.color_branch
    .0.bias
    torch.Size([128])
    aggregator.color_branch
    .2.weight
    torch.Size([128, 128])
    aggregator.color_branch
    .2.bias
    torch.Size([128])
    aggregator.color_branch
    .4.weight
    torch.Size([128, 128])
    aggregator.color_branch
    .4.bias
    torch.Size([128])
    aggregator.color_branch
    .6.weight
    torch.Size([3, 128])
    aggregator.color_branch
    .6.bias
    torch.Size([3])
    neural_points.xyz
    torch.Size([4242263, 3])
    neural_points.points_embeding
    torch.Size([1, 4242263, 32])
    neural_points.points_conf
    torch.Size([1, 4242263, 1])
    neural_points.points_dir
    torch.Size([1, 4242263, 3])
    neural_points.points_color
    torch.Size([1, 4242263, 3])
    neural_points.Rw2c
    torch.Size([3, 3])
    '''