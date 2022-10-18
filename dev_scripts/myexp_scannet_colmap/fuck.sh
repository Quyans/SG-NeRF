cd /home/vr717/Documents/qys/code/NSEPN_ori/NSEPN ;
CUDA_VISIBLE_DEVICES=0 /usr/bin/env /home/vr717/anaconda3/envs/pointnerf11/bin/python  /home/vr717/Documents/qys/code/NSEPN_ori/NSEPN/run/train_ft.py --name scene024102_Semantic_debug --scan scene0241_02 --data_root /home/vr717/Documents/qys/code/NSEPN_ori/NSEPN/data_src/scannet/scans --resume_iter best --load_points 1 --semantic_guidance 1 --feat_grad 1 --conf_grad 1 --dir_grad 0 --color_grad 1 --vox_res 900 --normview 0 --prune_thresh -1 --prune_iter 5000000 --prune_max_iter 100000 --feedforward 0 --ref_vid 0 --bgmodel no --depth_occ 0 --depth_vid 0 --trgt_id 0 --manual_depth_view 1 --init_view_num 3 --pre_d_est /home/vr717/Documents/qys/code/NSEPN_ori/NSEPN/checkpoints/MVSNet/model_000014.ckpt --manual_std_depth 0.0 --depth_conf_thresh 0.8 --geo_cnsst_num 0 --appr_feature_str0 imgfeat_0_0123 dir_0 point_conf --point_conf_mode 1 --point_dir_mode 1 --point_color_mode 1 --default_conf -1 --agg_feat_xyz_mode None --agg_alpha_xyz_mode None --agg_color_xyz_mode None --feature_init_method rand --agg_axis_weight 1. 1. 1. --agg_dist_pers 20 --radius_limit_scale 4 --depth_limit_scale 0 --vscale 2 2 2 --kernel_size 3 3 3 --query_size 3 3 3 --vsize 0.008 0.008 0.008 --wcoord_query 1 --z_depth_dim 400 --max_o 1600000 --ranges -10 -10 -10 10 10 10 --SR 40 --K 8 --P 32 --NN 2 --act_type LeakyReLU --agg_intrp_order 2 --agg_distance_kernel linear --point_features_dim 32 --shpnt_jitter passfunc --which_agg_model viewmlp --apply_pnt_mask 1 --shading_feature_mlp_layer0 1 --shading_feature_mlp_layer1 2 --shading_feature_mlp_layer2 0 --shading_feature_mlp_layer3 0 --shading_feature_mlp_layer4 1 --shading_alpha_mlp_layer 1 --shading_color_mlp_layer 4 --shading_feature_num 256 --dist_xyz_freq 5 --num_feat_freqs 3 --dist_xyz_deno 0 --raydist_mode_unit 1 --dataset_name scannet_ft --pin_data_in_memory 1 --model mvs_points_volumetric --near_plane 0.1 --far_plane 8.0 --which_ray_generation near_far_linear --dir_norm 0 --which_tonemap_func off --which_render_func radiance --which_blend_func alpha --out_channels 4 --num_pos_freqs 10 --num_viewdir_freqs 4 --random_sample random --random_sample_size 32 --batch_size 1 --plr 0.002 --lr 0.0005 --lr_policy iter_exponential_decay --lr_decay_iters 1000000 --lr_decay_exp 0.1 --gpu_ids 0 --checkpoints_dir /home/vr717/Documents/qys/code/NSEPN_ori/NSEPN/checkpoints/scannet/ --resume_dir /home/vr717/Documents/qys/code/NSEPN_ori/NSEPN/checkpoints/init/dtu_dgt_d012_img0123_conf_agg2_32_dirclr20 --save_iter_freq 10000 --save_point_freq 10000 --maximum_step 200000 --niter 10000 --niter_decay 10000 --n_threads 1 --train_and_test 0 --test_num 8 --test_freq 500000 --print_freq 40 --test_num_step 1 --prob_freq 5000000 --prob_num_step 50 --prob_thresh 0.7 --prob_mul 0.4 --prob_kernel_size 3 3 3 --prob_tiers 40000 --visual_items coarse_raycolor gt_image --zero_one_loss_items conf_coefficient --zero_one_loss_weights 0.0001 --sparse_loss_weight 0 --color_loss_weights 1.0 0.0 0.0 --color_loss_items ray_masked_coarse_raycolor ray_miss_coarse_raycolor coarse_raycolor --test_color_loss_items coarse_raycolor ray_miss_coarse_raycolor ray_masked_coarse_raycolor --bg_color white --split train --debug --semantic_guidance 1 --layers_2d 34 --arch_3d MinkUNet18A --classes 20 --predict_semantic 1
