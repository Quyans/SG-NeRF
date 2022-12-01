#!/bin/bash


nrCheckpoint="../../checkpoints"
nrDataRoot="../../data_src"

# name='scene024102_Semantic_step50_debug2'
name='scene0113_320240_nostep_layer4=0'


resume_iter=best #latest

data_root="${nrDataRoot}/scannet/scans/scene0113"
scan="scene0113_00"

img_wh="320 240"
predict_semantic=1
semantic_guidance=1
layers_2d=34
arch_3d=MinkUNet18A
classes=20


load_points=1
feat_grad=1
conf_grad=1
dir_grad=0
color_grad=1
vox_res=900
normview=0
prune_thresh=-1
prune_iter=-1 #10000

feedforward=0
ref_vid=0
bgmodel="no" #"plane"
depth_occ=0
depth_vid="0"
trgt_id=0
manual_depth_view=1
init_view_num=3
pre_d_est="${nrCheckpoint}/MVSNet/model_000014.ckpt"
manual_std_depth=0.0
depth_conf_thresh=0.8
geo_cnsst_num=0
edge_filter=10 # pixels crop out at image edge

appr_feature_str0="imgfeat_0_0123 dir_0 point_conf"
point_conf_mode="1" # 0 for only at features, 1 for multi at weight
point_dir_mode="1" # 0 for only at features, 1 for color branch
point_color_mode="1" # 0 for only at features, 1 for color branch
default_conf=-1

agg_feat_xyz_mode="None"
agg_alpha_xyz_mode="None"
agg_color_xyz_mode="None"
feature_init_method="rand" #"rand" # "zeros"
agg_axis_weight=" 1. 1. 1."
agg_dist_pers=20
radius_limit_scale=4
depth_limit_scale=0
vscale=" 2 2 2 "
kernel_size=" 3 3 3 "
query_size=" 3 3 3 "
vsize=" 0.008 0.008 0.008 " #" 0.005 0.005 0.005 "
wcoord_query=1
z_depth_dim=400
max_o=610000
ranges=" -10.0 -10.0 -10.0 10.0 10.0 10.0 "
SR=24 # 24
K=8 #8
P=32 # 32
NN=2

act_type="LeakyReLU"

agg_intrp_order=2
agg_distance_kernel="linear" #"avg" #"feat_intrp"
point_features_dim=32
shpnt_jitter="uniform" #"uniform" # uniform gaussian

which_agg_model="viewmlp"
apply_pnt_mask=1
shading_feature_mlp_layer0=1
shading_feature_mlp_layer1=2
shading_feature_mlp_layer2=0
shading_feature_mlp_layer2_bpnet=1
shading_feature_mlp_linear=0
shading_feature_mlp_layer3=2 #2 原本#0 
shading_feature_mlp_layer4=0 #0 原本 #1 yuze
shading_alpha_mlp_layer=1
shading_color_mlp_layer=4
shading_feature_num=256
dist_xyz_freq=5
num_feat_freqs=3
dist_xyz_deno=0


raydist_mode_unit=1
dataset_name='scannet_ft'
pin_data_in_memory=1
model='mvs_points_volumetric'
near_plane=0.1
far_plane=8.0
which_ray_generation='near_far_linear' #'nerf_near_far_linear' #

dir_norm=0

which_tonemap_func="off" #"gamma" #
which_render_func='radiance'
which_blend_func='alpha'
out_channels=4

num_pos_freqs=10
num_viewdir_freqs=4 #6

random_sample='random'
random_sample_size=32 # 32 * 32 = 1024
batch_size=1

plr=0.002
lr=0.0005 # 0.0005 #0.00015
lr_policy="iter_exponential_decay"
lr_decay_iters=1000000
lr_decay_exp=0.1

gpu_ids='0'
checkpoints_dir="${nrCheckpoint}/scannet/"
resume_dir="${nrCheckpoint}/init/dtu_dgt_d012_img0123_conf_agg2_32_dirclr20"

save_predict_label=1 
save_label_iter=5000 #预测一次
save_iter_freq=10000
save_point_freq=10000 #301840 #1
maximum_step=400000 #500000 #250000 #800000

niter=10000 #1000000
niter_decay=10000 #250000
n_threads=2

train_and_test=0 #1
test_num=25
test_freq=500000 #  #100 #1200 #1200 #30184 #30184 #50000
print_freq=100
test_num_step=350

prob_freq=1000000 #10001
prob_num_step=1000000
prob_kernel_size=" 3 3 3 1 1 1 "
prob_tiers=" 40000 120000 "
#prob_mode=0 # 0, n, 1 t, 10 t&n
prob_thresh=0.7
prob_mul=0.4

#zero_epsilon=1e-3

visual_items='coarse_raycolor gt_image '
zero_one_loss_items='conf_coefficient' #regularize background to be either 0 or 1
zero_one_loss_weights=" 0.0001 "
sparse_loss_weight=0

color_loss_weights=" 1.0 0.0 0.0 "
color_loss_items="ray_masked_coarse_raycolor ray_miss_coarse_raycolor coarse_raycolor"
test_color_loss_items="coarse_raycolor ray_miss_coarse_raycolor ray_masked_coarse_raycolor"



bg_color="white" #"0.0,0.0,0.0,1.0,1.0,1.0"
split="train"



cd run

python3 test_ft.py \
        --experiment $name \
        --img_wh $img_wh\
        --scan $scan \
        --data_root $data_root \
        --dataset_name $dataset_name \
        --model $model \
        --which_render_func $which_render_func \
        --which_blend_func $which_blend_func \
        --out_channels $out_channels \
        --num_pos_freqs $num_pos_freqs \
        --num_viewdir_freqs $num_viewdir_freqs \
        --random_sample $random_sample \
        --random_sample_size $random_sample_size \
        --batch_size $batch_size \
        --maximum_step $maximum_step \
        --plr $plr \
        --lr $lr \
        --lr_policy $lr_policy \
        --lr_decay_iters $lr_decay_iters \
        --lr_decay_exp $lr_decay_exp \
        --gpu_ids $gpu_ids \
        --checkpoints_dir $checkpoints_dir \
        --save_iter_freq $save_iter_freq \
        --niter $niter \
        --niter_decay $niter_decay \
        --n_threads $n_threads \
        --pin_data_in_memory $pin_data_in_memory \
        --train_and_test $train_and_test \
        --test_num $test_num \
        --test_freq $test_freq \
        --test_num_step $test_num_step \
        --test_color_loss_items $test_color_loss_items \
        --prob_freq $prob_freq \
        --prob_num_step $prob_num_step \
        --print_freq $print_freq \
        --bg_color $bg_color \
        --split $split \
        --which_ray_generation $which_ray_generation \
        --near_plane $near_plane \
        --far_plane $far_plane \
        --dir_norm $dir_norm \
        --which_tonemap_func $which_tonemap_func \
        --load_points $load_points \
        --resume_dir $resume_dir \
        --resume_iter $resume_iter \
        --feature_init_method $feature_init_method \
        --agg_axis_weight $agg_axis_weight \
        --agg_distance_kernel $agg_distance_kernel \
        --radius_limit_scale $radius_limit_scale \
        --depth_limit_scale $depth_limit_scale  \
        --vscale $vscale    \
        --kernel_size $kernel_size  \
        --SR $SR  \
        --K $K  \
        --P $P \
        --NN $NN \
        --agg_feat_xyz_mode $agg_feat_xyz_mode \
        --agg_alpha_xyz_mode $agg_alpha_xyz_mode \
        --agg_color_xyz_mode $agg_color_xyz_mode  \
        --save_point_freq $save_point_freq  \
        --raydist_mode_unit $raydist_mode_unit  \
        --agg_dist_pers $agg_dist_pers \
        --agg_intrp_order $agg_intrp_order \
        --shading_feature_mlp_layer0 $shading_feature_mlp_layer0 \
        --shading_feature_mlp_layer1 $shading_feature_mlp_layer1 \
        --shading_feature_mlp_layer2 $shading_feature_mlp_layer2 \
        --shading_feature_mlp_layer2_bpnet $shading_feature_mlp_layer2_bpnet\
        --shading_feature_mlp_layer3 $shading_feature_mlp_layer3 \
        --shading_feature_num $shading_feature_num \
        --dist_xyz_freq $dist_xyz_freq \
        --shpnt_jitter $shpnt_jitter \
        --shading_alpha_mlp_layer $shading_alpha_mlp_layer \
        --shading_color_mlp_layer $shading_color_mlp_layer \
        --which_agg_model $which_agg_model \
        --color_loss_weights $color_loss_weights \
        --num_feat_freqs $num_feat_freqs \
        --dist_xyz_deno $dist_xyz_deno \
        --apply_pnt_mask $apply_pnt_mask \
        --point_features_dim $point_features_dim \
        --color_loss_items $color_loss_items \
        --feedforward $feedforward \
        --trgt_id $trgt_id \
        --depth_vid $depth_vid \
        --ref_vid $ref_vid \
        --manual_depth_view $manual_depth_view \
        --pre_d_est $pre_d_est \
        --depth_occ $depth_occ \
        --manual_std_depth $manual_std_depth \
        --visual_items $visual_items \
        --appr_feature_str0 $appr_feature_str0 \
        --init_view_num $init_view_num \
        --feat_grad $feat_grad \
        --conf_grad $conf_grad \
        --dir_grad $dir_grad \
        --color_grad $color_grad \
        --depth_conf_thresh $depth_conf_thresh \
        --bgmodel $bgmodel \
        --vox_res $vox_res \
        --act_type $act_type \
        --geo_cnsst_num $geo_cnsst_num \
        --point_conf_mode $point_conf_mode \
        --point_dir_mode $point_dir_mode \
        --point_color_mode $point_color_mode \
        --normview $normview \
        --prune_thresh $prune_thresh \
        --prune_iter $prune_iter \
        --sparse_loss_weight $sparse_loss_weight \
        --zero_one_loss_items $zero_one_loss_items \
        --zero_one_loss_weights $zero_one_loss_weights \
        --default_conf $default_conf \
        --edge_filter $edge_filter \
        --vsize $vsize \
        --wcoord_query $wcoord_query \
        --ranges $ranges \
        --z_depth_dim $z_depth_dim \
        --max_o $max_o \
        --prob_thresh $prob_thresh \
        --prob_mul $prob_mul \
        --prob_kernel_size $prob_kernel_size \
        --prob_tiers $prob_tiers \
        --query_size $query_size \
        --debug \
        --predict_semantic $predict_semantic \
        --semantic_guidance $semantic_guidance
