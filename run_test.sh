# python test_cvusa.py \
# --name='usa_vgg_noshare_warm5_4PCBv_lr0.05' \
# --test_dir='/home/wangtyu/datasets/CVUSA/val' \
# --gpu_ids='1'

# python test.py \
# --name='three_view_long_share_d0.75_256_s1_google_PCB4_lr0.001' \
# --test_dir='/home/wangtyu/datasets/University-Release/test' \
# --batchsize=64 \
# --gpu_ids='1'

python test_cvusa.py \
--name='usa_resnet_noshare_warm5_8PCBv_lr0.01' \
--test_dir='/home/wangtyu/datasets/CVUSA/val' \
--gpu_ids='3'
