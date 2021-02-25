# # LPN vgg16
# python train_cvact.py \
# --name='act_vgg_noshare_warm5_8LPN-s-r_lr0.1' \
# --data_dir='/home/wangtyu/datasets/CVACT/train' \
# --warm_epoch=5 \
# --batchsize=16 \
# --use_vgg16 \
# --h=256 \
# --w=256 \
# --fp16 \
# --LPN \
# --lr=0.1 \
# --block=8 \
# --gpu_ids='2'

# python test_cvact.py \
# --name='act_vgg_noshare_warm5_8LPN-s-r_lr0.1' \
# --test_dir='/home/wangtyu/datasets/CVACT/val' \
# --gpu_ids='2'

# LPN resnet50
python train_cvact.py \
--name='act_res50_noshare_warm5_8LPN-s-r_lr0.05' \
--data_dir='/home/wangtyu/datasets/CVACT/train' \
--warm_epoch=5 \
--batchsize=16 \
--h=256 \
--w=256 \
--fp16 \
--LPN \
--lr=0.05 \
--block=8 \
--stride=1 \
--gpu_ids='2'

python test_cvact.py \
--name='act_res50_noshare_warm5_8LPN-s-r_lr0.05' \
--test_dir='/home/wangtyu/datasets/CVACT/val' \
--gpu_ids='2'