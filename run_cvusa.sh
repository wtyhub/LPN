# lpn vgg16
python train_cvusa.py \
--name='usa_vgg_noshare_warm5_8LPN-s-r_lr0.1' \
--data_dir='/home/wangtyu/datasets/CVUSA/train' \
--warm_epoch=5 \
--batchsize=16 \
--h=256 \
--w=256 \
--use_vgg16 \
--fp16 \
--LPN \
--lr=0.1 \
--block=8 \
--gpu_ids='3'

python test_cvusa.py \
--name='usa_vgg_noshare_warm5_8LPN-s-r_lr0.1' \
--test_dir='/home/wangtyu/datasets/CVUSA/val' \
--gpu_ids='3' \

# lpn resnet50
# python train_cvusa.py \
# --name='usa_res50_noshare_warm5_8LPN-s-r_lr0.05' \
# --data_dir='/home/wangtyu/datasets/CVUSA/train' \
# --warm_epoch=5 \
# --batchsize=16 \
# --h=256 \
# --w=256 \
# --fp16 \
# --LPN \
# --lr=0.05 \
# --block=8 \
# --stride=1 \
# --gpu_ids='0'

# python test_cvusa.py \
# --name='usa_res50_noshare_warm5_8LPN-s-r_lr0.05' \
# --test_dir='/home/wangtyu/datasets/CVUSA/val' \
# --gpu_ids='0' \