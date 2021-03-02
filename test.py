# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import scipy.io
import yaml
import math
from model import ft_net, two_view_net, three_view_net, three_view_net_test
from utils import load_network
from image_folder import customData, customData_one
#fp16
try:
    from apex.fp16_utils import *
except ImportError: # will be 3.x series
    print('This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')
######################################################################
# Options
# --------

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--which_epoch',default='last', type=str, help='0,1,2,3...or last')
parser.add_argument('--test_dir',default='./data/test',type=str, help='./test_data')
parser.add_argument('--name', default='three_view_long_share_d0.75_256_s1_google', type=str, help='save model path')
parser.add_argument('--pool', default='avg', type=str, help='avg|max')
parser.add_argument('--batchsize', default=128, type=int, help='batchsize')
parser.add_argument('--h', default=256, type=int, help='height')
parser.add_argument('--w', default=256, type=int, help='width')
parser.add_argument('--views', default=2, type=int, help='views')
parser.add_argument('--pad', default=0, type=int, help='padding')
parser.add_argument('--use_dense', action='store_true', help='use densenet121' )
parser.add_argument('--PCB', action='store_true', help='use PCB' )
parser.add_argument('--multi', action='store_true', help='use multiple query' )
parser.add_argument('--fp16', action='store_true', help='use fp16.' )
parser.add_argument('--scale_test', action='store_true', help='scale test' )
parser.add_argument('--ms',default='1', type=str,help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')

opt = parser.parse_args()
###load config###
# load the training config
config_path = os.path.join('./model',opt.name,'opts.yaml')
with open(config_path, 'r') as stream:
        config = yaml.load(stream)
opt.fp16 = config['fp16'] 
opt.use_dense = config['use_dense']
opt.use_NAS = config['use_NAS']
opt.stride = config['stride']
opt.views = config['views']
opt.PCB = config['PCB']
opt.block = config['block']
scale_test = opt.scale_test
if 'h' in config:
    opt.h = config['h']
    opt.w = config['w']
print('------------------------------',opt.h)
if 'nclasses' in config: # tp compatible with old config files
    opt.nclasses = config['nclasses']
else: 
    opt.nclasses = 729 

str_ids = opt.gpu_ids.split(',')
#which_epoch = opt.which_epoch
name = opt.name
test_dir = opt.test_dir

gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >=0:
        gpu_ids.append(id)

print('We use the scale: %s'%opt.ms)
str_ms = opt.ms.split(',')
ms = []
for s in str_ms:
    s_f = float(s)
    ms.append(math.sqrt(s_f))

# set gpu ids
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True

######################################################################
# Load Data
# ---------
#
# We will use torchvision and torch.utils.data packages for loading the
# data.
#
data_transforms = transforms.Compose([
        transforms.Resize((opt.h, opt.w), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

#像素点平移动的transforms
transform_move_list = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


if opt.PCB:
    data_transforms = transforms.Compose([
        # transforms.Resize((384,192), interpolation=3),
        transforms.Resize((opt.h,opt.w), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
    ])


data_dir = test_dir

if opt.multi:
    image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in ['gallery','query','multi-query']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=False, num_workers=16) for x in ['gallery','query','multi-query']}
else:
    # image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in ['gallery_satellite','gallery_drone', 'gallery_street', 'query_satellite', 'query_drone', 'query_street']}
    image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in ['gallery_satellite','gallery_drone', 'gallery_street', 'gallery_satellite_usa_un']}
    # image_datasets = {}
    # for x in ['gallery_satellite','gallery_drone', 'gallery_street', 'gallery_satellite_usa_un']:
    #     image_datasets[x] = customData( os.path.join(data_dir,x) ,data_transforms, rotate=0)
    if scale_test:
        for x in ['query_drone']:
            print('----------scale test--------------')
            image_datasets[x] = customData_one( os.path.join(data_dir,x) ,data_transforms, rotate=0, reverse=False)
    else:
        for x in ['query_satellite', 'query_drone', 'query_street', 'query_drone_one']:
            if opt.pad > 0:
                print('-----------move pixel test-----------')
                image_datasets[x] = customData( os.path.join(data_dir,x) ,transform_move_list, rotate=0, pad=opt.pad)
            else: 
                print('----------rotation test--------------')   
                image_datasets[x] = customData( os.path.join(data_dir,x) ,data_transforms, rotate=0)
    print(image_datasets.keys())
    # image_datasets = {x: customData( os.path.join(data_dir,x) ,data_transforms, rotate=0) for x in ['query_satellite', 'query_drone', 'query_street']}
    if scale_test:
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=False, num_workers=16) for x in ['gallery_satellite', 'gallery_drone','gallery_street', 'gallery_satellite_usa_un', 'query_drone']}
    else:
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=False, num_workers=16) for x in ['gallery_satellite', 'gallery_drone','gallery_street', 'gallery_satellite_usa_un', 'query_satellite', 'query_drone', 'query_street', 'query_drone_one']}
use_gpu = torch.cuda.is_available()

######################################################################
# Extract feature
# ----------------------
#
# Extract feature from  a trained model.
#
def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def which_view(name):
    if 'satellite' in name:
        return 1
    elif 'street' in name:
        return 2
    elif 'drone' in name:
        return 3
    else:
        print('unknown view')
    return -1

def extract_feature(model,dataloaders, view_index = 1):
    features = torch.FloatTensor()
    count = 0
    for data in dataloaders:
        img, label = data
        n, c, h, w = img.size()
        count += n
        print(count)
        ff = torch.FloatTensor(n,512).zero_().cuda()
        if opt.PCB:
            # ff = torch.FloatTensor(n,2048,6).zero_().cuda()
            ff = torch.FloatTensor(n,512,opt.block).zero_().cuda()
        for i in range(2):
            if(i==1):
                img = fliplr(img)
            input_img = Variable(img.cuda())
            for scale in ms:
                if scale != 1:
                    # bicubic is only  available in pytorch>= 1.1
                    input_img = nn.functional.interpolate(input_img, scale_factor=scale, mode='bilinear', align_corners=False)
                if opt.views ==2:
                    if view_index == 1:
                        outputs, _ = model(input_img, None) 
                    elif view_index ==2:
                        _, outputs = model(None, input_img) 
                elif opt.views ==3:
                    if view_index == 1:
                        outputs, _, _ = model(input_img, None, None)
                    elif view_index ==2:
                        _, outputs, _ = model(None, input_img, None)
                    elif view_index ==3:
                        _, _, outputs = model(None, None, input_img)
                ff += outputs
        # norm feature
        if opt.PCB:
            # feature size (n,2048,6)
            # 1. To treat every part equally, I calculate the norm for every 2048-dim part feature.
            # 2. To keep the cosine score==1, sqrt(6) is added to norm the whole feature (2048*6).
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(opt.block) 
            ff = ff.div(fnorm.expand_as(ff))
            ff = ff.view(ff.size(0), -1)
        else:
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))

        features = torch.cat((features,ff.data.cpu()), 0)
    return features

def get_id(img_path):
    camera_id = []
    labels = []
    paths = []
    for path, v in img_path:
        # print(path, v)
        folder_name = os.path.basename(os.path.dirname(path))
        labels.append(int(folder_name))
        paths.append(path)
    return labels, paths

######################################################################
# Load Collected data Trained model
print('-------test-----------')

model, _, epoch = load_network(opt.name, opt)
if opt.PCB:
    print('use PCB')
    # model = three_view_net_test(model)
    for i in range(opt.block):
        cls_name = 'classifier'+str(i)
        c = getattr(model, cls_name)
        c.classifier = nn.Sequential()
else:
    model.classifier.classifier = nn.Sequential()
model = model.eval()
if use_gpu:
    model = model.cuda()

# Extract feature
since = time.time()

# gallery_name = 'gallery_street' 
# query_name = 'query_satellite' 

gallery_name = 'gallery_satellite'
# query_name = 'query_street'

#gallery_name = 'gallery_street'
query_name = 'query_drone'
# query_name = 'query_drone_one'
# gallery_name = 'gallery_drone'
# gallery_name = 'gallery_satellite_usa_un'
which_gallery = which_view(gallery_name)
which_query = which_view(query_name)
print('%d -> %d:'%(which_query, which_gallery))

gallery_path = image_datasets[gallery_name].imgs
f = open('gallery_name.txt','w')
for p in gallery_path:
    f.write(p[0]+'\n')
query_path = image_datasets[query_name].imgs
f = open('query_name.txt','w')
for p in query_path:
    f.write(p[0]+'\n')

gallery_label, gallery_path  = get_id(gallery_path)
query_label, query_path  = get_id(query_path)

if __name__ == "__main__":
    with torch.no_grad():
        query_feature = extract_feature(model,dataloaders[query_name], which_query)
        gallery_feature = extract_feature(model,dataloaders[gallery_name], which_gallery)


    # For street-view image, we use the avg feature as the final feature.
    '''
    if which_query == 2:
        new_query_label = np.unique(query_label)
        new_query_feature = torch.FloatTensor(len(new_query_label) ,512).zero_()
        for i, query_index in enumerate(new_query_label):
            new_query_feature[i,:] = torch.sum(query_feature[query_label == query_index, :], dim=0)
        query_feature = new_query_feature
        fnorm = torch.norm(query_feature, p=2, dim=1, keepdim=True)
        query_feature = query_feature.div(fnorm.expand_as(query_feature))
        query_label   = new_query_label
    elif which_gallery == 2:
        new_gallery_label = np.unique(gallery_label)
        new_gallery_feature = torch.FloatTensor(len(new_gallery_label), 512).zero_()
        for i, gallery_index in enumerate(new_gallery_label):
            new_gallery_feature[i,:] = torch.sum(gallery_feature[gallery_label == gallery_index, :], dim=0)
        gallery_feature = new_gallery_feature
        fnorm = torch.norm(gallery_feature, p=2, dim=1, keepdim=True)
        gallery_feature = gallery_feature.div(fnorm.expand_as(gallery_feature))
        gallery_label   = new_gallery_label
    '''
    time_elapsed = time.time() - since
    print('Test complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # Save to Matlab for check
    result = {'gallery_f':gallery_feature.numpy(),'gallery_label':gallery_label,'gallery_path':gallery_path,'query_f':query_feature.numpy(),'query_label':query_label, 'query_path':query_path}
    scipy.io.savemat('pytorch_result.mat',result)

    print(opt.name)
    result = './model/%s/result.txt'%opt.name
    os.system('CUDA_VISIBLE_DEVICES=%d python evaluate_gpu.py | tee -a %s'%(gpu_ids[0],result))

    #test single part and combination
    '''
    # for i in range(7):
    #     if i == 0:
    #         query_feature_ = query_feature[:,0:512]
    #         gallery_feature_ = gallery_feature[:,0:512]
    #         print('-------------- 1 -----------------')
    #     if i == 1:
    #         query_feature_ = query_feature[:,512:1024]
    #         gallery_feature_ = gallery_feature[:,512:1024]
    #         print('-------------- 2 -----------------')
    #     if i == 2:
    #         query_feature_ = query_feature[:,1024:1536]
    #         gallery_feature_ = gallery_feature[:,1024:1536]
    #         print('-------------- 3 -----------------')
    #     if i == 3:
    #         query_feature_ = query_feature[:,1536:2048]
    #         gallery_feature_ = gallery_feature[:,1536:2048]
    #         print('-------------- 4 -----------------')
    #     if i == 4:
    #         query_feature_ = query_feature[:,0:1024]
    #         gallery_feature_ = gallery_feature[:,0:1024]
    #         print('-------------- 1+2 -----------------')
    #     if i == 5:
    #         query_feature_ = query_feature[:,0:1536]
    #         gallery_feature_ = gallery_feature[:,0:1536]
    #         print('-------------- 1+2+3 -----------------')
    #     if i == 6:
    #         query_feature_ = query_feature[:,0:2048]
    #         gallery_feature_ = gallery_feature[:,0:2048]
    #         print('-------------- 1+2+3+4 -----------------')
    #     result = {'gallery_f':gallery_feature_.numpy(),'gallery_label':gallery_label,'gallery_path':gallery_path,'query_f':query_feature_.numpy(),'query_label':query_label, 'query_path':query_path}
    #     scipy.io.savemat('pytorch_result.mat',result)
    #     print(opt.name)
    #     result = './model/%s/result.txt'%opt.name
    #     os.system('CUDA_VISIBLE_DEVICES=%d python evaluate_gpu.py | tee -a %s'%(gpu_ids[0],result))
    '''
    # query_feature_ = query_feature[:,0:1536]
    # gallery_feature_ = gallery_feature[:,512:2048]
    # print('-------------- （1+2+3，2+3+4） -----------------')
    # result = {'gallery_f':gallery_feature_.numpy(),'gallery_label':gallery_label,'gallery_path':gallery_path,'query_f':query_feature_.numpy(),'query_label':query_label, 'query_path':query_path}
    # scipy.io.savemat('pytorch_result.mat',result)
    # print(opt.name)
    # result = './model/%s/result.txt'%opt.name
    # os.system('CUDA_VISIBLE_DEVICES=%d python evaluate_gpu.py | tee -a %s'%(gpu_ids[0],result))