import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from model import three_view_net
from utils import load_network
import yaml
import argparse
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms
from PIL import Image
import cv2 

def draw_CAM(model, img_path, save_path, transform=None, visual_heatmap=False):

    img = Image.open(img_path).convert('RGB')
    if transform:
        img = transform(img)

    img = img.unsqueeze(0)

    model.eval().cuda()

    with torch.no_grad():
        x = model.model_3.model.conv1(img.cuda())
        x = model.model_3.model.bn1(x)
        x = model.model_3.model.relu(x)
        x = model.model_3.model.maxpool(x)
        x = model.model_3.model.layer1(x)
        x = model.model_3.model.layer2(x)
        x = model.model_3.model.layer3(x)
        output = model.model_3.model.layer4(x)
        print(output.shape)
    # 为了能读取到中间梯度定义的辅助函数
    def extract(g):
        global features_grad
        features_grad = g
 
    output.register_hook(extract)
    output.backward() # 计算梯度
 
    grads = features_grad   # 获取梯度
 
    pooled_grads = torch.nn.functional.adaptive_avg_pool2d(grads, (1, 1))
 
    # 此处batch size默认为1，所以去掉了第0维（batch size维）
    pooled_grads = pooled_grads[0]
    output = output[0]
    # 512是最后一层feature的通道数
    for i in range(2048):
        output[i, ...] *= pooled_grads[i, ...]
 
    # 以下部分同Keras版实现
    heatmap = output.detach().numpy()
    heatmap = np.mean(heatmap, axis=0)
 
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
 
    # 可视化原始热力图
    if visual_heatmap:
        plt.matshow(heatmap)
        plt.show()
 
    img = cv2.imread(img_path)  # 用cv2加载原始图像
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
    heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 将热力图应用于原始图像
    superimposed_img = heatmap * 0.4 + img  # 这里的0.4是热力图强度因子
    cv2.imwrite(save_path, superimposed_img)  # 将图像保存到硬盘






os.environ["CUDA_VISIBLE_DEVICES"] = '2'
parser = argparse.ArgumentParser(description='Training')

parser.add_argument('--data_dir',default='/home/wangtyu/datasets/University-Release/test',type=str, help='./test_data')
parser.add_argument('--name', default='three_view_long_share_d0.75_256_s1_google_PCB4_lr0.001', type=str, help='save model path')
parser.add_argument('--batchsize', default=1, type=int, help='batchsize')

opt = parser.parse_args()
config_path = os.path.join('./model',opt.name,'opts.yaml')
with open(config_path, 'r') as stream:
    config = yaml.load(stream)
opt.fp16 = config['fp16']
opt.PCB = config['PCB']
opt.stride = config['stride']
opt.block = config['block']
opt.views = config['views']
if 'h' in config:
    opt.h = config['h']
    opt.w = config['w']
if 'nclasses' in config: # tp compatible with old config files
    opt.nclasses = config['nclasses']
else:
    opt.nclasses = 751

data_transforms = transforms.Compose([
        transforms.Resize((opt.h, opt.w), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
imgname = 'gallery_drone/0721/image-28.jpeg'
print(opt.data_dir)
img_path = os.path.join(opt.data_dir,imgname)
save_path = '/home/wangtyu/cam_headmap.jpg'
model, _, epoch = load_network(opt.name, opt)
draw_CAM(model, img_path, save_path, transform=data_transforms, visual_heatmap=False)