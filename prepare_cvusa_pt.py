import os
from shutil import copyfile
import numpy as np
from scipy.misc import imread, imsave

############### polar transform function #############
def sample_within_bounds(signal, x, y, bounds):

    xmin, xmax, ymin, ymax = bounds

    idxs = (xmin <= x) & (x < xmax) & (ymin <= y) & (y < ymax)
    
    sample = np.zeros((x.shape[0], x.shape[1], signal.shape[-1]))
    sample[idxs, :] = signal[x[idxs], y[idxs], :]

    return sample

def sample_bilinear(signal, rx, ry):

    signal_dim_x = signal.shape[0]
    signal_dim_y = signal.shape[1]

    # obtain four sample coordinates
    ix0 = rx.astype(int)
    iy0 = ry.astype(int)
    ix1 = ix0 + 1
    iy1 = iy0 + 1

    bounds = (0, signal_dim_x, 0, signal_dim_y)

    # sample signal at each four positions
    signal_00 = sample_within_bounds(signal, ix0, iy0, bounds)
    signal_10 = sample_within_bounds(signal, ix1, iy0, bounds)
    signal_01 = sample_within_bounds(signal, ix0, iy1, bounds)
    signal_11 = sample_within_bounds(signal, ix1, iy1, bounds)

    na = np.newaxis
    # linear interpolation in x-direction
    fx1 = (ix1-rx)[...,na] * signal_00 + (rx-ix0)[...,na] * signal_10
    fx2 = (ix1-rx)[...,na] * signal_01 + (rx-ix0)[...,na] * signal_11

    # linear interpolation in y-direction
    return (iy1 - ry)[...,na] * fx1 + (ry - iy0)[...,na] * fx2

def apply_aerial_polar_transform(src_path, dst_path, imgname):
    S = 750  # Original size of the aerial image
    height = 112  # Height of polar transformed aerial image
    width = 616   # Width of polar transformed aerial image

    i = np.arange(0, height)
    j = np.arange(0, width)
    jj, ii = np.meshgrid(j, i) #坐标点

    y = S/2. - S/2./height*(height-1-ii)*np.sin(2*np.pi*jj/width)
    x = S/2. + S/2./height*(height-1-ii)*np.cos(2*np.pi*jj/width)

    # input_dir = '/Users/wongtyu/Downloads/cvusa/bingmap/19/'
    # output_dir = '/Users/wongtyu/Downloads/cvusa/polarmap/19/'

    signal = imread(src_path)
    image = sample_bilinear(signal, x, y)
    imsave(dst_path + '/' + imgname, image)

######################### prepare cvusa dataset ###########################

download_path = '/home/wangtyu/Datasets/cvusa/'
train_split = download_path + 'splits/train-19zl.csv'
train_save_path = download_path + 'train_pt/'  # polar transform satellite images

if not os.path.isdir(train_save_path):
    os.mkdir(train_save_path)
    os.mkdir(train_save_path + 'street')
    os.mkdir(train_save_path + 'satellite')

with open(train_split) as fp:
    line = fp.readline()
    while line:
        filename = line.split(',')
        #print(filename[0])
        src_path = download_path + '/' + filename[0]
        dst_path = train_save_path + '/satellite/' + os.path.basename(filename[0][:-4])
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
        # copyfile(src_path, dst_path + '/' + os.path.basename(filename[0]))
        apply_aerial_polar_transform(src_path, dst_path, os.path.basename(filename[0]))

        src_path = download_path + '/' + filename[1]
        dst_path = train_save_path + '/street/' + os.path.basename(filename[1][:-4])
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
        copyfile(src_path, dst_path + '/' + os.path.basename(filename[1]))       
        line = fp.readline()


val_split = download_path + 'splits/val-19zl.csv'
val_save_path = download_path + 'val_pt/'

if not os.path.isdir(val_save_path):
    os.mkdir(val_save_path)
    os.mkdir(val_save_path + 'street')
    os.mkdir(val_save_path + 'satellite')

with open(val_split) as fp:
    line = fp.readline()
    while line:
        filename = line.split(',')
        #print(filename[0])
        src_path = download_path + '/' + filename[0]
        dst_path = val_save_path + '/satellite/' + os.path.basename(filename[0][:-4])
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
        # copyfile(src_path, dst_path + '/' + os.path.basename(filename[0]))
        apply_aerial_polar_transform(src_path, dst_path, os.path.basename(filename[0]))

        src_path = download_path + '/' + filename[1]
        dst_path = val_save_path + '/street/' + os.path.basename(filename[1][:-4])
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
        copyfile(src_path, dst_path + '/' + os.path.basename(filename[1]))

        line = fp.readline()


