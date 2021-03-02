import os
from shutil import copyfile
import shutil
import numpy as np
import scipy.io as sio
from scipy.misc import imread, imsave
import cv2

allDataList = './ACT_data.mat'
anuData = sio.loadmat(allDataList)
img_root = '/home/wangtyu/ANU_data_small/'

idx = 0
id_alllist = []
id_idx_alllist = []
for i in range(0,len(anuData['panoIds'])):
    grd_id_align = img_root + 'streetview/' + anuData['panoIds'][i] + '_grdView.jpg'  
    sat_id_ori = img_root + 'satview_polish/' + anuData['panoIds'][i] + '_satView_polish.jpg'
    id_alllist.append([grd_id_align, sat_id_ori])
    id_idx_alllist.append(idx)
    idx += 1
all_data_size = len(id_alllist)
#get trainList
training_inds = anuData['trainSet']['trainInd'][0][0] - 1
trainNum = len(training_inds)
trainList = []
trainIdList = []
for k in range(trainNum):
    trainList.append(id_alllist[training_inds[k][0]])
    trainIdList.append(k)
# get valList
val_inds = anuData['valSet']['valInd'][0][0] - 1
valNum = len(val_inds)
valList = []
valIdList = []
for k in range(valNum):
    valList.append(id_alllist[val_inds[k][0]])
    valIdList.append(k)

# prepare training set
print('begin to prepare train')
d_train_dir = '/home/wangtyu/datasets/CVACT/train'
for m in range(trainNum):
    s_str_dir = trainList[m][0]
    s_sat_dir = trainList[m][1]
    d_str_dir = os.path.join(d_train_dir,'streetview',str(trainIdList[m]+1).zfill(7))
    d_sat_dir = os.path.join(d_train_dir,'satview_polish',str(trainIdList[m]+1).zfill(7))

    if os.path.exists(s_str_dir) and os.path.exists(s_sat_dir):
        if not os.path.exists(d_str_dir):
            os.makedirs(d_str_dir)
        if not os.path.exists(d_sat_dir):
            os.makedirs(d_sat_dir)
        str_name = os.path.basename(s_str_dir)
        sat_name = os.path.basename(s_sat_dir)

        #process satellite view image
        img = cv2.imread(s_sat_dir)
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
        imsave(os.path.join(d_sat_dir,sat_name), img)
        # shutil.copyfile(s_sat_dir,os.path.join(d_sat_dir,sat_name))

        #process street view image
        signal = imread(s_str_dir)
        start = int(832 / 4)
        image = signal[start: start + int(832 / 2), :, :]
        image = cv2.resize(image, (616, 112), interpolation=cv2.INTER_AREA)
        imsave(os.path.join(d_str_dir,str_name), image)
    else:
        print('unexist street file name: ', s_str_dir)
        print('unexist satellite file name: ', s_sat_dir)
        print('train id:', m)
        print('dataset index:', training_inds[m][0])
        print('dataset unexist pair: ', id_alllist[training_inds[m][0]])
        unexist_file = './unexist_train_file.txt'
        if not os.path.exists(unexist_file):
            os.system(r"touch {}".format(unexist_file))
        with open(unexist_file, 'a') as f:
            f.write(s_str_dir)
            f.write('\n')
            f.write(s_sat_dir)
            f.write('\n')
            f.write('m: '+str(m)+' training index: '+ str(training_inds[m][0]))
            f.write('\n')
            f.writelines(id_alllist[training_inds[m][0]])
            f.write('\n###################################')
    print(m)
 
#prepare val set
print('begin to prepare val')
d_val_dir = '/home/wangtyu/datasets/CVACT/val'
for m in range(valNum):
    s_str_dir = valList[m][0]
    s_sat_dir = valList[m][1]
    d_str_dir = os.path.join(d_val_dir,'streetview',str(valIdList[m]+1).zfill(7))
    d_sat_dir = os.path.join(d_val_dir,'satview_polish',str(valIdList[m]+1).zfill(7))
    
    if os.path.exists(s_str_dir) and os.path.exists(s_sat_dir):
        if not os.path.exists(d_str_dir):
            os.makedirs(d_str_dir)
        if not os.path.exists(d_sat_dir):
            os.makedirs(d_sat_dir)
        str_name = os.path.basename(s_str_dir)
        sat_name = os.path.basename(s_sat_dir)

        #process satellite view image
        img = cv2.imread(s_sat_dir)
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
        imsave(os.path.join(d_sat_dir,sat_name), img)
        
        #process street view image
        signal = imread(s_str_dir)
        start = int(832 / 4)
        image = signal[start: start + int(832 / 2), :, :]
        image = cv2.resize(image, (616, 112), interpolation=cv2.INTER_AREA)
        imsave(os.path.join(d_str_dir,str_name), image)
        
    else:
        print('unexist street file name: ', s_str_dir)
        print('unexist satellite file name: ', s_sat_dir)
        print('val id:', m)
        print('dataset index:', val_inds[m][0])
        print('dataset unexist pair: ', id_alllist[val_inds[m][0]])
        unexist_file = './unexist_val_file.txt'
        if not os.path.exists(unexist_file):
            os.system(r"touch {}".format(unexist_file))
        with open(unexist_file, 'a') as f:
            f.write(s_str_dir)
            f.write('\n')
            f.write(s_sat_dir)
            f.write('\n')
            f.write('m: '+str(m)+' val index: '+ str(val_inds[m][0]))
            f.write('\n')
            f.writelines(id_alllist[val_inds[m][0]])
            f.write('\n###################################')

    print(m)