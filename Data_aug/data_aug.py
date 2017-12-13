from scipy.misc import imread, imsave, imshow
import numpy as np
import os
from matplotlib import pyplot as plt
from skimage import transform
import random

def data_aug(input_frame,angel=5,resize_rate=0.9):
    flip = random.randint(0, 1)
    size = input_frame.shape[0]
    rsize = random.randint(np.floor(resize_rate*size),size)
    output_frame = np.zeros([size,size,3],dtype = 'uint8')
    w_s = random.randint(0,size - rsize)
    h_s = random.randint(0,size - rsize)
    rotate_angel = random.randint(-angel,angel)
    window = transform.rotate(input_frame,rotate_angel)
    window = window[w_s:w_s+size,h_s:h_s+size,:]
    if flip:
        window = window[:,::-1,:]
    output_frame = np.floor(transform.resize(window,(size,size),mode='reflect') * 255)
    return output_frame

pos_sam_dir = 'imageData/'
aug_sam_dir = 'imageData_aug/'
sub_folder = os.listdir(pos_sam_dir)
for i in range(4):
    sub_file = sub_folder[i]

    label_dir = pos_sam_dir+'%s/'%sub_file
    aug_dir = aug_sam_dir+'%s/'%sub_file
    print(aug_dir)
    img_list = os.listdir(label_dir)
    for img_name in img_list:
        if img_name[-5:]=='.JPEG':
            for j in range(20):       
                aug_path = aug_dir+'aug%d-%s'%(j,img_name)
                this_img = imread(label_dir+img_name)
                if len(this_img.shape)!=3:
                    break
		this_img = transform.resize(this_img,(224,224))
                aug_img = data_aug(this_img)
                print(aug_path)
                imsave(aug_path, aug_img, format=None)


