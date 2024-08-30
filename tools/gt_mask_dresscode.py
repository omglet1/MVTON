import cv2
import sys
import json
from os import path as osp
import os
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

input_path = sys.argv[1]   # 对应类别的数据所在路劲    datasets/dresscode/dresses 
mask_path = sys.argv[2]    # 对应掩码放置路径  datasets/dresscode/dresses/gt_mask

category = input_path.split('/')[-1]  #衣物类别标签


def get_img_agnostic( label, category):
    parse_array = np.array(label)

    parse_dress = ((parse_array == 4).astype(np.float32) +
                   (parse_array == 5).astype(np.float32) +
                   (parse_array == 6).astype(np.float32) +
                   (parse_array == 7).astype(np.float32) +
                   (parse_array == 8).astype(np.float32))

    
    parse_upper = ((parse_array == 4).astype(np.float32))


    parse_lower = ((parse_array == 5).astype(np.float32) +
                   (parse_array == 6).astype(np.float32) +
                   (parse_array == 7).astype(np.float32) +
                   (parse_array == 8).astype(np.float32))
  


    if category == 'dresses':
        return parse_dress  
    elif category == 'upper_body':
        return parse_upper 
    elif category == 'lower_body':
        return parse_lower   





#创建存储目录
if os.path.exists(mask_path) == False:
    os.makedirs(mask_path, exist_ok=True)

for im_name in tqdm(os.listdir(osp.join(input_path, 'images'))):
    if im_name.endswith('1.jpg'):
        continue

    label_name = im_name.replace('0.jpg', '4.png')
    im_label = Image.open(osp.join(input_path, 'label_maps',label_name)).convert('P')

    agnostic = get_img_agnostic( im_label, category)

    agnostic = (agnostic == 1).astype(int)
    agnostic = agnostic * 255.0
    agnostic = Image.fromarray(np.uint8(agnostic)).convert('L')
    agnostic.save(osp.join(mask_path, im_name.replace('.jpg', '.png')))
    
print("Done")