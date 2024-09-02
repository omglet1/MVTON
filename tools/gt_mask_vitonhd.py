import cv2
import sys
import json
from os import path as osp
import os
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm


#python /home/jeremy/MVTON/tools/gt_mask_vitonhd.py datasets/vitonhd/test datasets/vitonhd/test/gt_cloth_warped_mask

input_path = sys.argv[1]   # 对应类别的数据所在路劲    datasets/dresscode/dresses 
mask_path = sys.argv[2]    # 对应掩码放置路径  datasets/dresscode/dresses/gt_mask


def get_img_agnostic( label):
    parse_array = np.array(label)

    #错误属性4，6，7，8，9，10颈，11，12，13脸，14左臂，15右臂，16，17，18，19
    #上身衣物属性为5
    parse_upper = ((parse_array == 5).astype(np.float32))

    return parse_upper





#创建存储目录
if os.path.exists(mask_path) == False:
    os.makedirs(mask_path, exist_ok=True)


for im_name in tqdm(os.listdir(osp.join(input_path, 'image'))):

    im_name = im_name.replace('jpg', 'png')
    im_label = Image.open(osp.join(input_path, 'image-parse-v3', im_name)).convert('P')

    agnostic = get_img_agnostic( im_label)

    agnostic = (agnostic == 1).astype(int)
    agnostic = agnostic * 255.0
    agnostic = Image.fromarray(np.uint8(agnostic)).convert('L')
    agnostic.save(osp.join(mask_path, im_name.replace('png', 'jpg')))
    
print("Done")