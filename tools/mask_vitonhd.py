import sys
import json
from os import path as osp
import os
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

input_path = sys.argv[1]
output_path = sys.argv[2]

#关键点采用openpose-25定义

def get_img_agnostic(img, parse, pose_data): #img只有黑色
    #parse：语义分割结果（一维图片，但是个像素素值不同以区分物体类别）
    parse_array = np.array(parse)
    #人物头部
    parse_head  = ((parse_array == 4 ).astype(np.float32) +
                   (parse_array == 13).astype(np.float32))  
    #人物下半身
    parse_lower = ((parse_array == 9 ).astype(np.float32) +
                   (parse_array == 12).astype(np.float32) +
                   (parse_array == 16).astype(np.float32) +
                   (parse_array == 17).astype(np.float32) +
                   (parse_array == 18).astype(np.float32) +
                   (parse_array == 19).astype(np.float32))
    
    #衣物不可知图像绘制
    agnostic = img.copy()  #目标人物
    agnostic_draw = ImageDraw.Draw(agnostic)  #在该图像上进行绘制
    

    #得到关键点坐标间的距离（肢体长度）（OPENPOSE）
    length_a = np.linalg.norm(pose_data[5]  - pose_data[2])  #人物肩宽
    length_b = np.linalg.norm(pose_data[12] - pose_data[9])  #人物腿长
    point = (pose_data[9] + pose_data[12]) / 2        #mid-hip
    pose_data[9]  = point + (pose_data[9]  - point) / length_b * length_a   #右臀位置改动
    pose_data[12] = point + (pose_data[12] - point) / length_b * length_a   #左臀位置改动
    r = int(length_a / 16) + 1   #以肩宽作为一个遮盖的基本单位长度
    

    # mask arms
    #左键到右肩上画一条白线
    agnostic_draw.line([tuple(pose_data[i]) for i in [2, 5]], 'white', width=r*10)  
    #在肩部画圆形遮挡
    for i in [2, 5]:
        pointx, pointy = pose_data[i]  #左键或右肩位置
        agnostic_draw.ellipse((pointx-r*5, pointy-r*5, pointx+r*5, pointy+r*5), 'white', 'white') #画一个圆形遮挡
    #再俩臂其他关键位置画遮挡
    for i in [3, 4, 6, 7]:
        #如果该关键点为连接节点不存在，则跳过
        if (pose_data[i - 1, 0] == 0.0 and pose_data[i - 1, 1] == 0.0) or (pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
            continue
        #连线+关键点画圆遮挡方式
        agnostic_draw.line([tuple(pose_data[j]) for j in [i - 1, i]], 'white', width=r*10)
        pointx, pointy = pose_data[i]
        agnostic_draw.ellipse((pointx-r*5, pointy-r*5, pointx+r*5, pointy+r*5), 'white', 'white')


    # mask torso  躯干遮挡
    #关键点位置遮挡
    for i in [9, 12]:
        pointx, pointy = pose_data[i]
        agnostic_draw.ellipse((pointx-r*3, pointy-r*6, pointx+r*3, pointy+r*6), 'white', 'white')
    #躯干花了个方方块遮挡+宽边连线
    agnostic_draw.line([tuple(pose_data[i]) for i in [2, 9]], 'white', width=r*6)
    agnostic_draw.line([tuple(pose_data[i]) for i in [5, 12]], 'white', width=r*6)
    agnostic_draw.line([tuple(pose_data[i]) for i in [9, 12]], 'white', width=r*12)
    agnostic_draw.polygon([tuple(pose_data[i]) for i in [2, 5, 12, 9]], 'white', 'white')


    # mask neck
    #颈部方形遮挡
    pointx, pointy = pose_data[1]
    agnostic_draw.rectangle((pointx-r*7, pointy-r*7, pointx+r*7, pointy+r*7), 'white', 'white')
    #把颈部与下半身遮挡恢复
    agnostic.paste(img, None, Image.fromarray(np.uint8(parse_head * 255), 'L'))
    agnostic.paste(img, None, Image.fromarray(np.uint8(parse_lower * 255), 'L'))

    return agnostic

os.makedirs(output_path, exist_ok=True)

#遍历人物图像，开始加掩码
for im_name in tqdm(os.listdir(osp.join(input_path, 'image'))): #加载图像名（无路径）
    
    # Load pose image
    # 加载关键点文件名
    pose_name = im_name.replace('.jpg', '_keypoints.json')
    try:
        #读取关键点文件
        with open(osp.join(input_path, 'openpose_json', pose_name), 'r') as f:
            pose_label = json.load(f)   #加载标签
            pose_data = pose_label['people'][0]['pose_keypoints_2d'] #加载关键点位置
            pose_data = np.array(pose_data) #np
            pose_data = pose_data.reshape((-1, 3))[:, :2]  #处理为[num_K, 2]保留x与y轴坐标
    except IndexError:
        print(pose_name)
        continue

    # Load parsing image
    #加载语义分割图像
    label_name = im_name.replace('.jpg', '.png')
    im_label = Image.open(osp.join(input_path, 'image-parse-v3', label_name))


    # Generate agnostic image and save
    #生成掩码并保留
    agnostic = get_img_agnostic(Image.new("RGB", (768, 1024), (0, 0, 0)), im_label, pose_data)
    agnostic.save(osp.join(output_path, im_name.replace('.jpg', '.png')))

print("Done")