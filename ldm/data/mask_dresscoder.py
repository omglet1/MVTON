"""
该函数仅使用于mask生成器的训练与测试
目标数据集是VitonHD
"""
import os
import torch
import torchvision
import torch.utils.data as data
import torchvision.transforms.functional as F
import numpy as np
from PIL import Image

#加载Dress Code

class OpenImageDataset(data.Dataset):
    def __init__(self, state, dataset_dir, type="paired"):
        #基本参数存储
        self.state = state              # train or test
        self.dataset_dir = dataset_dir  # /home/sd/zjh/Dataset/DressCode

        # 确定状态
        if state == "train":
            self.dataset_file = os.path.join(dataset_dir, "train_pairs.txt")
        if state == "test":
            if type == "unpaired":
                self.dataset_file = os.path.join(dataset_dir, "test_pairs_unpaired.txt")
            if type == "paired":
                self.dataset_file = os.path.join(dataset_dir, "test_pairs_paired.txt")

        # 加载数据集
        self.people_list = []
        self.clothes_list = []
        self.category_list = []
        with open(self.dataset_file, 'r') as f:
            for line in f.readlines():
                people, clothes, category = line.strip().split()  #配对文件中，每列分别对应人物、衣物与衣物类别
                #获取各个衣物所对应的
                if category == "0":
                    category = "upper_body"
                elif category == "1":
                    category = "lower_body"
                elif category == "2":
                    category = "dresses"
                #记录人物与服饰的路径 （与viton存在差异，一个是分文件夹存储，一个这是尾号设0为人，设1为衣服）
                people_path = os.path.join(self.dataset_dir, category, "images", people)
                clothes_path = os.path.join(self.dataset_dir, category, "images", clothes)
                self.people_list.append(people_path)
                self.clothes_list.append(clothes_path)
                self.category_list.append(category)

        
    def __len__(self):
        return len(self.people_list)

    def __getitem__(self, index):

        #提取服饰类别属性
        categoty = self.category_list[index]

        #各类数据的路径
        people_path = self.people_list[index]   #人物路径
        # /home/sd/zjh/Dataset/DressCode/upper_body/images/000000_0.jpg

        clothes_path = self.clothes_list[index]    #衣物路径
        # /home/sd/zjh/Dataset/DressCode/upper_body/images/000000_1.jpg

        dense_path = clothes_path.replace("images", "dense")[:-5] + "5_uv.npz"  #densepose路径
        # /home/sd/zjh/Dataset/DressCode/upper_body/dense/000000_5_uv.npz

        skeleton_path = clothes_path.replace("images", "skeleton")[:-5] + "5.jpg"      #关键点连线图
        #/home/jwliu/MVTON/datasets/dresscode/dresses/skeletons/020714_5.jpg

        mask_path = clothes_path.replace("images", "gt_mask")[:-5] + "0.png"  #语义分割标签图（提取GT_mask）
        # /home/jwliu/MVTON/datasets/dresscode/dresses/gt_mask/020714_0.png

        #获取人物衣物配对名字
        mask_num = people_path[-12:-5]+clothes_path[-12:-6]

        # 加载图像

        #衣物图像
        refernce = Image.open(clothes_path).convert("RGB").resize((512, 512))
        refernce = torchvision.transforms.ToTensor()(refernce)

        #分割图像并处理成相应的掩码
        mask = Image.open(mask_path).convert("L").resize((512, 512))
        mask = torchvision.transforms.ToTensor()(mask)

        #densepose
        densepose = np.load(dense_path)
        densepose = torch.from_numpy(densepose['uv'])
        densepose = torch.nn.functional.interpolate(densepose.unsqueeze(0), size=(512, 512), mode='bilinear', align_corners=True).squeeze(0)

        #skeleton
        skeleton = Image.open(skeleton_path).convert("RGB").resize((512,512))
        skeleton = torchvision.transforms.ToTensor(skeleton)

        
        # 正则化
        refernce = torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                    (0.26862954, 0.26130258, 0.27577711))(refernce)
        # densepose = torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(densepose)

        # 生成 inpaint 和 hint

        hint = torch.cat((densepose, skeleton),dim = 0)


        return {"GT": mask,                  # [1, 512, 512] Mask
                "garment": refernce,         # [3, 224, 224] Garment
                "hint": hint,                # [6, 512, 512] Densepose keypoint
                "name": mask_num
                }


