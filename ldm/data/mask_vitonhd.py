"""
该函数仅用于mask 生成器的训练与测试
目标数据集是VitonHD
"""

import os
import torch
import torchvision
import torch.utils.data as data
import torchvision.transforms.functional as F
from PIL import Image

#加载VITON-HD

class OpenImageDataset_Mask(data.Dataset):
    def __init__(self, state, dataset_dir, type="paired"):
        self.state=state  #训练还是测试
        self.dataset_dir = dataset_dir  #数据集路径
        self.dataset_list = []

        #相应模式下的文件名字集合
        if state == "train":
            self.dataset_file = os.path.join(dataset_dir, "train_pairs.txt")
            with open(self.dataset_file, 'r') as f:
                for line in f.readlines():
                    person, garment = line.strip().split()
                    self.dataset_list.append([person, person])

        if state == "test":
            self.dataset_file = os.path.join(dataset_dir, "test_pairs.txt")
            if type == "unpaired":
                with open(self.dataset_file, 'r') as f:
                    for line in f.readlines():
                        person, garment = line.strip().split()
                        self.dataset_list.append([person, garment])

            if type == "paired":
                with open(self.dataset_file, 'r') as f:
                    for line in f.readlines():
                        person, garment = line.strip().split()
                        self.dataset_list.append([person, person])


    #数据集大小（多少配对人物与服饰）
    def __len__(self):
        return len(self.dataset_list)
    

    #返回相关数据（必须有，dataloader通过该函数得到相应数据）
    def __getitem__(self, index):

        person, garment = self.dataset_list[index]
        mask_num = str(person[:-6]+garment[:-7])

        category = person[-5]
        # 确定路径
        reference_path = os.path.join(self.dataset_dir, self.state, "cloth", garment)            #garment
        if self.state == 'train':
            mask_path = os.path.join(self.dataset_dir, self.state, "gt_cloth_warped_mask", person)   #mask    
        else:
            mask_path = os.path.join(self.dataset_dir, self.state, "cloth-mask", person)   #mask                    
        densepose_path = os.path.join(self.dataset_dir, self.state, "image-densepose", person)   #densepose
        keypoint_path = os.path.join(self.dataset_dir, self.state, "openpose_img", person[:-4]+"_rendered.png" )   #keypoint
        
        # 加载图像 RGB形式读取图像并处理为tensor
        refernce = Image.open(reference_path).convert("RGB").resize((512,512))  #衣物图像：256*256
        refernce = torchvision.transforms.ToTensor()(refernce)
        mask = Image.open(mask_path).convert("L").resize((128, 128))  #掩码图像：256*256
        mask = torchvision.transforms.ToTensor()(mask)
        mask = self.get_target(mask)  #[2, H, W] 
        densepose = Image.open(densepose_path).convert("RGB").resize((512,512))  #densepose图像：256*256
        densepose = torchvision.transforms.ToTensor()(densepose)
        keypoint = Image.open(keypoint_path).convert("RGB").resize((512,512))  #关键点图像：256*256
        keypoint = torchvision.transforms.ToTensor()(keypoint)

        # 正则化
        refernce = torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                    (0.26862954, 0.26130258, 0.27577711))(refernce)
        densepose = torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(densepose)
        keypoint = torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(keypoint)

        # 处理garment
        #refernce = torchvision.transforms.Resize((512, 512))(refernce) #提示图像 衣物
        hint = torch.cat((densepose, keypoint), dim = 0)

        return {"GT": mask,                 # [1, 512, 512] Mask  通道0代表无关背景，通道1代表扭曲的衣物
                "garment": refernce,        # [3, 512, 512] garment
                "hint": hint,               # [6, 512, 512] densepose  keypoint   
                "name": mask_num,       
                "category": category
                }
    

    def get_target(self, mask):
        mask_zero = torch.zeros_like(mask)
        mask_zero = mask_zero.repeat(2, 1, 1)  #[2, H, W]
        C,H,W = mask.shape
        for i in range(H):
            for j in range(W):
                if mask[0, i, j] == 0:
                    mask_zero[0, i, j] = 1
                else:
                    mask_zero[1, i, j] = 1
        return mask_zero