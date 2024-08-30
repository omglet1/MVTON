"""
该方法是用于cat-dm 虚拟视觉试衣模型的训练与测试
额外添加了一个初始化参数 mask_mv是用于获取不同掩码生成的结果
"""

import os
import torch
import torchvision
import torch.utils.data as data
import torchvision.transforms.functional as F
from PIL import Image

#加载VITON-HD

class OpenImageDataset(data.Dataset):
    def __init__(self, state, dataset_dir, type="paired", mask_mv = None):
        """mask_mv: 控制是否采用多数值掩码,其内容是生成掩码的路径
                    defauklt: None , 'mask_unet_pair' ,'mask_unet_unpair'. 'mask_dm_pair',  'mask_dm_unpair'"""
        self.state=state  #训练还是测试
        self.mask_mv = mask_mv # 是否采用多数值掩码
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

        # 确定路径
        img_path = os.path.join(self.dataset_dir, self.state, "image", person)
        reference_path = os.path.join(self.dataset_dir, self.state, "cloth", garment)

        mask_path = os.path.join(self.dataset_dir, self.state, "agnostic-mask", person[:-4]+"_mask.png")        #按需求修改                      
        densepose_path = os.path.join(self.dataset_dir, self.state, "image-densepose", person)
        
        # 加载图像 RGB形式读取图像并处理为tensor
        img = Image.open(img_path).convert("RGB").resize((512, 512))  #图像：512x152
        img = torchvision.transforms.ToTensor()(img)
        refernce = Image.open(reference_path).convert("RGB").resize((224, 224))  #图像：224x224
        refernce = torchvision.transforms.ToTensor()(refernce)
        mask = Image.open(mask_path).convert("L").resize((512, 512))  #图像：512x152
        mask = torchvision.transforms.ToTensor()(mask)
        mask = 1-mask   #最大的问题在于为何要调转色差  理由：目的是为了可以是inpaint图像保留衣物无关区域
        densepose = Image.open(densepose_path).convert("RGB").resize((512, 512))  #图像：512x512
        densepose = torchvision.transforms.ToTensor()(densepose)

        #如果采用多数值掩码，则需要对数据进行额外操作
        if self.mask_mv != None:
            pair_name = person[:-6]+garment[:-7]
            mask_gen_path = os.path.join(self.dataset_dir, self.state, self.mask_mv, pair_name+".png")
            mask_gen = Image.open(mask_gen_path).convert("L").resize((512, 512))
            mask_gen = torchvision.transforms.ToTensor()(mask_gen)
            mask_gen = 1-mask_gen
        else:
            pass

        # 正则化
        img = torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img)
        refernce = torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                    (0.26862954, 0.26130258, 0.27577711))(refernce)
        densepose = torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(densepose)

        # 生成 inpaint 和 hint
        if self.mask_mv == None:
            inpaint = img * mask  #需要画衣服的人的其他部位
        else:
            mask_sum, mask = self.MultiValueMask(mask, mask_gen)
            inpaint = img * mask_sum

        hint = torchvision.transforms.Resize((512, 512))(refernce) #提示图像 衣物+densepose
        hint = torch.cat((hint,densepose),dim = 0)

        return {"GT": img,                  # [3, 512, 512]
                "inpaint_image": inpaint,   # [3, 512, 512]
                "inpaint_mask": mask,       # [1, 512, 512]
                "ref_imgs": refernce,       # [3, 224, 224]
                "hint": hint,               # [6, 512, 512]
                }


    def MultiValueMask(self, Mask_person, Mask_garment): 
        """
        需要实现两个功能：
        1、将两个掩码合并,生成总的掩码覆盖范围
        2、按照逻辑构建一个多数值的掩码,

        注意: 两个掩码都是0代表目标衣物, 1代表背景与其他人物部分
        mask_person : 人物图像检测得到的衣物掩码
        mask_garment: 目标衣物图像对应到人身上的图像
        """
        #获取全部的掩码遮挡区域
        mask_sum = torch.logical_and(Mask_person, Mask_garment)

        mask_mv = torch.zeros_like(mask_sum)

        C, H, W =mask_mv.shape

        #获取多数值掩码
        for i in range(H):
            for j in range(W):
                value_person = Mask_person[0,i,j]
                value_garment = Mask_garment[0,i,j]

                if value_garment == 0:
                    mask_mv[0,i,j] = 0
                else:
                    if value_person == 0:
                        mask_mv[0,i,j] = 1
                    else :
                        mask_mv[0,i,j] = 2

        return mask_sum, mask_mv