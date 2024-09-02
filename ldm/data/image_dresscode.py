import os
import torch
import torchvision
import torch.utils.data as data
import torchvision.transforms.functional as F
import numpy as np
from PIL import Image

#加载Dress Code

class OpenImageDataset(data.Dataset):
    def __init__(self, state, dataset_dir, type="paired", mask_mv = None):
        """mask_mv: 控制是否采用多数值掩码,其内容是生成掩码的路径
                    defauklt: None , 'mask_unet_pair' ,'mask_unet_unpair'. 'mask_dm_pair',  'mask_dm_unpair'"""
        self.state = state              # train or test
        self.dataset_dir = dataset_dir  # /home/sd/zjh/Dataset/DressCode
        self.mask_mv = mask_mv

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
        with open(self.dataset_file, 'r') as f:
            for line in f.readlines():
                people, clothes, category = line.strip().split()
                if category == "0":
                    category = "upper_body"
                elif category == "1":
                    category = "lower_body"
                elif category == "2":
                    category = "dresses"
                people_path = os.path.join(self.dataset_dir, category, "images", people)
                clothes_path = os.path.join(self.dataset_dir, category, "images", clothes)

                #人物与衣物的具体路径（如果要找其他文件，需要替换掉路径里的 “images”）
                self.people_list.append(people_path)
                self.clothes_list.append(clothes_path)

        
    def __len__(self):
        return len(self.people_list)

    def __getitem__(self, index):

        #人物图像路径
        people_path = self.people_list[index]
        # /home/sd/zjh/Dataset/DressCode/upper_body/images/000000_0.jpg

        #衣物图像路径
        clothes_path = self.clothes_list[index]
        # /home/sd/zjh/Dataset/DressCode/upper_body/images/000000_1.jpg

        #densepose图像路径
        dense_path = people_path.replace("images", "dense")[:-5] + "5_uv.npz"
        # /home/sd/zjh/Dataset/DressCode/upper_body/dense/000000_5_uv.npz

        #掩码路径
        mask_path = people_path.replace("images", "mask")[:-3] + "png"
        # /home/sd/Harddisk/zjh/DressCode/upper_body/mask/000000_0.png
    

        # 加载图像
        img = Image.open(people_path).convert("RGB").resize((512, 512))
        img = torchvision.transforms.ToTensor()(img)

        refernce = Image.open(clothes_path).convert("RGB").resize((224, 224))
        refernce = torchvision.transforms.ToTensor()(refernce)

        mask = Image.open(mask_path).convert("L").resize((512, 512))
        mask = torchvision.transforms.ToTensor()(mask)
        mask = 1-mask

        densepose = np.load(dense_path)
        densepose = torch.from_numpy(densepose['uv'])
        densepose = torch.nn.functional.interpolate(densepose.unsqueeze(0), size=(512, 512), mode='bilinear', align_corners=True).squeeze(0)

        #TODO:这个地方可能根据全局情况进行一定修改
        #当前是假设每类衣物都有自己的mask_gen_path
        if self.mv != None:
            mask_name = people_path[-12:-5]+clothes_path[-12:-6]
            mask_gen_path = os.path.join(people_path.replace('images', self.mask_mv)[:-12], mask_name, ".png")
            mask_gen = Image.open(mask_gen_path).convert("L").resize((512, 512))
            mask_gen = torchvision.transforms.ToTensor()(mask)
            mask_gen = 1-mask
        else:
            pass            

        # 正则化
        img = torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img)
        refernce = torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                    (0.26862954, 0.26130258, 0.27577711))(refernce)
        # densepose = torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(densepose)


        #生成inpaint 
        if self.mask_mv == None:
            inpaint = img * mask
        else:
            mask_sum, mask = self.MultiValueMask(mask, mask_gen)
            inpaint = img * mask_sum

        # 生成 hint
        hint = torchvision.transforms.Resize((512, 512))(refernce)
        hint = torch.cat((hint,densepose),dim = 0)


        return {"GT": img,                  # [3, 512, 512]
                "inpaint_image": inpaint,   # [3, 512, 512]
                "inpaint_mask": mask,       # [1, 512, 512]
                "ref_imgs": refernce,       # [3, 224, 224]
                "hint": hint                # [5, 512, 512]
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

        #save_sum = torch.squeeze(mask_sum, dim=0)
        #save_sum = 255 * save_sum.cpu().numpy()
        #save_sum = Image.fromarray(save_sum.astype(np.uint8))
        #save_sum.save("test_image_sum.png")

        #save_mv =   torch.squeeze(mask_mv, dim=0)         
        #save_mv = 127 * save_mv.cpu().numpy()
        #save_mv = Image.fromarray(save_mv.astype(np.uint8))
        #save_mv.save("test_image.png")
        return mask_sum, mask_mv