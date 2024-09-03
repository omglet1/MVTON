import random
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from einops import rearrange
from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)
from ldm.models.diffusion.ddpm import DDPM
from ldm.models.diffusion.ddpm_mask import DDPM_Mask
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock
from ldm.modules.diffusionmodules.model import Model as UNet
from ldm.util import instantiate_from_config, default
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.ddim_mask import DDIMSampler_Mask

import torch
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LambdaLR
import pytorch_lightning as pl

def disabled_train(self, mode=True):
    return self



# ===================================================================
# 训练采用的主网络  应当可以选择是否采用diffusion
# 输入就两个，densepose与garment
# 网络
# ===================================================================
class Mask_Unet(pl.LightningModule):
    def __init__(self,
                 unet_config,
                 hint_channel = 6,
                 resolution = 512,
                 scale_factore = 1.0,
                 *args, **kwargs):
        #initial  basic model and diffusion paremeters
        self.num_timesteps_cond = 1  #TODO: If wouldn't use, delete
        super().__init__(*args, **kwargs)

        self.model = instantiate_from_config(unet_config)
        self.criterion = nn.CrossEntropyLoss()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.GroupNorm(32, 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.GroupNorm(32, 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.GroupNorm(32, 128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(resolution // 4))
        
        self.stem_hint = nn.Sequential(
            nn.Conv2d(hint_channel, 64, 7, 2, 3),
            nn.GroupNorm(32, 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.GroupNorm(32, 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.GroupNorm(32, 128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(resolution // 4))
        
        
        self.final_out = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1, 0),
            nn.GroupNorm(32, 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.GroupNorm(32, 64),
            nn.ReLU(inplace=True),            
            nn.ConvTranspose2d(64, 32, 4, 2, 1, 0),
            nn.GroupNorm(32, 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, 1)
        )

        #control parameters
        self.scale_factor = scale_factore



    #训练
    def training_step(self, batch, batch_idx):
        mask, garment, hint, __ = self.get_input(batch)     #加载数据
        loss = self(mask, garment, hint)                #计算损失
        self.log("loss",                                            # 记录损失
                     loss,                                  
                     prog_bar=True,
                     logger=True, 
                     on_step=True, 
                     on_epoch=True)
        self.log('lr_abs',                                          # 记录学习率
                     self.optimizers().param_groups[0]['lr'], 
                     prog_bar=True, 
                     logger=True, 
                     on_step=True, 
                     on_epoch=False)
        return loss


    #数据集加载
    @torch.no_grad()
    def get_input(self, batch):

        # 加载原始数据
        gt = batch['GT']
        garment = batch['garment']
        hint = batch['hint']
        name = batch['name']

        #数据格式
        gt = gt.to(memory_format=torch.contiguous_format).float()  
        garment = garment.to(memory_format=torch.contiguous_format).float()  
        hint = hint.to(memory_format=torch.contiguous_format).float()   

        out = [gt, garment, hint, name]

        return out     
        
    #计算损失
    def forward(self, gt, garment, hint):

        x = self.stem(garment)
        x = x + self.stem_hint(hint)
        output = self.model(x, timesteps = None, control = None)
        output = self.final_out(output)
        
        loss = self.criterion(output, gt)

        return loss


    # 优化器
    def configure_optimizers(self):
        # 学习率设置
        lr = self.learning_rate
        params =list(self.model.parameters())
        opt = torch.optim.AdamW(params, lr=lr,betas=(0.9, 0.999), weight_decay=0.01,)

        return opt

    #采样（预测）
    @torch.no_grad()
    def predict(self, batch, ddim_steps=50, ddim_eta=0. ,device=None):
        __, garment, hint, name = self.get_input(batch)
        garment = garment.to(device)
        hint = hint.to(device)

        x = self.stem(garment)
        x = x + self.stem_hint(hint)


        output = self.model(x, timesteps = None, control = None)
        output = F.softmax(output, dim=1)
        output = torch.argmax(output, dim=1, keepdim=True)
        return output, name



        