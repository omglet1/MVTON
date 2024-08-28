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


def disabled_train(self, mode=True):
    return self



# ===================================================================
# 训练采用的主网络  应当可以选择是否采用diffusion
# 输入就两个，densepose与garment
# 网络
# ===================================================================
class MaskLDM(DDPM_Mask):
    def __init__(self,
                 run_dm = False,
                 scale_factore = 1.0,
                 *args, **kwargs):
        #initial  basic model and diffusion paremeters
        self.num_timesteps_cond = 1  #TODO: If wouldn't use, delete
        super().__init__(*args, **kwargs)

        self.sigmoid = nn.Sigmoid()

        #control parameters
        self.run_dm = run_dm
        self.scale_factor = scale_factore


    #训练
    def training_step(self, batch, batch_idx):
        garment, hint, mask = self.get_input(batch)     #加载数据
        loss = self(garment, hint, mask)                #计算损失
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

        #数据格式
        gt = gt.to(memory_format=torch.contiguous_format).float()  
        garment = garment.to(memory_format=torch.contiguous_format).float()  
        hint = hint.to(memory_format=torch.contiguous_format).float()   

        out = [gt, garment, hint]

        return out     
        
    #计算损失
    def forward(self, gt, garment, hint):
        if self.run_dm == True:
            loss = self.forward_dm(gt, garment, hint)
        else:
            loss = self.forward_unet(gt, garment, hint)
            return loss

    def forward_unet(self, gt, garment, hint):
        """
        gt:      [B,1,H,W]
        garment: [B,3,H,W]
        hint:    [B,6,H,W]
        """
        output = self.model(garment, t = None, context = hint)
        loss = torch.binary_cross_entropy_with_logits(output, gt)
        return loss
    
    def forward_dm(self, gt, garment, hint):
        """
        gt:      [B,1,H,W]
        garment: [B,3,H,W]
        hint:    [B,6,H,W]
        """

        z = self.scale_factor * (gt.sample()).detach()            

        #随机时间 t
        t = torch.randint(0, self.num_timesteps, (z.shape[0], ), device=self.device).long()

        #随机加噪
        noise = torch.randn_like(z)
        z_noise = self.q_sample(x_start=z, t = t, noise = noise)
        z_noise = torch.cat((z_noise, garment), dim=1)

        #预测噪声
        output = self.model(z_noise, t, hint)

        #计算损失
        loss = self.get_loss(output, noise)

        return loss

    # 优化器
    def configure_optimizers(self):
        # 学习率设置
        lr = self.learning_rate
        params = list(self.model.parameters())
        opt = torch.optim.AdamW(params, lr=lr)

        return opt

    #采样（预测）
    @torch.no_grad()
    def predict(self, batch, ddim_steps=50, ddim_eta=0.):
        __, garment, hint = self.get_input(batch)

        if self.run_dm == False:
            output = self.demo_unet(garment, hint)
        else:
            output = self.demo_dm(garment, hint, ddim_steps, ddim_eta)

        output = self.sigmoid(output)

        output = torch.round(output)

        return output
    
    def demo_unet(self, garment, hint):
        output = self.model(garment, t = None, context = hint)
        return output

        
    def demo_dm(self, garment, hint, ddim_steps=50, ddim_eta=0.):
        ddim_sampler = DDIMSampler_Mask(self)    #TODO:添加扩散模型计划
        shape = (self.channels, self.image_size, self.image_size) 
        samples, __ =ddim_sampler(ddim_steps,
                                  garment.shape[0],
                                  shape,
                                  hint,
                                  garment,
                                  verbose=False, 
                                  eta=ddim_eta)


        x_samples = 1. / self.scale_factor * samples
        x_samples = torchvision.transforms.Resize([512, 512])(x_samples)

        return x_samples

        