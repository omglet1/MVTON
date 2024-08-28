import os
import torch
import argparse
import torchvision
import pytorch_lightning
import numpy as np

from PIL import Image
from torch import autocast
from einops import rearrange
from functools import partial
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.ddim_mask import DDIMSampler_Mask

def un_norm(x):
    return (x+1.0)/2.0

def un_norm_clip(x):
    x[0,:,:] = x[0,:,:] * 0.26862954 + 0.48145466
    x[1,:,:] = x[1,:,:] * 0.26130258 + 0.4578275
    x[2,:,:] = x[2,:,:] * 0.27577711 + 0.40821073
    return x

class DataModuleFromConfig(pytorch_lightning.LightningDataModule):
    def __init__(self, 
                 batch_size,                        # 1
                 test=None,                         # {...}
                 wrap=False,                        # False
                 shuffle=False,             
                 shuffle_test_loader=False,
                 use_worker_init_fn=False):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = batch_size * 2
        self.use_worker_init_fn = use_worker_init_fn
        self.wrap = wrap
        self.datasets = instantiate_from_config(test)
        self.dataloader = torch.utils.data.DataLoader(self.datasets, 
                                                      batch_size=self.batch_size,
                                                      num_workers=self.num_workers,
                                                      shuffle=shuffle,
                                                      worker_init_fn=None)



if __name__ == "__main__":
    # =============================================================
    # 处理 opt
    # =============================================================
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--base", type=str, default="configs/test.yaml")
    parser.add_argument("-c", "--ckpt", type=str, default="./model.ckpt")
    parser.add_argument("-s", "--seed", type=int, default=42)
    parser.add_argument("-d", "--ddim", type=int, default=64)
    opt = parser.parse_args()

    # =============================================================
    # 设置 seed
    # =============================================================
    seed_everything(opt.seed)

    # =============================================================
    # 初始化 config
    # =============================================================
    config = OmegaConf.load(f"{opt.base}")

    # =============================================================
    # 加载 dataloader
    # =============================================================
    data = instantiate_from_config(config.data)
    print(f"{data.__class__.__name__}, {len(data.dataloader)}")

    # =============================================================
    # 加载 model
    # =============================================================
    model = instantiate_from_config(config.model)
    model.load_state_dict(torch.load(opt.ckpt, map_location="cpu")["state_dict"], strict=False)
    model.cuda()
    model.eval()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler_Mask(model)

    # =============================================================
    # 设置精度
    # =============================================================
    precision_scope = autocast

    # =============================================================
    # 开始测试
    # =============================================================
    if os.path.exists("results/generate_mask") == False:
        os.makedirs("results/generate_mask")


    with torch.no_grad():
        with precision_scope("cuda"):
            for i,batch in enumerate(data.dataloader):
                # 加载数据

                truth = batch["GT"].to(torch.float16).to(device)
                hint = batch["hint"].to(torch.float16).to(device)
                garment = batch["garment"].to(torch.float16).to(device)


                # 数据处理


                shape = (model.channels, model.image_size, model.image_size)
                # 预测结果
                samples, _ = sampler.sample(S=opt.ddim,
                                                 batch_size=1,
                                                 shape=shape,
                                                 pose=hint,
                                                 conditioning=garment,
                                                 verbose=False,
                                                 eta=0)
                samples = 1. / model.scale_factor * samples
                x_samples = model.first_stage_model.decode(samples[:,:4,:,:])

                x_samples_ddim = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
                x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
                x_checked_image=x_samples_ddim
                x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)                
                # 保存图像
                all_img=[]
                # all_img.append(un_norm(truth[0]).cpu())
                # all_img.append(un_norm(inpaint[0]).cpu())
                # all_img.append(un_norm_clip(torchvision.transforms.Resize([512,512])(reference)[0].cpu()))

                x_checked_image_torch = torch.nn.functional.interpolate(x_checked_image_torch.float(), size=[512,384])

                
                all_img.append(x_checked_image_torch[0])

                grid = torch.stack(all_img, 0)
                grid = torchvision.utils.make_grid(grid)
                grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                img = Image.fromarray(grid.astype(np.uint8))
                img.save("results/generate_mask/"+str(i).rjust(5,"0")+"_00.png")
