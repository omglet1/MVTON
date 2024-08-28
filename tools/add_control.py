import sys
import os
import torch
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config

input_path = sys.argv[1]
output_path = sys.argv[2]
config_path = sys.argv[3]

'''
# Determine whether the weight needs to be controlled?
# In simple terms, in the model, all weights that begin with "control_" need to be controlled. For instance, "control_model.middle_block_out.0.bias" need to be controlled.
# In code,  "control_model.middle_block_out.0.bias" belongs to "self.control_model"
# Return True, "model.middle_block_out.0.bias"

# 将SDM(stable diffusion model) 对应架构参数取出,作为control Net部分参数
# 不存在的部分,则采用初始化定义参数
'''

#从name中去除前面部分名字
#要求name前几位必须是parent_name
#且name长度一定大于parent_name
def get_node_name(name, parent_name):
    if len(name) <= len(parent_name):
        return False, ''
    p = name[:len(parent_name)]
    if p != parent_name:
        return False, ''
    return True, name[len(parent_name):]

# Load model
#构建controlNet部分
configs = OmegaConf.load(config_path)
model = instantiate_from_config(configs["model"])
scratch_dict = model.state_dict()

# Load pre-trained weights
#加载预训练权重
pretrained_weights = torch.load(input_path, map_location='cpu')['state_dict']  #有报错

# Generate target weights
target_dict = {}

#遍历controlNet参数的关键字
for k in scratch_dict.keys():

    #去掉 controlnet参数关键词 开头的 "control_"
    is_control, name = get_node_name(k, 'control_')

    #按照需不需要控制定义不同名字，并于从不同权重文件中提取
    # Need to be controlled
    if is_control:
        copy_k = 'model.diffusion_' + name
    # Don't need to be controlled
    else:
        copy_k = k
    
    # The weights that exist in pbe, copy from it
    #从预训练权重或随机初始化权重中提取参数
    #在原有基础上加了controlnet部分参数
    if copy_k in pretrained_weights:
        target_dict[k] = pretrained_weights[copy_k].clone()
    # The weights not existing in pbe, set to zero
    else:
        target_dict[k] = scratch_dict[k].clone()
        print(f'These weights are newly added: {k}')


# Save
#存储模型
model.load_state_dict(target_dict, strict=True)
torch.save(model.state_dict(), output_path)
print('Done.')