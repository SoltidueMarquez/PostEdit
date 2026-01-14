from diffusers import DiffusionPipeline
import torch

# 预下载模型
pipe = DiffusionPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7")