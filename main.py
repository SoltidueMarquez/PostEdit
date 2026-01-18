import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
import os
import yaml
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
import ptp_utils
from forward_operator import get_operator
from typing import Optional, Union, Tuple, List, Dict
import torch.nn.functional as nnf
import abc
import seq_aligner
import pandas as pd
from nltk.corpus import stopwords
import argparse


# 全局常量配置
LOW_RESOURCE = False  # 是否处于低资源模式（影响注意力层处理）
NUM_DDIM_STEPS = 5    # DDIM 采样步数（此处针对 LCM 模型设置较小）
GUIDANCE_SCALE = 8.0  # 分类器自由引导 (CFG) 的缩放系数
GUIDANCE_SCALE_MAX = 8.0
GUIDANCE_SCALE_MIN = 8.0
MAX_NUM_WORDS = 77    # CLIP 文本编码器的最大 Token 数量


class LocalBlend:
    """
    局部融合类：用于根据交叉注意力图（Cross-Attention Maps）生成掩码，
    从而将编辑操作限制在特定的词语对应的区域，保持其他区域不变。
    """
    
    def get_mask(self, maps, alpha, use_pool):
        """
        根据注意力图生成二值化掩码
        maps: 注意力图张量
        alpha: 对应词语的权重
        use_pool: 是否使用最大池化扩展掩码范围
        """
        k = 1
        # 计算加权平均后的注意力图
        maps = (maps * alpha).sum(-1).mean(1)
        if use_pool:
            # 使用最大池化来平滑和扩展掩码边界
            maps = nnf.max_pool2d(maps, (k * 2 + 1, k * 2 +1), (1, 1), padding=(k, k))
        mask = nnf.interpolate(maps, size=(64, 64)) # 缩放到潜空间大小
        # 归一化处理
        mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
        # 根据阈值生成二值掩码
        mask = mask.gt(self.th[1-int(use_pool)])
        mask = mask[:1] + mask
        return mask
    
    def __call__(self, x_t, attention_store):
        """
        在扩散步骤中执行融合操作
        x_t: 当前步的潜变量 [batch_size, channels, h, w]
        attention_store: 存储的注意力图数据
        """
        self.counter += 1
        if self.counter > self.start_blend:
           
            # 提取不同层级的交叉注意力图
            maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
            maps = [item.reshape(self.alpha_layers.shape[0], -1, 1, 16, 16, MAX_NUM_WORDS) for item in maps]
            maps = torch.cat(maps, dim=1)
            # 生成融合掩码
            mask = self.get_mask(maps, self.alpha_layers, True)
            if self.substruct_layers is not None:
                # 处理需要排除的区域（减法操作）
                maps_sub = ~self.get_mask(maps, self.substruct_layers, False)
                mask = mask * maps_sub
            mask = mask.float()
            # 将编辑后的潜变量与原始潜变量根据掩码融合
            # x_t[:1] 是原始图像的潜变量，后面是编辑后的
            x_t = x_t[:1] + mask * (x_t - x_t[:1])
        return x_t
    
    def __init__(self, prompts: List[str], words: [List[List[str]]], substruct_words=None, start_blend=0.2, th=(.3, .3)):
        """
        初始化融合器
        prompts: 提示词列表
        words: 需要保留/融合的词语
        substruct_words: 需要排除的词语
        start_blend: 开始融合的时间步比例
        th: 掩码生成的阈值
        """
        alpha_layers = torch.zeros(len(prompts),  1, 1, 1, 1, MAX_NUM_WORDS)
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if type(words_) is str:
                words_ = [words_]
            for word in words_:
                # 获取词语在提示词中的索引
                ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                alpha_layers[i, :, :, :, :, ind] = 1
        
        if substruct_words is not None:
            substruct_layers = torch.zeros(len(prompts),  1, 1, 1, 1, MAX_NUM_WORDS)
            for i, (prompt, words_) in enumerate(zip(prompts, substruct_words)):
                if type(words_) is str:
                    words_ = [words_]
                for word in words_:
                    ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                    substruct_layers[i, :, :, :, :, ind] = 1
            self.substruct_layers = substruct_layers.to(device)
        else:
            self.substruct_layers = None
        self.alpha_layers = alpha_layers.to(device)
        self.start_blend = int(start_blend * NUM_DDIM_STEPS)
        self.counter = 0 
        self.th=th


class AttentionControl(abc.ABC):
    """
    注意力控制器基类：用于在 UNet 的前向传播中捕获或修改注意力图。
    """
    
    def step_callback(self, x_t):
        """每一步采样后的回调函数"""
        return x_t
    
    def between_steps(self):
        """两个采样步之间的处理"""
        return
    
    @property
    def num_uncond_att_layers(self):
        """获取无条件引导的注意力层数量"""
        return self.num_att_layers if LOW_RESOURCE else 0
    
    @abc.abstractmethod
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        """定义如何处理注意力图"""
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        """
        Hook 函数，会被注入到 UNet 的注意力层中
        attn: 当前层的注意力权重
        is_cross: 是否为交叉注意力
        place_in_unet: 处于 UNet 的哪个位置 (down, mid, up)
        """
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                # 在非低资源模式下，batch 通常包含 [uncond, cond]
                # 我们只处理有条件部分（后半部分）
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        # 当遍历完所有层后，重置层计数并进入下一步
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn
    
    def reset(self):
        """重置状态"""
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0


class AttentionStore(AttentionControl):
    """
    注意力存储器：用于收集并保存扩散过程中的注意力图。
    """

    @staticmethod
    def get_empty_store():
        """初始化一个空的存储字典"""
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        """保存当前层的注意力图"""
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # 避免内存开销过大，只保存分辨率较低的图
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        """在每步采样结束后，将当前步的注意力图累加到全局存储中"""
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        """计算平均注意力图"""
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention


    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}


class AttentionControlEdit(AttentionStore, abc.ABC):
    """
    编辑控制器基类：结合了注意力存储与修改功能，用于实现图像编辑。
    """
    
    def step_callback(self, x_t):
        """如果启用了局部融合，则在每步后调用"""
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store)
        return x_t
        
    def replace_self_attention(self, attn_base, att_replace, place_in_unet):
        """替换自注意力图（Self-Attention），用于保持原始图像的结构"""
        if att_replace.shape[2] <= 32 ** 2:
            attn_base = attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
            return attn_base
        else:
            return att_replace
    
    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, att_replace):
        """定义如何替换交叉注意力图（需要子类实现）"""
        raise NotImplementedError
    
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        """核心处理逻辑：决定何时以及如何替换注意力权重"""
        super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet)
        # 根据设置的时间步范围决定是否执行替换
        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
            h = attn.shape[0] // (self.batch_size)
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            attn_base, attn_repalce = attn[0], attn[1:] # 分离基准提示词和目标提示词的注意力
            if is_cross:
                # 交叉注意力替换逻辑
                alpha_words = self.cross_replace_alpha[self.cur_step]
                attn_repalce_new = self.replace_cross_attention(attn_base, attn_repalce) * alpha_words + (1 - alpha_words) * attn_repalce
                attn[1:] = attn_repalce_new
            else:
                # 自注意力替换逻辑
                attn[1:] = self.replace_self_attention(attn_base, attn_repalce, place_in_unet)
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        return attn
    
    def __init__(self, prompts, num_steps: int,
                 cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
                 self_replace_steps: Union[float, Tuple[float, float]],
                 local_blend: Optional[LocalBlend]):
        """
        初始化编辑控制器
        cross_replace_steps: 交叉注意力替换的时间步比例
        self_replace_steps: 自注意力替换的时间步比例
        """
        super(AttentionControlEdit, self).__init__()
        self.batch_size = len(prompts)
        # 获取词语级别的注意力混合权重
        self.cross_replace_alpha = ptp_utils.get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps, tokenizer).to(device)
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        self.local_blend = local_blend


class AttentionReplace(AttentionControlEdit):
    """
    注意力替换类：用于直接用新提示词映射后的注意力图替换旧图。
    常用于词语替换（Word Swap）场景。
    """

    def replace_cross_attention(self, attn_base, att_replace):
        """利用映射矩阵（mapper）重组注意力分布"""
        return torch.einsum('hpw,bwn->bhpn', attn_base, self.mapper)
      
    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None):
        super(AttentionReplace, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        # 获取词语对齐映射（基于 Levenshtein 距离或词法分析）
        self.mapper = seq_aligner.get_replacement_mapper(prompts, tokenizer).to(device)
        

class AttentionRefine(AttentionControlEdit):
    """
    注意力细化类：用于在原图基础上添加新属性，而不是完全替换对象。
    """

    def replace_cross_attention(self, attn_base, att_replace):
        """通过混合原注意力和新注意力来实现细节细化"""
        attn_base_replace = attn_base[:, :, self.mapper].permute(2, 0, 1, 3)
        attn_replace = attn_base_replace * self.alphas + att_replace * (1 - self.alphas)
        # attn_replace = attn_replace / attn_replace.sum(-1, keepdims=True)
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None):
        super(AttentionRefine, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        # 获取细化映射
        self.mapper, alphas = seq_aligner.get_refinement_mapper(prompts, tokenizer)
        self.mapper, alphas = self.mapper.to(device), alphas.to(device)
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])


class AttentionReweight(AttentionControlEdit):
    """
    注意力重加权类：通过调整特定词语的注意力权重来改变其在图中的显著程度。
    """

    def replace_cross_attention(self, attn_base, att_replace):
        if self.prev_controller is not None:
            attn_base = self.prev_controller.replace_cross_attention(attn_base, att_replace)
        # 对指定词语对应的通道进行加权
        attn_replace = attn_base[None, :, :, :] * self.equalizer[:, None, None, :]
        # attn_replace = attn_replace / attn_replace.sum(-1, keepdims=True)
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float, equalizer,
                local_blend: Optional[LocalBlend] = None, controller: Optional[AttentionControlEdit] = None):
        super(AttentionReweight, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.equalizer = equalizer.to(device)
        self.prev_controller = controller


def get_equalizer(text: str, word_select: Union[int, Tuple[int, ...]], values: Union[List[float],
                  Tuple[float, ...]]):
    """
    生成权重均衡器：用于为特定的词语分配不同的权重（Reweight）。
    text: 完整的提示词
    word_select: 要调整权重的词语（或索引）
    values: 对应的权重值
    """
    if type(word_select) is int or type(word_select) is str:
        word_select = (word_select,)
    equalizer = torch.ones(1, 77)
    
    for word, val in zip(word_select, values):
        inds = ptp_utils.get_word_inds(text, word, tokenizer)
        equalizer[:, inds] = val
    return equalizer


def make_controller(prompts: List[str], is_replace_controller: bool, cross_replace_steps: Dict[str, float], self_replace_steps: float, blend_words=None, equilizer_params=None) -> AttentionControlEdit:
    """
    工厂函数：根据配置创建相应的注意力控制器实例。
    """
    if blend_words is None:
        lb = None
    else:
        lb = LocalBlend(prompts, blend_words)
    if is_replace_controller:
        controller = AttentionReplace(prompts, NUM_DDIM_STEPS, cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, local_blend=lb)
    else:
        controller = AttentionRefine(prompts, NUM_DDIM_STEPS, cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, local_blend=lb)
    if equilizer_params is not None:
        eq = get_equalizer(prompts[1], equilizer_params["words"], equilizer_params["values"])
        controller = AttentionReweight(prompts, NUM_DDIM_STEPS, cross_replace_steps=cross_replace_steps,
                                       self_replace_steps=self_replace_steps, equalizer=eq, local_blend=lb, controller=controller)
    return controller


def load_512(image_path, left=0, right=0, top=0, bottom=0):
    """
    加载图像并将其裁剪/缩放至 512x512 分辨率。
    """
    if type(image_path) is str:
        image = np.array(Image.open(image_path))[:, :, :3]
    else:
        image = image_path
    h, w, _ = image.shape
    left = min(left, w-1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h-bottom, left:w-right]
    h, w, _ = image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]
    image = np.array(Image.fromarray(image).resize((512, 512)))
    return image


class LangevinDynamics(nn.Module):
    """
    郎之万动力学（Langevin Dynamics）：用于在潜空间进行采样优化，
    通过结合误差项和正则项来指导潜变量的更新，增强生成结果的保真度。
    """
    def __init__(self, num_steps, lr, tau=0.01, lr_min_ratio=0.01):
        super().__init__()
        self.num_steps = num_steps # 迭代步数
        self.lr = float(lr)        # 学习率
        self.tau = tau             # 噪声项系数
        self.lr_min_ratio = lr_min_ratio

    def sample(self, x0hat, operator, measurement, sigma, ratio, verbose=False, scores=None, steps=None):
        """
        采样函数：根据算子（operator）和测量值（measurement）优化潜变量。
        x0hat: 初始猜测的 x0
        operator: 前向算子（如模糊、掩码等）
        measurement: 实际观测到的测量值（退化图）
        sigma: 当前噪声水平
        """
        num_steps = self.num_steps if steps is None else steps
        print("根据算子（operator）和测量值（measurement）优化潜变量:")
        pbar = tqdm(range(num_steps), desc="      Langevin Optim", leave=False) if verbose else range(num_steps)
        lr = self.get_lr(ratio)
        x = x0hat.clone().detach().requires_grad_(True)
        optimizer = torch.optim.SGD([x], lr)
        for _ in pbar:
            optimizer.zero_grad()
            # 损失函数包含两部分：
            # 1. 测量误差项（Data Fidelity）：保证优化后的结果经过算子后与测量值一致
            loss = operator.error(x, measurement).sum() / (2 * self.tau ** 2)
            # 2. 正则项（Prior）：保证优化后的结果不偏离初始猜测太远
            loss += ((x - x0hat.detach()) ** 2).sum() / (2 * sigma ** 2)
            loss.backward()
            optimizer.step()
            # 添加随机扰动（郎之万项）
            with torch.no_grad():
                epsilon = torch.randn_like(x)
                x.data = x.data + np.sqrt(2 * lr) * epsilon

            if torch.isnan(x).any():
                return torch.zeros_like(x)

        return x.detach()

    def get_lr(self, ratio, p=1):
        """动态调整学习率"""
        multiplier = (1 ** (1 / p) + ratio * (self.lr_min_ratio ** (1 / p) - 1 ** (1 / p))) ** p
        return multiplier * self.lr


class NullInversion: 
    """
    空反向传播类（Null Inversion）：负责图像的编码、反演以及最终的编辑采样过程。
    它协调了 VAE、UNet 以及 LangevinDynamics 的交互。
    """
    def __init__(self, model, lgvd_config):
        self.model = model
        self.tokenizer = self.model.tokenizer
        self.prompt = None
        self.context = None
        self.lgvd = LangevinDynamics(**lgvd_config)

    def get_start(self, ref, starting_timestep=999, double=False):
        """
        根据参考图像（潜变量）添加指定步数的噪声，生成去噪过程的初始起点。
        """
        # 1. 如果 double 为 True，则将参考潜变量在 Batch 维度复制一份。
        # 这样可以同时并行处理两个分支：一个是保持原样的重构分支，一个是进行修改的编辑分支。
        if double: ref = torch.cat([ref] * 2) # [original, editted]
        
        # 2. 生成与潜变量形状完全一致的标准正态分布噪声
        noise = torch.randn(ref.shape).to(device)
        
        # 3. 确定起始的时间步（Timestep）
        timestep = torch.tensor(starting_timestep)
        
        # 4. 调用扩散模型的调度器（Scheduler）执行“前向加噪”：
        # 根据公式 x_t = sqrt(alpha_t)*x_0 + sqrt(1-alpha_t)*noise，
        # 将干净的潜变量 ref 混合噪声，得到第 T 步时的模糊状态 x_start。
        x_start = self.model.scheduler.add_noise(ref, noise, timestep)
        
        return x_start
    
    def prev_step(self, timestep, sample, operator, measurement, annel_interval=100, w = 0.25):
        """
        在采样循环中执行反向步处理，并结合郎之万动力学（Langevin Dynamics）进行迭代优化。
        这是 PostEdit 的核心，用于在去噪过程中强制让生成结果符合测量值（y）。
        """
        # 1. 计算总的迭代步数
        num_steps = int((timestep - 1) / annel_interval)
        print("根据 DDIM 采样循环预测潜变量:")
        
        # 使用进度条显示扩散步的进度
        step_pbar = tqdm(range(num_steps), desc="    Diffusion Steps", leave=False)
        for step in step_pbar:
            # 2. 预测去噪后的原始图像 (x0)
            # 通过当前的采样状态 sample 预测出对应的干净潜变量 pred_original_sample
            pred_original_sample = self.sampler_one_step(timestep, sample, guidance=4.0)
            
            # 3. 郎之万动力学优化
            # 调用 LGVD 模块，对预测出的 x0 进行优化。
            # 使其既接近模型预测的结果，又能够通过 operator 满足测量值 measurement。
            # (1 - alpha_t)^0.5 是当前的噪声水平，作为优化的约束强度之一
            pred_x0 = self.lgvd.sample(pred_original_sample, operator, measurement, (1 - self.scheduler.alphas_cumprod[timestep]) ** 0.5, step / num_steps, verbose=True)
            
            # 4. 时间步递减
            timestep = timestep - annel_interval
            
            # 5. 混合与重新加噪
            # 将优化后的结果与原始图像的潜变量 (latent_gt) 按比例 w 进行混合，以增强保真度。
            # 然后调用 get_start 重新加上噪声，得到下一个时间步的采样状态 sample。
            sample = self.get_start((1 - w) * pred_x0 + w * latent_gt, timestep)
            
        return pred_original_sample
        
    
    def sampler(self, timestep, latents, steps, guidance=GUIDANCE_SCALE):
        """标准的扩散采样循环"""
        interval = timestep // steps

        for _ in range(steps):
            """ 使用 CFG 预测噪声 """
            latent_model_input = torch.cat([latents] * 2)
            with torch.no_grad():
                noise_pred = self.model.unet(latent_model_input, timestep, encoder_hidden_states=self.context)["sample"]
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + 2.5 * (noise_pred_text - noise_pred_uncond)

            """ 计算上一步的样本 x_t -> x_t-1 """
            prev_timestep = timestep - interval
            alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
            alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
            beta_prod_t = 1 - alpha_prod_t
            pred_original_sample = (latents - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)
            pred_sample_direction = (1 - alpha_prod_t_prev) ** (0.5) * noise_pred
            prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

            """ 更新状态 """
            latents = prev_sample
            timestep = prev_timestep

        return pred_original_sample
    
    def sampler_one_step(self, timestep, latents, guidance=GUIDANCE_SCALE):
        """
        单步采样：从当前的噪声状态 x_t 预测最清晰的原始图像估计值 x_0。
        这一步是扩散模型的核心推理过程。
        """
        # 1. 准备 CFG（分类器自由引导）的输入
        # 将潜变量复制一份，以便同时计算“无条件”和“有条件”的噪声预测
        latent_model_input = torch.cat([latents] * 2)
        
        # 2. 调用 U-Net 预测噪声
        # 根据当前的潜变量、时间步和提示词上下文（context）预测图中包含的噪声
        noise_pred = self.model.unet(latent_model_input, timestep, encoder_hidden_states=self.context)["sample"]
        
        # 3. 执行 CFG 混合
        # 将预测出的噪声分为两部分：无提示词指导的噪声和有提示词指导的噪声
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        # 按照公式混合，通过 guidance 放大提示词的影响力，让结果更符合文本描述
        noise_pred = noise_pred_uncond + guidance * (noise_pred_text - noise_pred_uncond)

        # 4. 计算推断出的原始图像 x_0
        # 获取当前时间步的 alpha 累积乘积（代表图像中保留原图信息的比例）
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        # 计算 beta 累积乘积（代表图像中噪声的比例）
        beta_prod_t = 1 - alpha_prod_t
        
        # 应用扩散模型的基础公式：x_0 = (x_t - sqrt(1 - alpha_t) * epsilon) / sqrt(alpha_t)
        # 这一步尝试从乱码中“猜出”最底层的清晰图像
        pred_original_sample = (latents - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)

        return pred_original_sample
    
    def sample_in_batch(self, x_start, operator, y, starting_timestep=999):
        """批量执行反向步"""
        samples = self.prev_step(starting_timestep, x_start, operator, y)
        return samples

    def next_step(self, model_output, timestep, sample):
        """DDIM 前向步骤（反演时使用）"""
        timestep, next_timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample


    def get_added_cond_kwargs(self, use_cfg=False):
        """获取额外的条件参数（针对某些特定模型，如 SDXL）"""
        (   prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = (
            self.prompt_embeds,
            self.negative_prompt_embeds,
            self.pooled_prompt_embeds,
            self.negative_pooled_prompt_embeds,
        )
        add_text_embeds = pooled_prompt_embeds
        text_encoder_projection_dim = self.model.text_encoder_2.config.projection_dim
        add_time_ids = self.model._get_add_time_ids(
            (1024, 1024),
            (0, 0),
            (1024, 1024),
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        negative_add_time_ids = add_time_ids

        if use_cfg:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device)

        return {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
    
    def get_noise_pred_cfg(self, latents, t, guidance = GUIDANCE_SCALE):
        """带 CFG 的噪声预测"""
        latents_input = torch.cat([latents] * 2)
        with torch.no_grad():
            noise_pred = self.model.unet(latents_input, t, encoder_hidden_states=self.context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance * (noise_prediction_text - noise_pred_uncond)
        return noise_pred
    
    def get_noise_pred_single(self, latents, t, context):
        """不带 CFG 的单次噪声预测"""
        noise_pred = self.model.unet(latents, t, encoder_hidden_states=context)["sample"]
        return noise_pred
    
    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        """潜变量解码为图像"""
        latents = 1 / 0.18215 * latents.detach()
        image = self.model.vae.decode(latents)['sample']
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        return image

    @torch.no_grad()
    def image2latent(self, image):
        """
        将像素空间的图像编码为潜空间（Latent Space）的潜变量。
        这是扩散模型处理图像的第一步，将 512x512 的 RGB 图像压缩为 64x64 的特征图。
        """
        with torch.no_grad():
            # 1. 格式转换：如果是 PIL Image 对象，先转换为 numpy 数组
            if type(image) is Image:
                image = np.array(image)
            
            # 2. 如果输入已经是 4 维张量 [B, C, H, W]，则直接使用
            if type(image) is torch.Tensor and image.dim() == 4:
                latents = image
            else:
                # 3. 数据预处理：
                # 将像素值从 [0, 255] 归一化到 [-1, 1]（SD 模型要求的输入范围）
                image = torch.from_numpy(image).float() / 127.5 - 1
                # 调整维度顺序：从 [H, W, C] 变为 [C, H, W]，并增加 Batch 维度变为 [1, C, H, W]
                image = image.permute(2, 0, 1).unsqueeze(0).to(device)
                
                # 4. VAE 编码：
                # 使用预训练的 VAE 编码器将图像压缩到潜空间，取分布的均值作为结果
                latents = self.model.vae.encode(image)['latent_dist'].mean
                
                # 5. 缩放处理：
                # 乘以缩放系数 0.18215。这是 Stable Diffusion 官方为了让潜变量的方差接近 1 而使用的标准系数。
                latents = latents * 0.18215
        return latents

    @torch.no_grad()
    def init_prompt(self, prompt: str):
        # print(f"prompt: {prompt}")
        """
        初始化提示词编码。将文本提示词转换为模型可理解的向量嵌入（Embeddings）。
        此过程会同时生成“有条件”和“无条件（空文本）”的向量，用于后续的分类器自由引导（CFG）。
        """
        # 1. 准备无条件（Negative/Unconditional）输入：
        # 创建一个与 prompt 长度相同的空字符串列表，模拟“不提供任何描述”的情况
        uncond_input = self.model.tokenizer(
            [""]  * len(prompt),  # 结果就是 ["", ""]
            padding="max_length", # 填充到最大长度（通常是 77）
            max_length=self.model.tokenizer.model_max_length,
            truncation=True, # 超出部分截断
            return_tensors="pt" # 返回 PyTorch 张量
        )
        # uncond_input 是一个字典对象，包含 'input_ids' 和 'attention_mask'
        # 它本身没有 .shape 属性，需要访问其中的张量
        # print(f"uncond_input input_ids shape: {uncond_input.input_ids.shape}")
        # print(f"uncond_input attention_mask shape: {uncond_input.attention_mask.shape}")
        # 将空文本的 Token ID 传给文本编码器，获取“无条件嵌入”
        uncond_embeddings = self.model.text_encoder(uncond_input.input_ids.to(self.model.device))[0]
        # print(f"uncond_embeddings: {uncond_embeddings.shape}")

        # 2. 准备有条件（Conditional/Text）输入：
        # 将实际的提示词（如 "a cake on the desk"）进行分词
        text_input = self.model.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        # print(f"text_input input_ids shape: {text_input.input_ids.shape}")
        # print(f"text_input attention_mask shape: {text_input.attention_mask.shape}")
        # 将提示词的 Token ID 传给文本编码器，获取“有条件嵌入”
        text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.model.device))[0]
        # print(f"text_embeddings: {text_embeddings.shape}")
        # 3. 拼接上下文：
        # 将无条件嵌入和有条件嵌入在第 0 维（Batch 维度）拼接起来。
        # 拼接后的 self.context 形状通常为 [2*Batch, 77, 768]，供模型同时推理两者的结果
        self.context = torch.cat([uncond_embeddings, text_embeddings])
        # print(f"self.context: {self.context.shape}")
        self.prompt = prompt

    @property
    def scheduler(self):
        return self.model.scheduler

    def invert(self, latent, steps=50):
        """DDIM 反演：将潜变量映射回初始噪声轨迹"""
        ddim_latents = self.ddim_inversion(latent, steps)
        return ddim_latents[-1]

    def ddim_inversion(self, latent, steps):
        ddim_latents = self.ddim_loop(latent, steps)
        return ddim_latents
    
    def ddim_loop(self, latent, steps):
        """DDIM 反向迭代循环"""
        _, cond_embeddings = self.context.chunk(2)
        cond_embeddings = cond_embeddings[0].unsqueeze(0) 
        all_latent = [latent]
        latent = latent.clone().detach()
        for i in range(steps):
            # 获取当前步的时间戳（反向遍历）
            t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
            with torch.no_grad():
                noise_pred = self.get_noise_pred_single(latent, self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1], cond_embeddings.repeat(2,1,1))
            latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)
        return all_latent


def load_benchmark(path_to_prompts,
                   path_to_images=None):
    """
    加载测试集。
    path_to_prompts: CSV 文件路径，包含提示词等信息
    path_to_images: 图像目录路径（如果为编辑任务）
    """
    files = pd.read_csv(path_to_prompts)
    if path_to_images is None: # 这个没用了
        # 纯生成任务
        print(f'Generation benchmark: Loading from {path_to_prompts}')
        prompts = list(files['caption'])
        names = list(files['file_name'])
        return prompts, names
    else:
        # 图像编辑任务
        print(f'Editing benchmark: Loading prompts, images from {path_to_prompts}, {path_to_images}')
        files = files.reset_index()
        benchmark = []
        for _, row in files.iterrows():
            name = row['file_name']
            img_path = f'{path_to_images}/{name}'
            orig_prompt = row['old_caption']
            edited_prompt = row['edited_caption']
            blended_words = row['blended_words']
            benchmark.append((img_path,
                              {'before': orig_prompt,
                               'after': edited_prompt},
                              blended_words
                              )
                             )
        return benchmark


def find_difference2(word1, word2):
    """找出两个句子之间的差异词（在新句子中但不在旧句子中）"""
    splitted_w1 = word1.split(' ')
    splitted_w2 = word2.split(' ')
    out = []
    for i in splitted_w2:
        if i not in splitted_w1:
            out.append(i)
    return out


if __name__ == "__main__":
    # region 0. 解析命令行参数，这个是我加的
    parser = argparse.ArgumentParser(description="PostEdit 图像编辑脚本")
    parser.add_argument('--path_to_prompts', type=str, default='benchmarks/instructions/editing_pie_bench_700.csv', help='测试集提示词 CSV 文件路径')
    parser.add_argument('--path_to_images', type=str, default='benchmarks/images/pie_bench_700_images', help='测试集图像目录路径')
    parser.add_argument('--path_to_recon', type=str, default='results/recon/', help='重构结果保存路径')
    parser.add_argument('--path_to_editted', type=str, default='results/editted/', help='编辑结果保存路径')
    parser.add_argument('--single_image', action='store_true', help='是否仅运行单张图像测试（示例功能，可根据需要扩展逻辑）')
    args = parser.parse_args()
    # endregion

    # region 1. 基础模型加载
    """ 初始化计算设备并加载预训练的 LCM 扩散模型及其文本分词器，为后续的图像生成与编辑提供核心引擎 """
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    # 使用 LCM 模型，支持极速生成（少量步数即可获得高质量结果）
    # ldm_stable 是来自 diffusers 库的 StableDiffusionPipeline 实例。具体模型是 LCM (Latent Consistency Model) 的一个变体（LCM_Dreamshaper_v7）。
    ldm_stable = StableDiffusionPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7").to(device)
    # tokenizer是 CLIP 模型的文本分词器（Tokenizer）
    tokenizer = ldm_stable.tokenizer
    # endregion

    # region 2. 加载配置文件
    """设置要做一个随机遮盖 50% 区域的图像补全任务，并且在编辑时使用郎之万优化算法运行 200 步来保证画面的质量和一致性。"""
    with open('./config/task/inpainting_rand.yaml', 'r') as f:
        config_lgvd = yaml.safe_load(f)
    lgvd_config = config_lgvd.get('lgvd_config') # num_steps: 200 # 优化迭代的步数。
                                                 # lr: 1e-5 # 学习率。
                                                 # tau: 0.01 # 噪声项系数。
                                                 # lr_min_ratio: 0.01 # 最小学习率与最大学习率之比。
    # 初始化 NullInversion 核心组件
    null_inversion = NullInversion(ldm_stable, lgvd_config)
    # endregion

    # region 3. 加载测试数据集 (PIE-Bench)
    path_to_prompts = args.path_to_prompts
    path_to_images = args.path_to_images
    editing_benchmark = load_benchmark(path_to_prompts, path_to_images)
    # endregion

    # region 4. 创建结果保存目录
    path_to_recon = args.path_to_recon
    path_to_editted = args.path_to_editted
    os.makedirs(path_to_editted, exist_ok=True) # 编辑后的图像
    os.makedirs(path_to_recon, exist_ok=True)   # 重构出的原图（用于比对）
    # endregion

    # 遍历数据集进行实验
    print("开始遍历数据集进行实验:")
    pbar = tqdm(editing_benchmark)
    for index, (image_path, prompts_dict, blended_words) in enumerate(pbar):
        image_name = os.path.basename(image_path)
        print(f"处理图像: {image_name}")
        pbar.set_description(f"Processing {image_name}")
     
        # region 5. 图像预处理与编码
        print(f"\n[{index}] 图像预处理与编码: {image_path}")
        offsets=(0,0,0,0)
        images = load_512(image_path, *offsets) # 这边因为offset是0，所以没有切割，直接是缩放到了512*512
        # print(f"images shape: {images.shape}")
        # 将原始图像通过 VAE 编码到潜空间
        latent = null_inversion.image2latent(images)
        # print(f"latent shape: {latent.shape}")
        full_samples= []
        full_samples.append(images) # 存入原始图像作为对比

        full_samples_rec= []
        full_samples_rec.append(images) 
        # endregion

        # region 6. 提示词处理
        # 获取编辑前后的提示词
        prompts = [prompts_dict['before'], prompts_dict['after']]
        print(f"[{index}] 提示词: Before='{prompts[0]}', After='{prompts[1]}'")
        null_inversion.init_prompt(prompts)
        # endregion

        # region 7. 获取测量算子（如退化操作）
        task_operator = config_lgvd.get('operator')  
        operator = get_operator(**task_operator)
        # 对原始潜变量执行测量操作
        print(f"[{index}] 对原始潜变量执行测量操作: {task_operator.get('name', 'unknown')}")
        y = operator.measure(latent)
        latent_gt = latent
        # print(f"y: {y.shape}")
        # endregion

        """ 8. 执行编辑与重构循环 """
        starting_timestep = 501 # 扩散步的起点
        num_runs = 5            # 对同一张图进行多次生成以观察稳定性

        print(f"[{index}] 执行编辑与重构循环: {num_runs}次")
        run_pbar = tqdm(range(num_runs), desc=f"  Runs", leave=False)
        for r in run_pbar:
            # 获取带噪声的潜变量起点
            latent_t = null_inversion.get_start(latent, starting_timestep=starting_timestep, double=True)
            # 执行带有郎之万优化和注意力控制的采样
            samples = null_inversion.sample_in_batch(latent_t, operator, y, starting_timestep=starting_timestep)
            print(f"samples shape: {samples.shape}")
            
            # 保存重构的原图（验证反演准确度）
            image = null_inversion.latent2image(samples[0].unsqueeze(0))
            full_samples_rec.append(image)
            Image.fromarray(image).save(os.path.join(path_to_recon, str(index) + '_' + str(r) + '.png'))
            
            # 保存编辑后的图像
            image = null_inversion.latent2image(samples[1].unsqueeze(0))
            full_samples.append(image)
            Image.fromarray(image).save(os.path.join(path_to_editted, str(index) + '_' + str(r) + '.png'))

        # 将多次运行的结果拼接并保存，方便对比
        print(f"[{index}] 将多次运行的结果拼接并保存...")
        full_samples = np.concatenate(full_samples, 1)
        Image.fromarray(full_samples).save(os.path.join(path_to_editted, str(index) + '.png'))

        full_samples_rec = np.concatenate(full_samples_rec, 1)
        Image.fromarray(full_samples_rec).save(os.path.join(path_to_recon, str(index) + '.png'))

        
