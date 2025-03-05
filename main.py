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


LOW_RESOURCE = False 
NUM_DDIM_STEPS = 5
GUIDANCE_SCALE = 8.0
GUIDANCE_SCALE_MAX = 8.0
GUIDANCE_SCALE_MIN = 8.0
MAX_NUM_WORDS = 77


class LocalBlend:
    
    def get_mask(self, maps, alpha, use_pool):
        k = 1
        maps = (maps * alpha).sum(-1).mean(1)
        if use_pool:
            maps = nnf.max_pool2d(maps, (k * 2 + 1, k * 2 +1), (1, 1), padding=(k, k))
        mask = nnf.interpolate(maps, size=(64, 64))
        mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
        mask = mask.gt(self.th[1-int(use_pool)])
        mask = mask[:1] + mask
        return mask
    
    def __call__(self, x_t, attention_store):
        self.counter += 1
        if self.counter > self.start_blend:
           
            maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
            maps = [item.reshape(self.alpha_layers.shape[0], -1, 1, 16, 16, MAX_NUM_WORDS) for item in maps]
            maps = torch.cat(maps, dim=1)
            mask = self.get_mask(maps, self.alpha_layers, True)
            if self.substruct_layers is not None:
                maps_sub = ~self.get_mask(maps, self.substruct_layers, False)
                mask = mask * maps_sub
            mask = mask.float()
            x_t = x_t[:1] + mask * (x_t - x_t[:1])
        return x_t
       
    def __init__(self, prompts: List[str], words: [List[List[str]]], substruct_words=None, start_blend=0.2, th=(.3, .3)):
        alpha_layers = torch.zeros(len(prompts),  1, 1, 1, 1, MAX_NUM_WORDS)
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if type(words_) is str:
                words_ = [words_]
            for word in words_:
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
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0
    
    @abc.abstractmethod
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0


class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
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
    
    def step_callback(self, x_t):
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store)
        return x_t
        
    def replace_self_attention(self, attn_base, att_replace, place_in_unet):
        if att_replace.shape[2] <= 32 ** 2:
            attn_base = attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
            return attn_base
        else:
            return att_replace
    
    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, att_replace):
        raise NotImplementedError
    
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet)
        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
            h = attn.shape[0] // (self.batch_size)
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            attn_base, attn_repalce = attn[0], attn[1:]
            if is_cross:
                alpha_words = self.cross_replace_alpha[self.cur_step]
                attn_repalce_new = self.replace_cross_attention(attn_base, attn_repalce) * alpha_words + (1 - alpha_words) * attn_repalce
                attn[1:] = attn_repalce_new
            else:
                attn[1:] = self.replace_self_attention(attn_base, attn_repalce, place_in_unet)
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        return attn
    
    def __init__(self, prompts, num_steps: int,
                 cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
                 self_replace_steps: Union[float, Tuple[float, float]],
                 local_blend: Optional[LocalBlend]):
        super(AttentionControlEdit, self).__init__()
        self.batch_size = len(prompts)
        self.cross_replace_alpha = ptp_utils.get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps, tokenizer).to(device)
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        self.local_blend = local_blend


class AttentionReplace(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        return torch.einsum('hpw,bwn->bhpn', attn_base, self.mapper)
      
    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None):
        super(AttentionReplace, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.mapper = seq_aligner.get_replacement_mapper(prompts, tokenizer).to(device)
        

class AttentionRefine(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        attn_base_replace = attn_base[:, :, self.mapper].permute(2, 0, 1, 3)
        attn_replace = attn_base_replace * self.alphas + att_replace * (1 - self.alphas)
        # attn_replace = attn_replace / attn_replace.sum(-1, keepdims=True)
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None):
        super(AttentionRefine, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.mapper, alphas = seq_aligner.get_refinement_mapper(prompts, tokenizer)
        self.mapper, alphas = self.mapper.to(device), alphas.to(device)
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])


class AttentionReweight(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        if self.prev_controller is not None:
            attn_base = self.prev_controller.replace_cross_attention(attn_base, att_replace)
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
    if type(word_select) is int or type(word_select) is str:
        word_select = (word_select,)
    equalizer = torch.ones(1, 77)
    
    for word, val in zip(word_select, values):
        inds = ptp_utils.get_word_inds(text, word, tokenizer)
        equalizer[:, inds] = val
    return equalizer


def make_controller(prompts: List[str], is_replace_controller: bool, cross_replace_steps: Dict[str, float], self_replace_steps: float, blend_words=None, equilizer_params=None) -> AttentionControlEdit:
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
    def __init__(self, num_steps, lr, tau=0.01, lr_min_ratio=0.01):
        super().__init__()
        self.num_steps = num_steps
        self.lr = float(lr)
        self.tau = tau
        self.lr_min_ratio = lr_min_ratio

    def sample(self, x0hat, operator, measurement, sigma, ratio, verbose=False, scores=None, steps=None):
        num_steps = self.num_steps if steps is None else steps
        pbar = tqdm.trange(num_steps) if verbose else range(num_steps)
        lr = self.get_lr(ratio)
        x = x0hat.clone().detach().requires_grad_(True)
        optimizer = torch.optim.SGD([x], lr)
        for _ in pbar:
            optimizer.zero_grad()
            loss = operator.error(x, measurement).sum() / (2 * self.tau ** 2)
            loss += ((x - x0hat.detach()) ** 2).sum() / (2 * sigma ** 2)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                epsilon = torch.randn_like(x)
                x.data = x.data + np.sqrt(2 * lr) * epsilon

            if torch.isnan(x).any():
                return torch.zeros_like(x)

        return x.detach()

    def get_lr(self, ratio, p=1):
        multiplier = (1 ** (1 / p) + ratio * (self.lr_min_ratio ** (1 / p) - 1 ** (1 / p))) ** p
        return multiplier * self.lr


class NullInversion: 
    def __init__(self, model, lgvd_config):
        self.model = model
        self.tokenizer = self.model.tokenizer
        self.prompt = None
        self.context = None
        self.lgvd = LangevinDynamics(**lgvd_config)

    def get_start(self, ref, starting_timestep=999, double=False):
        if double: ref = torch.cat([ref] * 2) # [original, editted]
        noise = torch.randn(ref.shape).to(device)
        timestep = torch.tensor(starting_timestep)
        x_start = self.model.scheduler.add_noise(ref, noise, timestep)
        return x_start
    
    def prev_step(self, timestep, sample, operator, measurement, annel_interval=100, w = 0.25):
        num_steps = int((timestep - 1) / annel_interval)
        for step in range(num_steps):
            pred_original_sample = self.sampler_one_step(timestep, sample, guidance=4.0)
            pred_x0 = self.lgvd.sample(pred_original_sample, operator, measurement, (1 - self.scheduler.alphas_cumprod[timestep]) ** 0.5, step / num_steps)
            timestep = timestep - annel_interval
            sample = self.get_start((1 - w) * pred_x0 + w * latent_gt, timestep)
        return pred_original_sample
        
    
    def sampler(self, timestep, latents, steps, guidance=GUIDANCE_SCALE):
        interval = timestep // steps

        for _ in range(steps):
            """ Predict noise with CFG """
            latent_model_input = torch.cat([latents] * 2)
            with torch.no_grad():
                noise_pred = self.model.unet(latent_model_input, timestep, encoder_hidden_states=self.context)["sample"]
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + 2.5 * (noise_pred_text - noise_pred_uncond)

            """ Compute the previous noisy sample x_t -> x_t-1 """
            prev_timestep = timestep - interval
            alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
            alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
            beta_prod_t = 1 - alpha_prod_t
            pred_original_sample = (latents - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)
            pred_sample_direction = (1 - alpha_prod_t_prev) ** (0.5) * noise_pred
            prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

            """ Update x_t and t """
            latents = prev_sample
            timestep = prev_timestep

        return pred_original_sample
    
    def sampler_one_step(self, timestep, latents, guidance=GUIDANCE_SCALE):
        """ Predict noise with CFG """
        latent_model_input = torch.cat([latents] * 2)
        # with torch.no_grad(): 
        noise_pred = self.model.unet(latent_model_input, timestep, encoder_hidden_states=self.context)["sample"]
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance * (noise_pred_text - noise_pred_uncond)

        """ Compute the previous noisy sample x_t -> x_t-1 """
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (latents - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)

        return pred_original_sample
    
    def sample_in_batch(self, x_start, operator, y, starting_timestep=999):
        samples = self.prev_step(starting_timestep, x_start, operator, y)
        return samples

    def next_step(self, model_output, timestep, sample):
        timestep, next_timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample


    def get_added_cond_kwargs(self, use_cfg=False):
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
        latents_input = torch.cat([latents] * 2)
        with torch.no_grad():
            noise_pred = self.model.unet(latents_input, t, encoder_hidden_states=self.context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance * (noise_prediction_text - noise_pred_uncond)
        return noise_pred
    
    def get_noise_pred_single(self, latents, t, context):
        noise_pred = self.model.unet(latents, t, encoder_hidden_states=context)["sample"]
        return noise_pred
    
    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        latents = 1 / 0.18215 * latents.detach()
        image = self.model.vae.decode(latents)['sample']
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        return image

    @torch.no_grad()
    def image2latent(self, image):
        with torch.no_grad():
            if type(image) is Image:
                image = np.array(image)
            if type(image) is torch.Tensor and image.dim() == 4:
                latents = image
            else:
                image = torch.from_numpy(image).float() / 127.5 - 1
                image = image.permute(2, 0, 1).unsqueeze(0).to(device)
                latents = self.model.vae.encode(image)['latent_dist'].mean
                latents = latents * 0.18215
        return latents

    @torch.no_grad()
    def init_prompt(self, prompt: str):
        uncond_input = self.model.tokenizer(
            [""]  * len(prompt), padding="max_length", 
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        uncond_embeddings = self.model.text_encoder(uncond_input.input_ids.to(self.model.device))[0]
        text_input = self.model.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.model.device))[0]
        self.context = torch.cat([uncond_embeddings, text_embeddings])
        self.prompt = prompt

    @property
    def scheduler(self):
        return self.model.scheduler

    def invert(self, latent, steps=50):
        ddim_latents = self.ddim_inversion(latent, steps)
        return ddim_latents[-1]

    def ddim_inversion(self, latent, steps):
        ddim_latents = self.ddim_loop(latent, steps)
        return ddim_latents
    
    def ddim_loop(self, latent, steps):
        _, cond_embeddings = self.context.chunk(2)
        cond_embeddings = cond_embeddings[0].unsqueeze(0) 
        all_latent = [latent]
        latent = latent.clone().detach()
        for i in range(steps):
            t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
            with torch.no_grad():
                noise_pred = self.get_noise_pred_single(latent, self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1], cond_embeddings.repeat(2,1,1))
            latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)
        return all_latent


def load_benchmark(path_to_prompts,
                   path_to_images=None):
    files = pd.read_csv(path_to_prompts)
    if path_to_images is None:
        print(f'Generation benchmark: Loading from {path_to_prompts}')
        prompts = list(files['caption'])
        names = list(files['file_name'])
        return prompts, names
    else:
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
    splitted_w1 = word1.split(' ')
    splitted_w2 = word2.split(' ')
    out = []
    for i in splitted_w2:
        if i not in splitted_w1:
            out.append(i)
    return out


if __name__ == "__main__":
    """ Setting some base models """
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    ldm_stable = StableDiffusionPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7").to(device)
    tokenizer = ldm_stable.tokenizer

    """ Loading config from yaml files """
    with open('./config/task/inpainting_rand.yaml', 'r') as f:
        config_lgvd = yaml.safe_load(f)
    lgvd_config = config_lgvd.get('lgvd_config')
    null_inversion = NullInversion(ldm_stable, lgvd_config)

    """ Loading the whole benchmark dataset """
    path_to_prompts = 'benchmarks/instructions/editing_pie_bench_700.csv'
    path_to_images = 'benchmarks/images/pie_bench_700_images'
    editing_benchmark = load_benchmark(path_to_prompts, path_to_images)

    """ Making directory for saving results """
    os.makedirs('results/editted/', exist_ok=True)
    os.makedirs('results/recon/', exist_ok=True)

    for j, (image_path, prompts_dict, blended_words) in enumerate(tqdm(editing_benchmark)):
     
        """ Loading the input image"""
        offsets=(0,0,0,0)
        images = load_512(image_path, *offsets)
        latent = null_inversion.image2latent(images)
        full_samples= []
        full_samples.append(images) 

        full_samples_rec= []
        full_samples_rec.append(images) 

        """ Setting prompts """
        prompts = [prompts_dict['before'], prompts_dict['after']]
        null_inversion.init_prompt(prompts)
    
        """ Getting measurement """
        task_operator = config_lgvd.get('operator')  
        operator = get_operator(**task_operator)
        y = operator.measure(latent)
        latent_gt = latent

        """ Editting """
        starting_timestep = 501
        num_runs = 5

        for r in range(num_runs):
            latent_t = null_inversion.get_start(latent, starting_timestep=starting_timestep, double=True)
            samples = null_inversion.sample_in_batch(latent_t, operator, y, starting_timestep=starting_timestep)
            image = null_inversion.latent2image(samples[0].unsqueeze(0))
            full_samples_rec.append(image)
            Image.fromarray(image).save('results/recon/' + str(index) + '_' + str(r) + '.png')
            image = null_inversion.latent2image(samples[1].unsqueeze(0))
            full_samples.append(image)
            Image.fromarray(image).save('results/editted/' + str(index) + '_' + str(r) + '.png')

        full_samples = np.concatenate(full_samples, 1)
        Image.fromarray(full_samples).save('results/editted/' + str(index) + '.png')

        full_samples_rec = np.concatenate(full_samples_rec, 1)
        Image.fromarray(full_samples_rec).save('results/recon/' + str(index) + '.png')

        
