# PostEdit: Posterior Sampling for Efficient Zero-Shot Image Editing

> ⚡️ PostEdit is both inversion- and training-free, necessitating approximately 1.5 seconds and 18 GB of GPU memory to generate high-quality results.
> 
> 💥 PostEdit is accepted as a poster in International Conference on Learning Representations (ICLR) 2025!


![exp](exp.png)
![detail](detail.png)
[Paper](https://arxiv.org/pdf/2410.04844)


## Setup

This code was tested with Python 3.9, [Pytorch](https://pytorch.org/) 2.4.0 using pre-trained models through [huggingface / diffusers](https://github.com/huggingface/diffusers#readme).
Specifically, we implemented our method over [LCM](https://arxiv.org/pdf/2310.04378).
Additional required packages are listed in the requirements file.
The code was tested on a single NVIDIA A100 GPU.

## Preparation

### Dataset
Download [PIE-Bench](https://docs.google.com/forms/d/e/1FAIpQLSftGgDwLLMwrad9pX3Odbnd4UXGvcRuXDkRp6BT1nPk8fcH_g/viewform) dataset, and place it in your `PIE_Bench_PATH`.

### Installation

Download the code:
```
git clone https://github.com/TFNTF/PostEdit.git
```
Download pre-trained models:

```python
from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7")

```

## Quickstart
``` python
pip install -r requirements.txt
python main.py
(Optional) Save a specific image to "all_images" file for single image editing.
```


## Citation

``` bibtex
@article{DBLP:journals/corr/abs-2410-04844,
  author       = {Feng Tian and
                  Yixuan Li and
                  Yichao Yan and
                  Shanyan Guan and
                  Yanhao Ge and
                  Xiaokang Yang},
  title        = {PostEdit: Posterior Sampling for Efficient Zero-Shot Image Editing},
  journal      = {ICLR},
  year         = {2024},
}
```

## Acknowledgements

We thank vivo for granting us access to GPUs.

## Contact

If you have any questions, feel free to contact me through email (tf1021@sjtu.edu.cn). Enjoy!

