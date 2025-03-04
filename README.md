# PostEdit

> PostEdit is both inversion- and training-free, necessitating approximately 1.5 seconds and 18 GB of GPU memory to generate high-quality results. PostEdit is accepted as a poster in International Conference on Learning Representations (ICLR) 2025!


![teaser](docs/intro_Page1.png)
[Paper](https://arxiv.org/pdf/2410.04844)


## Setup

This code was tested with Python 3.9, [Pytorch](https://pytorch.org/) 2.4.0 using pre-trained models through [huggingface / diffusers](https://github.com/huggingface/diffusers#readme).
Specifically, we implemented our method over  [Latent Diffusion] LCM.
Additional required packages are listed in the requirements file.
The code was tested on a single NVIDIA A100 GPU.

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
  journal      = {CoRR},
  volume       = {abs/2410.04844},
  year         = {2024},
  url          = {https://doi.org/10.48550/arXiv.2410.04844},
  doi          = {10.48550/ARXIV.2410.04844},
  eprinttype    = {arXiv},
  eprint       = {2410.04844},
  timestamp    = {Tue, 12 Nov 2024 18:39:25 +0100},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2410-04844.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
