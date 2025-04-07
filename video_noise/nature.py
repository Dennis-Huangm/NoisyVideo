# Denis
# -*- coding: utf-8 -*-
from torchvision import transforms
from .noise_applier import NoiseRegistry
from .utils import *


@NoiseRegistry.register("bright_transform") # 亮度变换
def add_brightness_tf(video: torch.Tensor, ratio):
    noise_indice = sample_noise_frame(video, ratio)
    transform = transforms.Compose([
        lambda x: global_brightness_shift(x, (-0.3, 0.3)),
        lambda x: gamma_correction(x, (0.6, 1.4))
    ])
    for i in noise_indice:
        video[i] = transform(video[i])
    return video


@NoiseRegistry.register("contrast") # 对比度变换
def add_contrast_noise(
    video: torch.Tensor,
    ratio: float, 
    contrast_range: tuple = (-5.0, 5.0),  
) -> torch.Tensor:
    noise_indice = sample_noise_frame(video, ratio)
    min_val, max_val = contrast_range

    # 根据强度插值调整对比度范围
    for i in noise_indice:
        mean = video[i].float().mean(dim=(-1, -2), keepdim=True)
        contrast_factor = torch.empty(1, device=video.device).uniform_(min_val, max_val)
        if contrast_factor < 0:
            contrast_factor = -1 / contrast_factor
        video[i] = torch.clamp(contrast_factor * (video[i] - mean) + mean, 0.0, 255.0)
    return video


@NoiseRegistry.register("color_shift") # 色彩偏移
def add_color_shift_noise(
    video: torch.Tensor,
    ratio: float,
    shift: float = 1,
    per_channel: bool = True
) -> torch.Tensor:
    noise_indice = sample_noise_frame(video, ratio)

    for i in noise_indice:
        # 生成随机颜色增益系数
        if per_channel:
            color_factors = torch.empty(3, 1, 1, device=video.device).uniform_(1 - shift, 1 + shift)
        else:
            color_factors = torch.empty(1, device=video.device).uniform_(1 - shift, 1 + shift)
        video[i] = torch.clamp(video[i] * color_factors, 0.0, 255.0)
    return video


@NoiseRegistry.register("flicker") # 闪烁效应
def random_flicker(video: torch.Tensor, ratio=0.2):
    device = video.device
    for i in range(len(video)):
        # 生成每帧亮度系数
        alphas = 1 + torch.randn(1, device=device) * ratio
        video[i] = torch.clamp(video[i] * alphas, 0.0, 255.0)
    return video


@NoiseRegistry.register("overexposure") # 过曝光
def add_over_exposure(video: torch.Tensor, ratio):
    noise_indice = sample_noise_frame(video, ratio)
    transform = transforms.Compose([
        lambda x: global_brightness_shift(x, (0.1, 0.3)),
        lambda x: gamma_correction(x, (1.1, 1.4))
    ])
    for i in noise_indice:
        video[i] = transform(video[i])
    return video


@NoiseRegistry.register("underexposure") # 欠曝光
def add_under_exposure(video: torch.Tensor, ratio):
    noise_indice = sample_noise_frame(video, ratio)
    transform = transforms.Compose([
        lambda x: global_brightness_shift(x, (-0.3, -0.1)),
        lambda x: gamma_correction(x, (0.6, 0.9))
    ])
    for i in noise_indice:
        video[i] = transform(video[i])
    return video








