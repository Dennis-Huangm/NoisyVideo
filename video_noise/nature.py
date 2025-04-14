# Denis
# -*- coding: utf-8 -*-
from torchvision import transforms
from .noise_applier import NoiseRegistry
from .utils import *
import torch
import numpy as np
import skimage.color as sk_color

@NoiseRegistry.register("bright_transform") # 亮度变换
# def add_brightness_tf(video: torch.Tensor, ratio):
    # noise_indice = sample_noise_frame(video, ratio)
    # transform = transforms.Compose([
    #     lambda x: global_brightness_shift(x, (-0.3, 0.3)),
    #     lambda x: gamma_correction(x, (0.6, 1.4))
    # ])
    # for i in noise_indice:
    #     video[i] = transform(video[i])
    # return video
def add_brightness_hsv(video: torch.Tensor, ratio, severity=5):
    """
    使用HSV颜色空间调整视频亮度，类似第二个brightness函数，
    但保留第一个函数的视频处理框架
    
    参数:
        video: 输入视频张量 [T, C, H, W]
        ratio: 要处理的帧的比例
        severity: 亮度变化的严重程度(1-5)
    """
 
    
    noise_indices = sample_noise_frame(video, ratio)
    
    # 亮度调整值，从第二个函数中获取
    c_values = [0.1, 0.2, 0.3, 0.4, 0.5]
    c = c_values[min(severity - 1, 4)]  # 确保severity在1-5范围内
    
    for i in noise_indices:
        # 将PyTorch张量转换为Numpy数组
        frame_np = video[i].permute(1, 2, 0).cpu().numpy() / 255.0  # [H, W, C]
        
        if frame_np.shape[2] == 3:  # RGB图像
            # 转换为HSV颜色空间
            frame_hsv = sk_color.rgb2hsv(frame_np)
            # 调整亮度通道(V)
            frame_hsv[:, :, 2] = np.clip(frame_hsv[:, :, 2] + c, 0, 1)
            # 转回RGB
            frame_np = sk_color.hsv2rgb(frame_hsv)
        else:  # 灰度图像或其他通道数
            frame_np = np.clip(frame_np + c, 0, 1)
        
        # 转回PyTorch张量并更新原视频
        frame_np = np.clip(frame_np * 255.0, 0, 255.0)
        if len(frame_np.shape) == 3:  # RGB
            frame_tensor = torch.from_numpy(frame_np).permute(2, 0, 1).to(video.device)
        else:  # 灰度
            frame_tensor = torch.from_numpy(frame_np).unsqueeze(0).to(video.device)
        
        video[i] = frame_tensor
    
    return video.to(torch.uint8)


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
    return video.to(torch.uint8)




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
    return video.to(torch.uint8)


@NoiseRegistry.register("flicker") # 闪烁效应
def random_flicker(video: torch.Tensor, ratio=0.2):
    device = video.device
    for i in range(len(video)):
        # 生成每帧亮度系数
        alphas = 1 + torch.randn(1, device=device) * ratio
        video[i] = torch.clamp(video[i] * alphas, 0.0, 255.0)
    return video.to(torch.uint8)


@NoiseRegistry.register("overexposure") # 过曝光
def add_over_exposure(video: torch.Tensor, ratio):
    noise_indice = sample_noise_frame(video, ratio)
    transform = transforms.Compose([
        lambda x: global_brightness_shift(x, (0.1, 0.3)),
        lambda x: gamma_correction(x, (1.1, 1.4))
    ])
    for i in noise_indice:
        video[i] = transform(video[i])
    return video.to(torch.uint8)


@NoiseRegistry.register("underexposure") # 欠曝光
def add_under_exposure(video: torch.Tensor, ratio):
    noise_indice = sample_noise_frame(video, ratio)
    transform = transforms.Compose([
        lambda x: global_brightness_shift(x, (-0.3, -0.1)),
        lambda x: gamma_correction(x, (0.6, 0.9))
    ])
    for i in noise_indice:
        video[i] = transform(video[i])
    return video.to(torch.uint8)








