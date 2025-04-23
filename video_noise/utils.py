# Denis
# -*- coding: utf-8 -*-
import torch
import math
import random
from PIL import Image
import torch.nn.functional as F
from noise import pnoise2
import numpy as np
from torchvision.transforms import ToPILImage


def gaussian_kernel(kernel_size, sigma, device):
    # 生成坐标网格
    ax = torch.linspace(-(kernel_size-1)//2, (kernel_size-1)//2, steps=kernel_size)
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    
    # 计算高斯分布
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()  # normalization
    
    return kernel.to(device)


def motion_kernel(kernel_size, angle, device):
    kernel = torch.zeros((kernel_size, kernel_size), device=device)
    center = (kernel_size-1)//2
    
    # 角度转弧度
    theta = math.radians(angle)
    dx = math.cos(theta)
    dy = math.sin(theta)
    
    # 绘制运动路径
    for i in range(kernel_size):
        x = int(center + (i - center)*dx)
        y = int(center + (i - center)*dy)
        if 0 <= x < kernel_size and 0 <= y < kernel_size:
            kernel[x, y] = 1
    
    # 归一化
    if kernel.sum() == 0:
        kernel[center, center] = 1  # 避免全零核
    return kernel / kernel.sum()


def sample_noise_frame(video: torch.Tensor, ratio: float):
    n_frames = video.shape[0]
    n_noisy_frames = int(n_frames * ratio)
    noise_indice = random.sample(range(n_frames), n_noisy_frames)
    return noise_indice 


def global_brightness_shift(image, delta_range=(-0.3, 0.3)):
    if isinstance(delta_range, (tuple, list)):
        delta = torch.empty(1, device=image.device).uniform_(*delta_range)
    else:
        delta = delta_range
    
    # 线性变换：new = image * gamma + delta
    adjusted = torch.clamp(image + delta, 0.0, 255.0)
    return adjusted


def gamma_correction(image, gamma_range=(0.5, 1.5)):
    if isinstance(gamma_range, (tuple, list)):
        gamma = torch.empty(1, device=image.device).uniform_(*gamma_range)
    else:
        gamma = gamma_range
    
    # 伽马公式：new = image ** gamma
    adjusted = torch.clamp(image ** gamma, 0.0, 255.0)
    return adjusted


def save_image(image, name='./sample.png'):
    to_pil = ToPILImage()
    pil_image = to_pil(image)
    # frame_np = image.permute(1, 2, 0).cpu().numpy()
    # img_pil = Image.fromarray(frame_np)  # RGB
    pil_image.save(name) 


# 修正颜色空间转换维度处理
def rgb_to_ycbcr(x: torch.Tensor) -> torch.Tensor:
    """ 正确处理形状 [T, C, H, W] 的RGB转YCbCr """
    matrix = torch.tensor([
        [0.299, 0.587, 0.114],
        [-0.168736, -0.331264, 0.5],
        [0.5, -0.418688, -0.081312]
    ], device=x.device)
    ycbcr = torch.einsum('...c,cd->...d', x.float().permute(0,2,3,1), matrix)
    ycbcr[..., 1:] += 128
    return ycbcr.permute(0,3,1,2)  # 恢复 [T, C, H, W]


def ycbcr_to_rgb(x: torch.Tensor) -> torch.Tensor:
    """ 正确处理形状 [T, C, H, W] 的YCbCr转RGB """
    matrix = torch.tensor([
        [1.0, 0.0, 1.402],
        [1.0, -0.344136, -0.714136],
        [1.0, 1.772, 0.0]
    ], device=x.device)
    x = x.permute(0,2,3,1)  # [T, H, W, C]
    x[..., 1:] -= 128
    rgb = torch.einsum('...c,cd->...d', x, matrix)
    return rgb.permute(0,3,1,2).clamp(0,255)  # 恢复 [T, C, H, W]


# 模拟块效应（分块量化）
def add_block_artifacts(x, block_size, intensity):
    _, _, H, W = x.shape
    x_blocks = F.unfold(x, block_size, stride=block_size)
    noise = torch.randn_like(x_blocks) * intensity
    return F.fold(x_blocks + noise, (H, W), block_size, stride=block_size)


def generate_perlin_noise(height, width, scale=100.0, octaves=6):
    """生成Perlin噪声图（0-255范围）"""
    x = np.linspace(0, width/scale, width)
    y = np.linspace(0, height/scale, height)
    xv, yv = np.meshgrid(x, y)
    
    # 向量化计算Perlin噪声
    noise = np.vectorize(pnoise2)(xv, yv, octaves=octaves, repeatx=1024, repeaty=1024)
    noise = (noise - np.min(noise)) / (np.max(noise) - np.min(noise))  
    return (noise * 255).astype(np.uint8)  


def preprocess(tensor, target_size=640):
    tensor = tensor.unsqueeze(0).float() / 255.0
    _, _, h, w = tensor.shape
    scale = min(target_size / h, target_size / w)  # 计算缩放比例
    new_h, new_w = int(h * scale), int(w * scale)
    
    # 调整尺寸
    resized = F.interpolate(tensor, size=(new_h, new_w), mode="bilinear")
    
    # 计算填充量
    pad_h = target_size - new_h
    pad_w = target_size - new_w
    top = pad_h // 2
    left = pad_w // 2
    
    # 填充并返回参数
    padded = F.pad(resized, [left, pad_w - left, top, pad_h - top], value=0.5)
    return padded, scale, (top, left)
