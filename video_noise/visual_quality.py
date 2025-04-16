# Denis
# -*- coding: utf-8 -*-
from .noise_applier import NoiseRegistry
from .utils import *
from io import BytesIO


@NoiseRegistry.register("gaussian") # 高斯噪声
def add_gaussian_noise(video: torch.Tensor, ratio, std=100) -> torch.Tensor:
    video = video.to('cuda')
    noise_indice = sample_noise_frame(video, ratio)
    for i in noise_indice:
        noise = torch.randn_like(video[i], dtype=torch.float) * std
        video[i] = torch.clamp(video[i].float() + noise, 0.0, 255.0)
    return video.to(torch.uint8) 


@NoiseRegistry.register("impulse") # 椒盐噪声
def add_impulse_noise(video: torch.Tensor, ratio: float, prob: float = 0.7) -> torch.Tensor:
    noise_indices = sample_noise_frame(video, ratio)  # 假设该函数返回需要加噪的帧索引
    
    for i in noise_indices:
        _, H, W = video[i].shape
        
        rand_mask = torch.rand(H, W, device=video.device)
        black_mask = rand_mask < (prob / 2)        
        white_mask = (prob / 2 <= rand_mask) & (rand_mask < prob)

        video[i, :, black_mask] = 0.0    # 纯黑（所有通道置0）
        video[i, :, white_mask] = 255.0    # 纯白（所有通道置1）

    return video.to(torch.uint8) 

@NoiseRegistry.register("speckle") # 斑点噪声
def add_speckle_noise(
    video: torch.Tensor, 
    ratio: float, 
    intensity: float = 0.7,
    grain_size: int = 2
) -> torch.Tensor:
    """
    向视频中添加斑点噪声（乘性噪声）
    
    参数:
        video: 输入视频张量 [T, C, H, W]
        ratio: 要处理的帧的比例
        intensity: 噪声强度，控制斑点的亮度变化程度
        grain_size: 斑点的大小（像素）
    
    返回:
        添加斑点噪声后的视频
    """
    noise_indices = sample_noise_frame(video, ratio)
    
    for i in noise_indices:
        _, H, W = video[i].shape
        
        # 创建噪声
        if grain_size == 1:
            # 单像素斑点
            noise = torch.randn(1, H, W, device=video.device) * intensity
        else:
            # 创建较大斑点（先生成小噪声图，然后上采样）
            small_h, small_w = H // grain_size, W // grain_size
            small_noise = torch.randn(1, small_h, small_w, device=video.device) * intensity
            noise = F.interpolate(
                small_noise.unsqueeze(0), 
                size=(H, W), 
                mode='nearest'
            ).squeeze(0)
        
        # 斑点噪声是乘性噪声: I_noisy = I * (1 + noise)
        video[i] = torch.clamp(video[i] * (1.0 + noise), 0.0, 255.0)
    
    return video.to(torch.uint8) 


@NoiseRegistry.register("poisson") # 泊松噪声
def add_poisson_noise(video: torch.Tensor, ratio, gain=0.01) -> torch.Tensor:
    video = video.to('cuda')
    noise_indice = sample_noise_frame(video, ratio)
    for i in noise_indice:
        scaled = video[i] * gain
        noisy_scaled = torch.poisson(scaled)
        video[i] = noisy_scaled / gain
    return video.to(torch.uint8)

