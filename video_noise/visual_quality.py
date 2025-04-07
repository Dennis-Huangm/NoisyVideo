# Denis
# -*- coding: utf-8 -*-
from .noise_applier import NoiseRegistry
from .utils import *


@NoiseRegistry.register("gaussian") # 高斯噪声
def add_gaussian_noise(video: torch.Tensor, ratio, std=100) -> torch.Tensor:
    video = video.to('cuda')
    noise_indice = sample_noise_frame(video, ratio)
    for i in noise_indice:
        noise = torch.randn_like(video[i], dtype=torch.float) * std
        noisy_frame = torch.clamp(video[i].float() + noise, 0.0, 255.0)
        video[i] = noisy_frame.to(video.dtype)
    return video


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

    return video


@NoiseRegistry.register("poisson") # 泊松噪声
def add_poisson_noise(video: torch.Tensor, ratio, gain=0.01) -> torch.Tensor:
    video = video.to('cuda')
    noise_indice = sample_noise_frame(video, ratio)
    for i in noise_indice:
        scaled = video[i] * gain
        noisy_scaled = torch.poisson(scaled)
        video[i] = noisy_scaled / gain
    return video


@NoiseRegistry.register("gaussian_blur") # 高斯模糊
def add_gaussian_blur(video: torch.Tensor, ratio, kernel_size=101, sigma=20) -> torch.Tensor:
    video = video.to('cuda')
    kernel = gaussian_kernel(kernel_size, sigma, video.device)
    kernel = kernel.view(1, 1, kernel_size, kernel_size)

    kernel = kernel.repeat(video.shape[-3], 1, 1, 1)
    padding = kernel_size // 2

    noise_indice = sample_noise_frame(video, ratio)
    for i in noise_indice:
        video[i] = torch.conv2d(video[i].unsqueeze(0).float(), kernel, padding=padding, groups=video[i].shape[-3])[0]
    return video


@NoiseRegistry.register("motion_blur") # 运动模糊
def add_motion_blur(video: torch.Tensor, ratio, kernel_size=101, angle=45) -> torch.Tensor:
    video = video.to('cuda')
    kernel = motion_kernel(kernel_size, angle, video.device)
    kernel = kernel.view(1, 1, kernel_size, kernel_size)

    kernel = kernel.repeat(video.shape[-3], 1, 1, 1)
    padding = kernel_size // 2

    noise_indice = sample_noise_frame(video, ratio)
    for i in noise_indice:
        video[i] = torch.conv2d(video[i].unsqueeze(0).float(), kernel, padding=padding, groups=video[i].shape[-3])[0]
    return video


@NoiseRegistry.register("jpeg_artifact")
def add_jpeg_noise(
    video: torch.Tensor,  # 输入形状 [T, C, H, W]
    ratio: float,
    intensity: float = 15,
    block_size: int = 32,
    subsample_chroma: bool = True
) -> torch.Tensor:
    # 确保高宽是块大小的整数倍
    video = video.to('cuda')
    _, _, H, W = video.shape
    H_pad = (H + block_size - 1) // block_size * block_size
    W_pad = (W + block_size - 1) // block_size * block_size
    pad = (0, W_pad - W, 0, H_pad - H)
    padded_video = F.pad(video, pad, mode='reflect')

    # 执行颜色空间转换
    ycbcr_video = rgb_to_ycbcr(padded_video)
    noise_indice = sample_noise_frame(video, ratio)
    for i in noise_indice:
        frame = ycbcr_video[i].unsqueeze(0)
        # 色度子采样（4:2:0）
        if subsample_chroma:
            # 下采样CbCr通道
            cbcr = F.interpolate(
                frame[:, 1:3, :, :].float(), 
                scale_factor=0.02, 
                mode='bicubic'
            )
            # 上采样回原尺寸
            cbcr = F.interpolate(
                cbcr, 
                size=(H_pad, W_pad), 
                mode='bicubic', 
            )
            frame = torch.cat([frame[:, 0:1], cbcr], dim=1)

        # 对每个通道独立添加块噪声
        noisy_ycbcr = torch.cat([
            add_block_artifacts(frame[:, i:i+1], block_size, intensity) for i in range(3)
        ], dim=1)

        # 转回RGB并裁剪
        noisy_rgb = ycbcr_to_rgb(noisy_ycbcr)
        video[i] = torch.clamp(noisy_rgb[:, :, :H, :W], 0.0, 255.0)
    return video