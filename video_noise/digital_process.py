# Denis
# -*- coding: utf-8 -*-
from .noise_applier import NoiseRegistry
from .utils import *
import cv2


@NoiseRegistry.register("random_pixel") # 像素值随机变化
def add_random_pixel_noise(
    video: torch.Tensor,
    ratio: float,
    probs: float = 0.8,  
    value_range: tuple = (0, 255)  # 噪声值范围
) -> torch.Tensor:
    video = video.to('cuda')
    _, C, H, W = video.shape
    device = video.device
    noise_indice = sample_noise_frame(video, ratio)
    
    for i in noise_indice:
        mask = torch.rand(1, H, W, device=device) < probs
        min_val, max_val = value_range
        noise = torch.rand(C, H, W, device=device) * (max_val - min_val) + min_val
        video[i] = torch.where(mask.expand(C, -1, -1), noise, video[i].float())
    return video.to(torch.uint8)


@NoiseRegistry.register("resolution_degrade")  # 降低分辨率
def add_resolution_noise(
    video: torch.Tensor,
    ratio: float,
    scale_factor: float = 0.1,
) -> torch.Tensor:
    _, _, H, W = video.shape
    
    new_H = int(H * scale_factor)
    new_W = int(W * scale_factor)
    
    noise_indice = sample_noise_frame(video, ratio)
    for i in noise_indice:
        # 下采样
        downsampled = F.interpolate(
            video[i].unsqueeze(0).float(),
            size=(new_H, new_W),
            mode='bicubic'
        )
        
        # 上采样回原始尺寸
        upsampled = F.interpolate(
            downsampled,
            size=(H, W),
            mode='bicubic'
        )
        video[i] = torch.clamp(upsampled, 0.0, 255.0)
    return video.to(torch.uint8)


@NoiseRegistry.register("stretch_squish")  # 图像拉伸/压缩
def add_stretch_squish_noise(
    video: torch.Tensor,
    ratio: float, 
    stretch_ratio : float = 1 / 30.0
) -> torch.Tensor:
    direction = "horizontal" if torch.rand(1) > 0.5 else "vertical"
    _, _, H, W = video.shape
    
    # 计算新尺寸
    if direction == "horizontal":
        new_W = int(W * stretch_ratio)
        new_H = H  # 保持高度不变
    else:
        new_H = int(H * stretch_ratio)
        new_W = W  # 保持宽度不变
    
    noise_indice = sample_noise_frame(video, ratio)
    for i in noise_indice:

        # 下采样（拉伸/压缩）
        distorted = F.interpolate(
            video[i].unsqueeze(0),
            size=(new_H, new_W),  
            mode='bicubic'
        )
        
        # 上采样回原始尺寸（模拟分辨率损失）
        output = F.interpolate(
            distorted,
            size=(H, W),
            mode='bicubic'
        )
        video[i] = torch.clamp(output, 0.0, 255.0)
    
    return video.to(torch.uint8)


@NoiseRegistry.register("edge_sawtooth")  # 边缘锯齿
def add_canny_jagged(
    video: torch.Tensor,
    ratio: float,  
    low_threshold=50, 
    high_threshold=150, 
    edge_dilate_size=2,    # 边缘膨胀核大小
    surround_noise_ratio=0.3  # 周边像素扰动比例
) -> torch.Tensor:
    video = video.cpu()
    np_frames = video.permute(0, 2, 3, 1).numpy().astype(np.uint8)  # NHWC
    noise_indice = sample_noise_frame(video, ratio) 
    
    # 定义膨胀核
    dilate_kernel = np.ones((edge_dilate_size, edge_dilate_size), np.uint8)
    
    noisy_frames = []
    for i, frame in enumerate(np_frames):
        if i in noise_indice:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, low_threshold, high_threshold)
            dilated_edges = cv2.dilate(edges, dilate_kernel, iterations=1)

            edge_mask = (dilated_edges == 255)
            h, w = edges.shape
            y_coords, x_coords = np.where(edge_mask)
            
            # 为每个边缘像素添加随机偏移的周边点
            surround_mask = np.zeros_like(edge_mask)
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0: continue
                    new_y = np.clip(y_coords + dy, 0, h-1)
                    new_x = np.clip(x_coords + dx, 0, w-1)
                    surround_mask[new_y, new_x] = True
            
            # 合并主边缘和随机周边区域
            combined_mask = edge_mask | (surround_mask & (np.random.rand(h,w) < surround_noise_ratio))
            
            noisy_pixels = np.random.randint(0, 256, (frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
            processed_frame = np.where(combined_mask[..., None], noisy_pixels, frame)
            noisy_frames.append(processed_frame)
        else:
            noisy_frames.append(frame)
    
    np_noisy = np.stack(noisy_frames, axis=0)  
    tensor_noisy = torch.from_numpy(np_noisy).permute(0, 3, 1, 2)
    return tensor_noisy.to(torch.uint8)


@NoiseRegistry.register("color_quantized")  # 色彩量化
def uniform_color_quantize_uint8(
    video: torch.Tensor, 
    ratio: float, 
    bits: int = 3
) -> torch.Tensor:
    levels = 2 ** bits
    step = 255.0 / (levels - 1)  
    
    noise_indice = sample_noise_frame(video, ratio) 
    for i in noise_indice:
        # 均匀量化公式
        quantized = torch.round(video[i].float() / step) * step
        video[i] = torch.clamp(quantized, 0.0, 255.0)
    return video.to(torch.uint8)
