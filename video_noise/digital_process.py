# Denis
# -*- coding: utf-8 -*-
from .noise_applier import NoiseRegistry
from .utils import *
import cv2


@NoiseRegistry.register("rolling_shutter") # 像素值随机变化
def add_rolling_shutter(
    video: torch.Tensor,
    ratio: float,
    delay_factor: float = 2.0,
    buffer_size: int = 10
) -> torch.Tensor:
    """
    在视频张量的选定帧上应用滚动快门效应，确保效果均匀分布
    
    参数:
    - video: 形状为 [T, C, H, W] 的视频张量
    - ratio: 应用效果的帧比例 (0到1之间)
    - delay_factor: 控制滚动效果强度的因子
    - buffer_size: 用于模拟滚动效果的缓冲帧数量
    
    返回:
    - 应用了均匀滚动快门效应的视频张量
    """
    # 创建副本以避免修改原始数据
    result_video = video.clone()
    T, C, H, W = video.shape
    
    # 选择要应用效果的帧索引
    noise_indice = sample_noise_frame(video, ratio)
    
    # 确保缓冲区大小合理
    buffer_size = min(buffer_size, T-1)
    
    for t_idx in noise_indice:
        # 跳过无法处理的帧
        if t_idx < 1:
            continue
            
        # 决定方向：垂直(0)或水平(1)
        is_horizontal = torch.rand(1).item() < 0.5
        
        # 决定是否反向
        is_reverse = torch.rand(1).item() < 0.5
        
        # 获取可用的缓冲帧数
        available_buffer = min(t_idx, buffer_size)
        
        # 创建结果帧
        output_frame = torch.zeros_like(video[t_idx])
        
        if is_horizontal:  # 水平方向
            for x in range(W):
                # 计算此列应使用的帧
                if is_reverse:
                    # 从右到左
                    position = (W - 1 - x) / (W - 1)
                else:
                    # 从左到右
                    position = x / (W - 1)
                
                # 计算帧索引，确保使用完整范围
                frame_idx = t_idx - min(int(position * available_buffer * delay_factor), available_buffer)
                frame_idx = max(0, frame_idx)  # 确保索引不为负
                
                # 应用到结果帧
                output_frame[:, :, x] = video[frame_idx, :, :, x]
        else:  # 垂直方向
            for y in range(H):
                # 计算此行应使用的帧
                if is_reverse:
                    # 从下到上
                    position = (H - 1 - y) / (H - 1)
                else:
                    # 从上到下
                    position = y / (H - 1)
                
                # 计算帧索引，确保使用完整范围
                frame_idx = t_idx - min(int(position * available_buffer * delay_factor), available_buffer)
                frame_idx = max(0, frame_idx)  # 确保索引不为负
                
                # 应用到结果帧
                output_frame[:, y, :] = video[frame_idx, :, y, :]
        
        # 更新结果视频
        result_video[t_idx] = output_frame
    
    return result_video.to(torch.uint8)


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



@NoiseRegistry.register("elastic")
def add_elastic_transform(video: torch.Tensor, ratio: float, severity: int = 5) -> torch.Tensor:
    """
    为视频添加弹性变换效果
    
    参数:
        video: 输入视频张量 [T, C, H, W]
        ratio: 要处理的帧的比例
        severity: 变换的严重程度(1-5)
    """
    import torch
    import numpy as np
    from scipy.ndimage.interpolation import map_coordinates
    
    def gaussian_filter(image, sigma, mode='reflect', truncate=3.0):
        """
        高斯滤波的简单实现
        """
        # 计算核大小
        kernel_size = int(truncate * sigma + 0.5) * 2 + 1
        # 创建1D高斯核
        kernel_1d = np.exp(-0.5 * (np.arange(kernel_size) - (kernel_size - 1) / 2)**2 / sigma**2)
        kernel_1d = kernel_1d / np.sum(kernel_1d)
        
        # 对图像进行填充
        pad_width = kernel_size // 2
        if mode == 'reflect':
            padded = np.pad(image, pad_width, mode='reflect')
        else:
            padded = np.pad(image, pad_width, mode='constant')
        
        # 应用高斯模糊
        result = np.copy(image)
        
        # 水平方向滤波
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if len(image.shape) == 3:  # 彩色图像
                    for c in range(image.shape[2]):
                        window = padded[i:i+kernel_size, j+pad_width, c]
                        result[i, j, c] = np.sum(window * kernel_1d)
                else:  # 灰度图像
                    window = padded[i:i+kernel_size, j+pad_width]
                    result[i, j] = np.sum(window * kernel_1d)
        
        # 垂直方向滤波
        padded = np.pad(result, pad_width, mode='reflect' if mode == 'reflect' else 'constant')
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if len(image.shape) == 3:  # 彩色图像
                    for c in range(image.shape[2]):
                        window = padded[i+pad_width, j:j+kernel_size, c]
                        result[i, j, c] = np.sum(window * kernel_1d)
                else:  # 灰度图像
                    window = padded[i+pad_width, j:j+kernel_size]
                    result[i, j] = np.sum(window * kernel_1d)
        
        return result
    
    def apply_elastic_transform(image, severity=1):
        """
        对图像应用弹性变换
        """
        image = np.array(image, dtype=np.float32) / 255.
        shape = image.shape
        shape_size = shape[:2]
        
        # 设置变换参数
        sigma = np.array(shape_size) * 0.01
        alpha_values = [250 * 0.05, 250 * 0.065, 250 * 0.085, 250 * 0.1, 250 * 0.12]
        alpha = alpha_values[severity - 1]
        max_dx = shape[0] * 0.005
        max_dy = shape[0] * 0.005
        
        # 生成随机位移场
        try:
            # 尝试使用skimage的gaussian滤波
            from skimage.filters import gaussian
            dx = (gaussian(np.random.uniform(-max_dx, max_dx, size=shape[:2]),
                          sigma, mode='reflect', truncate=3) * alpha).astype(np.float32)
            dy = (gaussian(np.random.uniform(-max_dy, max_dy, size=shape[:2]),
                          sigma, mode='reflect', truncate=3) * alpha).astype(np.float32)
        except ImportError:
            # 如果skimage不可用，使用自定义高斯滤波
            dx = (gaussian_filter(np.random.uniform(-max_dx, max_dx, size=shape[:2]),
                                 sigma[0], mode='reflect', truncate=3) * alpha).astype(np.float32)
            dy = (gaussian_filter(np.random.uniform(-max_dy, max_dy, size=shape[:2]),
                                 sigma[0], mode='reflect', truncate=3) * alpha).astype(np.float32)
        
        # 应用变换
        if len(image.shape) < 3 or image.shape[2] < 3:
            # 灰度图像
            x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
            indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
        else:
            # 彩色图像
            dx, dy = dx[..., np.newaxis], dy[..., np.newaxis]
            x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
            indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))
        
        # 使用双线性插值对扭曲后的图像进行重采样
        transformed = map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)
        
        return np.clip(transformed, 0, 1) * 255
    
    # 选择要添加弹性变换效果的帧
    noise_indices = sample_noise_frame(video, ratio)
    device = video.device
    
    # 设置弹性变换的严重程度(1-5)
    severity = max(1, min(severity, 5))  # 确保severity在1-5范围内
    
    # 处理选定的帧
    for i in noise_indices:
        # 将帧转换为numpy数组
        frame_np = video[i].permute(1, 2, 0).cpu().numpy()  # [H, W, C]
        
        # 应用弹性变换
        transformed_frame = apply_elastic_transform(frame_np, severity)
        
        # 转回PyTorch张量
        if len(frame_np.shape) == 2 or frame_np.shape[2] == 1:  # 灰度
            frame_tensor = torch.from_numpy(transformed_frame).squeeze(-1).unsqueeze(0)
        else:  # RGB或其他多通道
            frame_tensor = torch.from_numpy(transformed_frame).permute(2, 0, 1)
        
        # 更新原视频
        video[i] = frame_tensor.to(device)
    
    return video.to(torch.uint8)
