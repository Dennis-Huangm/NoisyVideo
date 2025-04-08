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
    import cv2
    
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
        video[i] = frame_tensor.to(device).to(video.dtype)
    
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








