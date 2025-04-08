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


@NoiseRegistry.register("defocus_blur")
def add_defocus_blur(video: torch.Tensor, ratio: float, severity: int = 5) -> torch.Tensor:
    """
    为视频添加离焦模糊效果
    
    参数:
        video: 输入视频张量 [T, C, H, W]
        ratio: 要处理的帧的比例
        severity: 模糊的严重程度(1-5)
    """
    import torch
    import numpy as np
    import cv2
    
    def disk(radius, alias_blur=0.1, dtype=np.float32):
        """生成一个圆盘形卷积核"""
        if radius <= 8:
            L = np.arange(-8, 8 + 1)
            ksize = (3, 3)
        else:
            L = np.arange(-radius, radius + 1)
            ksize = (5, 5)
        X, Y = np.meshgrid(L, L)
        aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
        aliased_disk /= np.sum(aliased_disk)

        # 对磁盘进行超采样以消除锯齿
        return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)
    
    def apply_defocus_blur(x, severity=1):
        """对单帧应用离焦模糊效果"""
        # 根据严重程度选择参数
        c = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)][severity - 1]

        x = np.array(x) / 255.
        kernel = disk(radius=c[0], alias_blur=c[1])

        channels = []
        if len(x.shape) < 3 or x.shape[2] < 3:
            # 灰度图像
            channels = np.array(cv2.filter2D(x, -1, kernel))
        else:
            # 彩色图像
            for d in range(3):
                channels.append(cv2.filter2D(x[:, :, d], -1, kernel))
            channels = np.array(channels).transpose((1, 2, 0))

        return np.clip(channels, 0, 1) * 255
    
    # 选择要添加离焦模糊效果的帧
    noise_indices = sample_noise_frame(video, ratio)
    device = video.device
    
    # 设置模糊的严重程度(1-5)
    severity = max(1, min(severity, 5))  # 确保severity在1-5范围内
    
    # 处理选定的帧
    for i in noise_indices:
        # 将帧转换为numpy数组
        frame_np = video[i].permute(1, 2, 0).cpu().numpy()  # [H, W, C]
        
        # 应用离焦模糊
        blurred_frame = apply_defocus_blur(frame_np, severity)
        
        # 转回PyTorch张量
        if len(frame_np.shape) == 2 or frame_np.shape[2] == 1:  # 灰度
            frame_tensor = torch.from_numpy(blurred_frame).squeeze(-1).unsqueeze(0)
        else:  # RGB或其他多通道
            frame_tensor = torch.from_numpy(blurred_frame).permute(2, 0, 1)
        
        # 更新原视频
        video[i] = frame_tensor.to(device).to(video.dtype)
    
    return video


@NoiseRegistry.register("glass_blur")
def add_glass_blur(video: torch.Tensor, ratio: float, severity: int = 5) -> torch.Tensor:
    """
    为视频添加玻璃模糊效果
    
    参数:
        video: 输入视频张量 [T, C, H, W]
        ratio: 要处理的帧的比例
        severity: 模糊的严重程度(1-5)
    """
    import torch
    import numpy as np
    import cv2
    
    # 尝试导入numba和skimage，如果不可用则使用替代方法
    try:
        from numba import njit
        from skimage.filters import gaussian as sk_gaussian
        has_numba = True
        
        @njit()
        def _shuffle_pixels_njit(d0, d1, x, c):
            """使用Numba加速的像素打乱函数"""
            # 局部打乱像素
            for i in range(c[2]):
                for h in range(d0 - c[1], c[1], -1):
                    for w in range(d1 - c[1], c[1], -1):
                        if (h < 0 or w < 0 or h >= d0 or w >= d1):
                            continue
                        dx, dy = np.random.randint(-c[1], c[1], size=(2,))
                        h_prime, w_prime = h + dy, w + dx
                        # 检查边界
                        if (h_prime < 0 or w_prime < 0 or h_prime >= d0 or w_prime >= d1):
                            continue
                        # 交换像素
                        x[h, w], x[h_prime, w_prime] = x[h_prime, w_prime].copy(), x[h, w].copy()
            return x
        
        def gaussian(img, sigma, channel_axis=None):
            """使用skimage的高斯滤波"""
            return sk_gaussian(img, sigma=sigma, channel_axis=channel_axis)
            
    except ImportError:
        has_numba = False
        
        def _shuffle_pixels_fast(d0, d1, x, c):
            """非Numba版本的像素打乱，使用向量化操作加速"""
            result = np.copy(x)
            
            # 为每个迭代创建随机位移数组
            for _ in range(c[2]):
                # 创建随机位移矩阵
                h_indices = np.arange(c[1], d0 - c[1])
                w_indices = np.arange(c[1], d1 - c[1])
                H, W = np.meshgrid(h_indices, w_indices)
                H = H.flatten()
                W = W.flatten()
                
                # 随机打乱顺序，只处理部分像素以加速
                sample_size = min(10000, len(H))  # 限制处理的像素数量
                if sample_size < len(H):
                    idx = np.random.choice(len(H), sample_size, replace=False)
                    H, W = H[idx], W[idx]
                
                # 为每个像素生成随机位移
                dxdy = np.random.randint(-c[1], c[1], size=(len(H), 2))
                H_prime = H + dxdy[:, 0]
                W_prime = W + dxdy[:, 1]
                
                # 确保在图像边界内
                valid = (H_prime >= 0) & (W_prime >= 0) & (H_prime < d0) & (W_prime < d1)
                H, W = H[valid], W[valid]
                H_prime, W_prime = H_prime[valid], W_prime[valid]
                
                # 交换像素 (仅处理一个通道，然后复制到其他通道)
                if len(x.shape) == 3:
                    # 彩色图像
                    for ch in range(x.shape[2]):
                        temp = result[H, W, ch].copy()
                        result[H, W, ch] = result[H_prime, W_prime, ch]
                        result[H_prime, W_prime, ch] = temp
                else:
                    # 灰度图像
                    temp = result[H, W].copy()
                    result[H, W] = result[H_prime, W_prime]
                    result[H_prime, W_prime] = temp
                    
            return result
            
        def gaussian(img, sigma, channel_axis=None):
            """简单的高斯模糊实现"""
            # 使用OpenCV的高斯模糊
            ksize = int(sigma * 5) | 1  # 确保为奇数
            if channel_axis is None or len(img.shape) == 2:
                return cv2.GaussianBlur(img, (ksize, ksize), sigma)
            else:
                result = np.copy(img)
                # 处理彩色图像
                if channel_axis == -1 or channel_axis == 2:
                    for i in range(img.shape[2]):
                        result[..., i] = cv2.GaussianBlur(img[..., i], (ksize, ksize), sigma)
                return result
    
    def apply_glass_blur(x, severity=1):
        """对单帧应用玻璃模糊效果"""
        # sigma, max_delta, iterations
        c = [(0.7, 1, 2), (0.9, 2, 1), (1, 2, 3), (1.1, 3, 2), (1.5, 4, 2)][severity - 1]

        # 第一次高斯模糊
        x = np.uint8(gaussian(np.array(x) / 255., sigma=c[0], channel_axis=-1) * 255)

        # 局部像素打乱
        if has_numba:
            x = _shuffle_pixels_njit(np.array(x).shape[0], np.array(x).shape[1], x.copy(), c)
        else:
            x = _shuffle_pixels_fast(np.array(x).shape[0], np.array(x).shape[1], x, c)

        # 第二次高斯模糊
        return np.clip(gaussian(x / 255., sigma=c[0], channel_axis=-1), 0, 1) * 255
    
    # 选择要添加玻璃模糊效果的帧
    noise_indices = sample_noise_frame(video, ratio)
    device = video.device
    
    # 设置模糊的严重程度(1-5)
    severity = max(1, min(severity, 5))  # 确保severity在1-5范围内
    
    # 处理选定的帧
    for i in noise_indices:
        # 将帧转换为numpy数组
        frame_np = video[i].permute(1, 2, 0).cpu().numpy()  # [H, W, C]
        
        # 应用玻璃模糊
        blurred_frame = apply_glass_blur(frame_np, severity)
        
        # 转回PyTorch张量
        if len(frame_np.shape) == 2 or frame_np.shape[2] == 1:  # 灰度
            frame_tensor = torch.from_numpy(blurred_frame).squeeze(-1).unsqueeze(0)
        else:  # RGB或其他多通道
            frame_tensor = torch.from_numpy(blurred_frame).permute(2, 0, 1)
        
        # 更新原视频
        video[i] = frame_tensor.to(device).to(video.dtype)
    
    return video


@NoiseRegistry.register("zoom_blur")
def add_zoom_blur(video: torch.Tensor, ratio: float, severity: int = 5) -> torch.Tensor:
    """
    为视频添加缩放模糊效果（优化版）
    
    参数:
        video: 输入视频张量 [T, C, H, W]
        ratio: 要处理的帧的比例
        severity: 模糊的严重程度(1-5)
    """
    import torch
    import numpy as np
    import cv2
    
    def apply_zoom_blur_fast(x, severity=1):
        """对单帧应用缩放模糊效果（优化版）"""
        # 根据严重程度选择缩放因子
        c = [np.arange(1, 1.11, 0.01),
             np.arange(1, 1.16, 0.01),
             np.arange(1, 1.21, 0.02),
             np.arange(1, 1.26, 0.02),
             np.arange(1, 1.31, 0.03)][severity - 1]

        x = np.array(x, dtype=np.float32) / 255.
        h, w = x.shape[:2]
        
        # 创建累积结果图像
        out = x.copy()
        
        # 使用更少的缩放因子
        if len(c) > 10:
            # 如果缩放因子太多，只取部分
            step = max(1, len(c) // 10)
            c = c[::step]
        
        # 使用OpenCV的缩放函数替代scipy的缩放
        for zoom_factor in c:
            # 计算缩放后的尺寸
            zh, zw = int(h / zoom_factor), int(w / zoom_factor)
            
            # 确保非零尺寸
            zh, zw = max(1, zh), max(1, zw)
            
            # 计算裁剪区域
            top = (h - zh) // 2
            left = (w - zw) // 2
            
            # 确保有效的裁剪区域
            if top < 0 or left < 0 or top + zh > h or left + zw > w:
                continue
            
            # 裁剪中心区域
            cropped = x[top:top + zh, left:left + zw].copy()
            
            # 使用OpenCV的resize函数进行缩放（更快）
            if len(x.shape) == 3:  # 彩色图像
                zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
            else:  # 灰度图像
                zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
            
            # 累加缩放层
            out += zoomed
        
        # 计算平均值
        out /= (len(c) + 1)
        
        return np.clip(out, 0, 1) * 255
    
    # 选择要添加缩放模糊效果的帧
    noise_indices = sample_noise_frame(video, ratio)
    device = video.device
    
    # 设置模糊的严重程度(1-5)
    severity = max(1, min(severity, 5))  # 确保severity在1-5范围内
    
    # 处理选定的帧
    for i in noise_indices:
        # 将帧转换为numpy数组
        frame_np = video[i].permute(1, 2, 0).cpu().numpy()  # [H, W, C]
        
        # 应用缩放模糊
        blurred_frame = apply_zoom_blur_fast(frame_np, severity)
        
        # 转回PyTorch张量
        if len(frame_np.shape) == 2 or frame_np.shape[2] == 1:  # 灰度
            frame_tensor = torch.from_numpy(blurred_frame).squeeze(-1).unsqueeze(0)
        else:  # RGB或其他多通道
            frame_tensor = torch.from_numpy(blurred_frame).permute(2, 0, 1)
        
        # 更新原视频
        video[i] = frame_tensor.to(device).to(video.dtype)
    
    return video
@NoiseRegistry.register("jpeg_artifact")
def add_jpeg_noise(
    video: torch.Tensor,  # 输入形状 [T, C, H, W]
    ratio: float,
    quality: int = 1,    # 直接使用JPEG质量参数，范围1-100，默认值设低以产生明显失真
) -> torch.Tensor:
    """
    对视频中随机选择的帧添加真实的JPEG压缩噪声
    
    参数:
        video: 输入视频张量 [T, C, H, W]
        ratio: 要处理的帧的比例
        quality: JPEG压缩质量参数(1-100)，值越低失真越严重
    """
    video = video.to('cuda')
    noise_indice = sample_noise_frame(video, ratio)
    
    for i in noise_indice:
        # 提取当前帧并转换为PIL图像
        frame = video[i].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        pil_frame = Image.fromarray(frame)
        
        # 应用真实的JPEG压缩
        output = BytesIO()
        pil_frame.save(output, 'JPEG', quality=quality)
        output.seek(0)
        compressed_frame = Image.open(output)
        
        # 转回Tensor并放回视频
        compressed_array = np.array(compressed_frame)
        compressed_tensor = torch.from_numpy(compressed_array).permute(2, 0, 1).to(video.device)
        video[i] = compressed_tensor
    
    return video
# def add_jpeg_noise(
#     video: torch.Tensor,  # 输入形状 [T, C, H, W]
#     ratio: float,
#     intensity: float = 15, # 15
#     block_size: int = 32, # 32
#     subsample_chroma: bool = True
# ) -> torch.Tensor:
#     # 确保高宽是块大小的整数倍
#     video = video.to('cuda')
#     _, _, H, W = video.shape
#     H_pad = (H + block_size - 1) // block_size * block_size
#     W_pad = (W + block_size - 1) // block_size * block_size
#     pad = (0, W_pad - W, 0, H_pad - H)
#     video_float = video.float()
#     padded_video = F.pad(video_float, pad, mode='reflect')

#     # 执行颜色空间转换
#     ycbcr_video = rgb_to_ycbcr(padded_video)
#     noise_indice = sample_noise_frame(video, ratio)
#     for i in noise_indice:
#         frame = ycbcr_video[i].unsqueeze(0)
#         # 色度子采样（4:2:0）
#         if subsample_chroma:
#             # 下采样CbCr通道
#             cbcr = F.interpolate(
#                 frame[:, 1:3, :, :].float(), 
#                 scale_factor=0.02, 
#                 mode='bicubic'
#             )
#             # 上采样回原尺寸
#             cbcr = F.interpolate(
#                 cbcr, 
#                 size=(H_pad, W_pad), 
#                 mode='bicubic', 
#             )
#             frame = torch.cat([frame[:, 0:1], cbcr], dim=1)

#         # 对每个通道独立添加块噪声
#         noisy_ycbcr = torch.cat([
#             add_block_artifacts(frame[:, i:i+1], block_size, intensity) for i in range(3)
#         ], dim=1)

#         # 转回RGB并裁剪
#         noisy_rgb = ycbcr_to_rgb(noisy_ycbcr)
#         video[i] = torch.clamp(noisy_rgb[:, :, :H, :W], 0.0, 255.0)
#     return video