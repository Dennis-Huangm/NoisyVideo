# Denis
# -*- coding: utf-8 -*-
from .noise_applier import NoiseRegistry
from .utils import *
from torchvision.transforms import functional as F
import cv2
from torchvision.io import read_image


@NoiseRegistry.register("rainy") # 雨
def add_rain_fractal(video: torch.Tensor, ratio: float, severity: int = 5) -> torch.Tensor:
    """
    使用分形生成和图像处理技术添加雨效果到视频中
    
    参数:
        video: 输入视频张量 [T, C, H, W]
        ratio: 要处理的帧的比例
        severity: 雨的严重程度(1-5)
    """
    import torch
    import numpy as np
    import torch.nn.functional as F
    from scipy import ndimage
    
    def generate_rain_map(shape, slant, drop_length, rain_density):
        """
        生成雨滴图
        
        参数:
            shape: 输出图像形状 (H, W)
            slant: 雨滴倾斜角度
            drop_length: 雨滴长度
            rain_density: 雨滴密度 (0-1)
        """
        H, W = shape
        # 创建空白图像
        rain_map = np.zeros((H, W), dtype=np.float32)
        
        # 根据密度确定雨滴数量
        num_drops = int(rain_density * H * W // 100)  # 减少雨滴总数
        
        # 随机生成雨滴起始位置
        x_pos = np.random.randint(0, W, num_drops)
        y_pos = np.random.randint(0, H, num_drops)
        
        # 生成随机雨滴长度
        drop_lengths = np.random.randint(drop_length // 2, drop_length * 2, size=num_drops)
        
        # 绘制每个雨滴
        for i in range(num_drops):
            # 计算雨滴终点
            x_end = x_pos[i] + int(slant * drop_lengths[i])
            y_end = y_pos[i] + drop_lengths[i]
            
            # 确保终点在图像内
            if x_end >= W:
                x_end = W - 1
            if y_end >= H:
                y_end = H - 1
            
            # 绘制雨滴线条
            rr, cc = np.linspace(y_pos[i], y_end, drop_lengths[i]).astype(np.int32), \
                    np.linspace(x_pos[i], x_end, drop_lengths[i]).astype(np.int32)
            
            # 限制在图像范围内
            valid_indices = (rr < H) & (cc < W) & (rr >= 0) & (cc >= 0)
            rr, cc = rr[valid_indices], cc[valid_indices]
            
            # 设置雨滴强度（从强到弱）
            drop_intensity = np.linspace(0.7, 0.2, len(rr))  # 降低雨滴亮度
            rain_map[rr, cc] = np.maximum(rain_map[rr, cc], drop_intensity)
        
        # 对雨滴图进行模糊以增加真实感
        rain_map = ndimage.gaussian_filter(rain_map, sigma=0.5)
        return rain_map
    
    def apply_rain_effect(image, rain_map, brightness_factor, contrast_factor, fog_factor):
        """
        将雨效果应用到图像上
        """
        # 转换图像到浮点数
        image_float = image.astype(np.float32)
        
        # 创建雨滴层
        rain_layer = np.expand_dims(rain_map, axis=-1) if len(image.shape) > 2 else rain_map
        if len(image.shape) > 2:
            rain_layer = np.repeat(rain_layer, image.shape[2], axis=2)
        
        # 添加雨滴
        rain_affected = image_float * (1 - rain_layer * 0.3) + rain_layer * 200 * brightness_factor
        
        # 轻微调整对比度
        mean = np.mean(rain_affected, axis=(0, 1), keepdims=True)
        rain_affected = (rain_affected - mean) * contrast_factor + mean
        
        # 添加轻微雾气效果
        rain_affected = image_float * (1 - fog_factor) + np.ones_like(image_float) * fog_factor * 120
        
        # 叠加雨滴
        rain_affected += rain_layer * 200 * brightness_factor
        
        return np.clip(rain_affected, 0, 255).astype(np.uint8)
    
    # 选择要添加雨效果的帧
    noise_indices = sample_noise_frame(video, ratio)
    device = video.device
    
    # 根据严重程度选择参数
    # (雨滴倾斜角度, 雨滴长度, 雨滴密度, 亮度因子, 对比度因子, 雾气因子)
#     (0.05, 10, 1, 0.6, 1.1), # 轻微
# (0.07, 14, 2, 0.65, 1.1), # 较轻
# (0.09, 18, 3, 0.7, 1.15), # 中等
# (0.11, 20, 4, 0.75, 1.15), # 较重
# (0.13, 25, 5, 0.8, 1.2) # 暴雨


    rain_params = [
        (0.05, 8, 1, 0.2, 1.03, 0.05),  # 轻微
        (0.07, 10, 2, 0.25, 1.04, 0.07),  # 较轻
        (0.09, 15, 3, 0.30, 1.05, 0.10),  # 中等
        (0.11, 18, 4, 0.35, 1.06, 0.13),  # 较重
        (0.13, 22, 5, 0.40, 1.07, 0.15)   # 暴雨
    ]
    
    params = rain_params[min(severity - 1, 4)]  # 确保severity在1-5范围内
    slant, drop_length, rain_density, brightness_factor, contrast_factor, fog_factor = params
    
    # 获取视频帧的高和宽
    _, _, H, W = video.shape
    
    # 生成雨滴图
    rain_map = generate_rain_map((H, W), slant, drop_length, rain_density)
    
    # 处理选定的帧
    for i in noise_indices:
        # 将帧转换为numpy数组
        frame_np = video[i].permute(1, 2, 0).cpu().numpy().astype(np.uint8)  # [H, W, C]
        
        # 应用雨效果
        rainy_frame = apply_rain_effect(frame_np, rain_map, brightness_factor, contrast_factor, fog_factor)
        
        # 转回PyTorch张量
        if len(frame_np.shape) == 2 or frame_np.shape[2] == 1:  # 灰度
            frame_tensor = torch.from_numpy(rainy_frame).squeeze(-1).unsqueeze(0)
        else:  # RGB或其他多通道
            frame_tensor = torch.from_numpy(rainy_frame).permute(2, 0, 1)
        
        # 更新原视频
        video[i] = frame_tensor.to(device)
    
    return video.to(torch.uint8)
# def add_rain(video: torch.Tensor, ratio: float) -> torch.Tensor:
#     noise_indices = sample_noise_frame(video, ratio)
#     device = video.device

#     rain_layer = read_image('video_noise/sample/template_rain.png').float().to(device)
#     alpha, belta = 0.6, 1.0
#     rain_layer = F.resize(rain_layer, [video.shape[2], video.shape[3]])

#     for i in noise_indices:
#         synthetic = alpha * rain_layer + belta * video[i]
#         video[i] = torch.clamp(synthetic, 0, 255).to(torch.uint8)
#     return video


@NoiseRegistry.register("foggy") # 雾
def add_fog_fractal(video: torch.Tensor, ratio: float, severity: int = 5) -> torch.Tensor:
    """
    使用分形噪声(plasma fractal)添加雾效果到视频中
    
    参数:
        video: 输入视频张量 [T, C, H, W]
        ratio: 要处理的帧的比例
        severity: 雾的严重程度(1-5)
    """
    import torch
    import numpy as np
    import torch.nn.functional as F
    
    def plasma_fractal(mapsize=256, wibbledecay=3):
        """
        使用diamond-square算法生成高度图
        返回大小为mapsize x mapsize的浮点数组，值范围0-1
        """
        assert (mapsize & (mapsize - 1) == 0)
        maparray = np.empty((mapsize, mapsize), dtype=np.float32)
        maparray[0, 0] = 0
        stepsize = mapsize
        wibble = 100

        def wibbledmean(array):
            return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)

        def fillsquares():
            cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
            squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
            squareaccum += np.roll(squareaccum, shift=-1, axis=1)
            maparray[stepsize // 2:mapsize:stepsize,
                    stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

        def filldiamonds():
            mapsize = maparray.shape[0]
            drgrid = maparray[stepsize // 2:mapsize:stepsize,
                            stepsize // 2:mapsize:stepsize]
            ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
            ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
            lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
            ltsum = ldrsum + lulsum
            maparray[0:mapsize:stepsize,
                    stepsize // 2:mapsize:stepsize] = wibbledmean(ltsum)
            tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
            tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
            ttsum = tdrsum + tulsum
            maparray[stepsize // 2:mapsize:stepsize,
                    0:mapsize:stepsize] = wibbledmean(ttsum)

        while stepsize >= 2:
            fillsquares()
            filldiamonds()
            stepsize //= 2
            wibble /= wibbledecay

        maparray -= maparray.min()
        return maparray / maparray.max()
    
    def next_power_of_2(x):
        return 1 if x == 0 else 2 ** (x - 1).bit_length()
    
    # 选择要添加雾效果的帧
    noise_indices = sample_noise_frame(video, ratio)
    device = video.device
    
    # 根据严重程度选择参数
    fog_params = [(1.5, 2), (2., 2), (2.5, 1.7), (2.5, 1.5), (3., 1.4)]
    c = fog_params[min(severity - 1, 4)]  # 确保severity在1-5范围内
    
    # 获取视频帧的高和宽
    _, _, H, W = video.shape
    max_side = max(H, W)
    map_size = next_power_of_2(int(max_side))
    
    # 生成分形雾
    fog_fractal = plasma_fractal(mapsize=map_size, wibbledecay=c[1])
    # 裁剪到视频帧大小
    fog_fractal = fog_fractal[:H, :W]
    
    # 处理选定的帧
    for i in noise_indices:
        # 将帧转换为numpy数组并归一化
        frame_np = video[i].permute(1, 2, 0).cpu().numpy() / 255.0  # [H, W, C]
        max_val = frame_np.max()
        
        # 应用雾效果
        if frame_np.shape[2] == 1:  # 灰度图像
            frame_np += c[0] * fog_fractal
        else:  # RGB或其他多通道
            frame_np += c[0] * fog_fractal[..., np.newaxis]
        
        # 缩放并裁剪结果
        frame_np = np.clip(frame_np * max_val / (max_val + c[0]), 0, 1) * 255.0
        
        # 转回PyTorch张量
        if frame_np.shape[2] == 1:  # 灰度
            frame_tensor = torch.from_numpy(frame_np).squeeze(-1).unsqueeze(0)
        else:  # RGB
            frame_tensor = torch.from_numpy(frame_np).permute(2, 0, 1)
        
        # 更新原视频
        video[i] = frame_tensor.to(device)
    
    return video.to(torch.uint8)
# def add_rain(video: torch.Tensor, ratio: float) -> torch.Tensor:
#     noise_indices = sample_noise_frame(video, ratio)
#     device = video.device

#     rain_layer = read_image('video_noise/sample/template_fog.png').float().to(device)
#     alpha, belta = 0.7, 0.5
#     rain_layer = F.resize(rain_layer, [video.shape[2], video.shape[3]])

#     for i in noise_indices:
#         synthetic = alpha * rain_layer + belta * video[i]
#         video[i] = torch.clamp(synthetic, 0, 255).to(torch.uint8)
#     return video

@NoiseRegistry.register("snow") # 雪
def add_snow_effect(video: torch.Tensor, ratio: float, severity: int = 5) -> torch.Tensor:
    """
    为视频添加雪效果
    
    参数:
        video: 输入视频张量 [T, C, H, W]
        ratio: 要处理的帧的比例
        severity: 雪的严重程度(1-5)
    """
    import torch
    import numpy as np
    import cv2
    from scipy.ndimage import zoom as scizoom
    from scipy.ndimage import shift
    import math
    
    def getOptimalKernelWidth1D(radius, sigma):
        return max(2, int(2 * math.ceil(radius) + 1))
    
    def getMotionBlurKernel(width, sigma):
        kernel = np.zeros(width)
        kernel[width // 2] = 1
        return gaussian_filter1d(kernel, sigma)
    
    def gaussian_filter1d(input_array, sigma):
        """简单的高斯滤波实现"""
        size = int(sigma * 4) * 2 + 1
        x = np.arange(-(size // 2), size // 2 + 1)
        kernel = np.exp(-x**2 / (2 * sigma**2))
        kernel = kernel / np.sum(kernel)
        return np.convolve(input_array, kernel, mode='same')
    
    def clipped_zoom(img, zoom_factor):
        # 在宽度维度上裁剪:
        ch0 = int(np.ceil(img.shape[0] / float(zoom_factor)))
        top0 = (img.shape[0] - ch0) // 2

        # 在高度维度上裁剪:
        ch1 = int(np.ceil(img.shape[1] / float(zoom_factor)))
        top1 = (img.shape[1] - ch1) // 2

        if top0 + ch0 > img.shape[0] or top1 + ch1 > img.shape[1]:
            # 确保不超出边界
            ch0 = min(ch0, img.shape[0] - top0)
            ch1 = min(ch1, img.shape[1] - top1)

        cropped = img[top0:top0 + ch0, top1:top1 + ch1]
        
        # 处理不同维度的输入
        if len(img.shape) == 3:
            return scizoom(cropped, (zoom_factor, zoom_factor, 1), order=1)
        else:
            return scizoom(cropped, (zoom_factor, zoom_factor), order=1)
    
    def _motion_blur(x, radius, sigma, angle):
        width = getOptimalKernelWidth1D(radius, sigma)
        kernel = getMotionBlurKernel(width, sigma)
        point = (width * np.sin(np.deg2rad(angle)), width * np.cos(np.deg2rad(angle)))
        hypot = math.hypot(point[0], point[1])

        blurred = np.zeros_like(x, dtype=np.float32)
        for i in range(width):
            dy = -math.ceil(((i*point[0]) / hypot) - 0.5)
            dx = -math.ceil(((i*point[1]) / hypot) - 0.5)
            if (np.abs(dy) >= x.shape[0] or np.abs(dx) >= x.shape[1]):
                # 当模拟的运动超出图像边界时停止
                break
            shifted = shift(x, (dy, dx) if len(x.shape) == 2 else (dy, dx, 0))
            blurred = blurred + kernel[i] * shifted
        return blurred
    
    def apply_snow_effect(x, severity=1):
        snow_params = [
            (0.1, 0.3, 3, 0.5, 10, 4, 0.8),
            (0.2, 0.3, 2, 0.5, 12, 4, 0.7),
            (0.55, 0.3, 4, 0.9, 12, 8, 0.7),
            (0.55, 0.3, 4.5, 0.85, 12, 8, 0.65),
            (0.55, 0.3, 2.5, 0.85, 12, 12, 0.55)
        ][severity - 1]

        x = np.array(x, dtype=np.float32) / 255.
        
        # 生成雪花层
        snow_layer = np.random.normal(size=x.shape[:2], loc=snow_params[0], scale=snow_params[1])

        # 放大雪花
        snow_layer = clipped_zoom(snow_layer[..., np.newaxis] if len(snow_layer.shape) == 2 else snow_layer, 
                                 snow_params[2])
        
        # 阈值处理
        if len(snow_layer.shape) == 3:
            snow_layer[snow_layer < snow_params[3]] = 0
            snow_layer = np.clip(snow_layer.squeeze(), 0, 1)
        else:
            snow_layer[snow_layer < snow_params[3]] = 0
            snow_layer = np.clip(snow_layer, 0, 1)

        # 应用运动模糊使雪花看起来更真实
        snow_layer = _motion_blur(snow_layer, radius=snow_params[4], 
                                 sigma=snow_params[5], 
                                 angle=np.random.uniform(-135, -45))

        # 将雪花层四舍五入并裁剪到图像尺寸
        snow_layer = np.round(snow_layer * 255).astype(np.uint8) / 255.
        
        # 确保形状正确
        if len(snow_layer.shape) == 2:
            snow_layer = snow_layer[..., np.newaxis]
            
        # 裁剪到原图尺寸
        snow_layer = snow_layer[:x.shape[0], :x.shape[1], :] if len(snow_layer.shape) == 3 else snow_layer[:x.shape[0], :x.shape[1]]

        # 应用雪效果
        if len(x.shape) < 3 or x.shape[2] == 1:
            # 处理灰度图像
            x = snow_params[6] * x + (1 - snow_params[6]) * np.maximum(
                x, x.reshape(x.shape[0], x.shape[1]) * 1.5 + 0.5)
            if snow_layer.shape[-1] == 1:
                snow_layer = snow_layer.squeeze(-1)
        else:
            # 处理彩色图像
            gray = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
            gray = gray.reshape(x.shape[0], x.shape[1], 1)
            x = snow_params[6] * x + (1 - snow_params[6]) * np.maximum(x, gray * 1.5 + 0.5)

        try:
            # 叠加雪花层
            return np.clip(x + snow_layer + np.rot90(snow_layer, k=2), 0, 1) * 255
        except ValueError:
            # 异常处理
            x[:snow_layer.shape[0], :snow_layer.shape[1]] += snow_layer
            if len(x.shape) == 3 and len(snow_layer.shape) == 2:
                x[:snow_layer.shape[0], :snow_layer.shape[1]] += np.rot90(snow_layer, k=2)[..., np.newaxis]
            else:
                x[:snow_layer.shape[0], :snow_layer.shape[1]] += np.rot90(snow_layer, k=2)
            return np.clip(x, 0, 1) * 255
    
    # 选择要添加雪效果的帧
    noise_indices = sample_noise_frame(video, ratio)
    device = video.device
    
    # 设置雪效果的严重程度(1-5)
    severity = max(1, min(severity, 5))  # 确保severity在1-5范围内
    
    # 处理选定的帧
    for i in noise_indices:
        # 将帧转换为numpy数组
        frame_np = video[i].permute(1, 2, 0).cpu().numpy()  # [H, W, C]
        
        # 应用雪效果
        snowy_frame = apply_snow_effect(frame_np, severity)
        
        # 转回PyTorch张量
        if len(frame_np.shape) == 2 or frame_np.shape[2] == 1:  # 灰度
            frame_tensor = torch.from_numpy(snowy_frame).squeeze(-1).unsqueeze(0)
        else:  # RGB或其他多通道
            frame_tensor = torch.from_numpy(snowy_frame).permute(2, 0, 1)
        
        # 更新原视频
        video[i] = frame_tensor.to(device)
    
    return video.to(torch.uint8)


@NoiseRegistry.register("frost")
def add_frost_effect(video: torch.Tensor, ratio: float, severity: int = 5, frost_dir: str = 'video_noise/sample/') -> torch.Tensor:
    """
    为视频添加霜冻/冰霜效果
    
    参数:
        video: 输入视频张量 [T, C, H, W]
        ratio: 要处理的帧的比例
        severity: 霜冻的严重程度(1-5)
        frost_dir: 包含霜冻纹理图像的目录路径
    """
    import torch
    import numpy as np
    import cv2
    import os
    
    def rgb2gray(rgb):
        """将RGB图像转换为灰度图像"""
        return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
    
    def apply_frost_effect(x, severity=1, frost_dir='./frost/'):
        frost_params = [
            (1, 0.4),
            (0.8, 0.6),
            (0.7, 0.7),
            (0.65, 0.7),
            (0.6, 0.75)
        ][severity - 1]
        
        # 霜冻纹理文件名列表
        frost_files = [
            'frost1.png', 'frost2.png', 'frost3.png',
            'frost4.jpg', 'frost5.jpg', 'frost6.jpg'
        ]
        
        # 随机选择一个霜冻纹理
        idx = np.random.randint(len(frost_files))
        filename = os.path.join(frost_dir, frost_files[idx])
        
        # 尝试读取霜冻纹理图像
        try:
            frost = cv2.imread(filename)
            if frost is None:
                # 如果找不到文件，生成一个简单的随机霜冻纹理
                frost_shape = (max(x.shape[0], 500), max(x.shape[1], 500), 3)
                frost = np.random.randint(200, 255, size=frost_shape).astype(np.uint8)
                # 添加一些结构，使其看起来像霜冻
                kernel = np.ones((5, 5), np.uint8)
                frost = cv2.erode(frost, kernel, iterations=1)
                frost = cv2.GaussianBlur(frost, (7, 7), 0)
        except Exception as e:
            # 如果读取失败，生成一个简单的随机霜冻纹理
            frost_shape = (max(x.shape[0], 500), max(x.shape[1], 500), 3)
            frost = np.random.randint(200, 255, size=frost_shape).astype(np.uint8)
            # 添加一些结构，使其看起来像霜冻
            kernel = np.ones((5, 5), np.uint8)
            frost = cv2.erode(frost, kernel, iterations=1)
            frost = cv2.GaussianBlur(frost, (7, 7), 0)
        
        frost_shape = frost.shape
        x_shape = np.array(x).shape
        
        # 调整霜冻图像大小以适应图像尺寸
        scaling_factor = 1
        if frost_shape[0] >= x_shape[0] and frost_shape[1] >= x_shape[1]:
            scaling_factor = 1
        elif frost_shape[0] < x_shape[0] and frost_shape[1] >= x_shape[1]:
            scaling_factor = x_shape[0] / frost_shape[0]
        elif frost_shape[0] >= x_shape[0] and frost_shape[1] < x_shape[1]:
            scaling_factor = x_shape[1] / frost_shape[1]
        elif frost_shape[0] < x_shape[0] and frost_shape[1] < x_shape[1]:
            # 如果两个维度都太小，选择更大的缩放因子
            scaling_factor_0 = x_shape[0] / frost_shape[0]
            scaling_factor_1 = x_shape[1] / frost_shape[1]
            scaling_factor = max(scaling_factor_0, scaling_factor_1)
        
        # 添加额外的缩放以确保覆盖
        scaling_factor *= 1.1
        
        # 计算新形状并重新缩放霜冻图像
        new_shape = (int(np.ceil(frost_shape[1] * scaling_factor)),
                     int(np.ceil(frost_shape[0] * scaling_factor)))
        
        # 确保新形状足够大
        if new_shape[0] <= x_shape[1] or new_shape[1] <= x_shape[0]:
            # 如果重新缩放后仍然太小，则增加缩放因子
            extra_scale = max(x_shape[1] / new_shape[0], x_shape[0] / new_shape[1]) * 1.2
            new_shape = (int(np.ceil(new_shape[0] * extra_scale)),
                         int(np.ceil(new_shape[1] * extra_scale)))
        
        # 重新缩放霜冻图像
        try:
            frost_rescaled = cv2.resize(frost, dsize=new_shape, interpolation=cv2.INTER_CUBIC)
        except Exception:
            # 如果调整大小失败，创建一个简单的霜冻纹理
            frost_rescaled = np.ones((max(x_shape[0] + 50, 500), 
                                     max(x_shape[1] + 50, 500), 3), dtype=np.uint8) * 200
            frost_rescaled = cv2.GaussianBlur(frost_rescaled, (15, 15), 0)
        
        # 确保霜冻图像足够大
        if frost_rescaled.shape[0] <= x_shape[0] or frost_rescaled.shape[1] <= x_shape[1]:
            # 创建一个更大的图像并平铺霜冻图案
            bigger_frost = np.zeros((max(x_shape[0] + 50, frost_rescaled.shape[0]), 
                                    max(x_shape[1] + 50, frost_rescaled.shape[1]), 
                                    frost_rescaled.shape[2]), dtype=np.uint8)
            
            # 平铺霜冻图案
            for i in range(0, bigger_frost.shape[0], frost_rescaled.shape[0]):
                for j in range(0, bigger_frost.shape[1], frost_rescaled.shape[1]):
                    i_end = min(i + frost_rescaled.shape[0], bigger_frost.shape[0])
                    j_end = min(j + frost_rescaled.shape[1], bigger_frost.shape[1])
                    bigger_frost[i:i_end, j:j_end] = frost_rescaled[:i_end-i, :j_end-j]
            
            frost_rescaled = bigger_frost
        
        # 随机裁剪
        try:
            x_start = np.random.randint(0, frost_rescaled.shape[0] - x_shape[0])
            y_start = np.random.randint(0, frost_rescaled.shape[1] - x_shape[1])
            
            if len(x_shape) < 3 or x_shape[2] < 3:
                frost_cropped = frost_rescaled[x_start:x_start + x_shape[0],
                                            y_start:y_start + x_shape[1]]
                frost_cropped = rgb2gray(frost_cropped)
            else:
                frost_cropped = frost_rescaled[x_start:x_start + x_shape[0],
                                            y_start:y_start + x_shape[1]][..., [2, 1, 0]]
        except Exception:
            # 如果裁剪失败，创建一个与原图像大小相同的霜冻纹理
            if len(x_shape) < 3 or x_shape[2] < 3:
                frost_cropped = np.random.randint(200, 255, size=(x_shape[0], x_shape[1])).astype(np.uint8)
                frost_cropped = cv2.GaussianBlur(frost_cropped, (7, 7), 0)
            else:
                frost_cropped = np.random.randint(200, 255, size=(x_shape[0], x_shape[1], 3)).astype(np.uint8)
                frost_cropped = cv2.GaussianBlur(frost_cropped, (7, 7), 0)
                frost_cropped = frost_cropped[..., [2, 1, 0]]
        
        # 应用霜冻效果
        return np.clip(frost_params[0] * np.array(x) + frost_params[1] * frost_cropped, 0, 255)
    
    # 选择要添加霜冻效果的帧
    noise_indices = sample_noise_frame(video, ratio)
    device = video.device
    
    # 设置霜冻效果的严重程度(1-5)
    severity = max(1, min(severity, 5))  # 确保severity在1-5范围内
    
    # 处理选定的帧
    for i in noise_indices:
        # 将帧转换为numpy数组
        frame_np = video[i].permute(1, 2, 0).cpu().numpy()  # [H, W, C]
        
        # 应用霜冻效果
        frosted_frame = apply_frost_effect(frame_np, severity, frost_dir)
        
        # 转回PyTorch张量
        if len(frame_np.shape) == 2 or frame_np.shape[2] == 1:  # 灰度
            frame_tensor = torch.from_numpy(frosted_frame).squeeze(-1).unsqueeze(0)
        else:  # RGB或其他多通道
            frame_tensor = torch.from_numpy(frosted_frame).permute(2, 0, 1)
        
        # 更新原视频
        video[i] = frame_tensor.to(device)
    
    return video.to(torch.uint8)



@NoiseRegistry.register("reflect") # 反光/镜面反射
def add_specular_reflection(video: torch.Tensor, ratio: float) -> torch.Tensor:
    video = video.to('cuda')
    _, _, h, w = video.shape
    noise_indices = sample_noise_frame(video, ratio)

    # 预生成Perlin噪声图（转换为Tensor）
    perlin_noise = generate_perlin_noise(h, w)
    perlin_noise = torch.from_numpy(perlin_noise).to(video.device)

    for i in noise_indices:
        frame = video[i].float()  
        # 检测高光区域（阈值调整为0-255范围）
        gray = frame.mean(dim=0)  
        mask = (gray > 100)       
        
        # 高斯模糊
        blurred = F.gaussian_blur(frame.unsqueeze(0), kernel_size=15, sigma=5)[0]
        
        # 高光区域增强
        highlighted = 1.5 * frame + 0.5 * blurred
        highlighted = torch.clamp(highlighted, 0, 255)  
        
        # 合成高光
        frame = torch.where(mask, highlighted, frame)

        # 叠加Perlin噪声
        noise_layer = perlin_noise.to(frame.dtype) 
        frame = 0.9 * frame + 0.1 * noise_layer
        video[i] = torch.clamp(frame, 0, 255) 

    return video.to(torch.uint8)


@NoiseRegistry.register("shadow")  # 阴影
def add_shadow_noise(video: torch.Tensor, ratio: float) -> torch.Tensor:
    video = video.cpu()
    _, _, H, W = video.shape
    
    np_frames = video.permute(0, 2, 3, 1).numpy().astype(np.uint8)  # 转换为 NHWC
    noise_indice = sample_noise_frame(video, ratio)
    noisy_frames = []

    for i, frame in enumerate(np_frames):
        if i in noise_indice:
            # 随机生成椭圆参数
            center = (random.randint(W//4, 3*W//4), random.randint(H//4, 3*H//4))
            axes = (random.randint(W//4, W//2), random.randint(H//4, H//2))
            angle = random.randint(0, 180)  # 椭圆旋转角度
            
            # 创建单通道掩膜并绘制椭圆
            mask = np.zeros((H, W), dtype=np.float32)
            cv2.ellipse(mask, center, axes, angle, 0, 360, color=1, thickness=-1)
            
            # 应用高斯模糊并限制范围
            mask = cv2.GaussianBlur(mask, (51, 51), 0)
            mask = np.clip(mask, 0, 1)
            
            # 应用变暗效果
            frame_float = frame.astype(np.float32)
            darkened = frame_float * (1 - 0.7 * mask[..., np.newaxis])
            darkened = np.clip(darkened, 0, 255).astype(np.uint8)
            noisy_frames.append(darkened)
        else:
            noisy_frames.append(frame)
    
    np_noisy = np.stack(noisy_frames, axis=0)  
    tensor_noisy = torch.from_numpy(np_noisy).permute(0, 3, 1, 2)
    return tensor_noisy.to(torch.uint8) 