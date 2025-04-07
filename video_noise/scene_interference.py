# Denis
# -*- coding: utf-8 -*-
from .noise_applier import NoiseRegistry
from .utils import *
from torchvision.transforms import functional as F
import cv2
from torchvision.io import read_image


@NoiseRegistry.register("rainy") # 雨
def add_rain(video: torch.Tensor, ratio: float) -> torch.Tensor:
    noise_indices = sample_noise_frame(video, ratio)
    device = video.device

    rain_layer = read_image('video_noise/sample/template_rain.png').float().to(device)
    alpha, belta = 0.6, 1.0
    rain_layer = F.resize(rain_layer, [video.shape[2], video.shape[3]])

    for i in noise_indices:
        synthetic = alpha * rain_layer + belta * video[i]
        video[i] = torch.clamp(synthetic, 0, 255).to(video.dtype)
    return video


@NoiseRegistry.register("foggy") # 雾
def add_rain(video: torch.Tensor, ratio: float) -> torch.Tensor:
    noise_indices = sample_noise_frame(video, ratio)
    device = video.device

    rain_layer = read_image('video_noise/sample/template_fog.png').float().to(device)
    alpha, belta = 0.7, 0.5
    rain_layer = F.resize(rain_layer, [video.shape[2], video.shape[3]])

    for i in noise_indices:
        synthetic = alpha * rain_layer + belta * video[i]
        video[i] = torch.clamp(synthetic, 0, 255).to(video.dtype)
    return video


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
        frame = torch.clamp(frame, 0, 255) 
        
        video[i] = frame.to(video.dtype) 

    return video


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
    tensor_noisy = torch.from_numpy(np_noisy).permute(0, 3, 1, 2).to(torch.uint8) 
    return tensor_noisy