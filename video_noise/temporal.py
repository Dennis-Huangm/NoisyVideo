# Denis
# -*- coding: utf-8 -*-
from .noise_applier import NoiseRegistry
from .utils import *
import torchvision
import torch
import numpy as np
import cv2
import tempfile
import os
from typing import Optional, Union
import sys

other_frames, _, info = torchvision.io.read_video(
    'video_noise/sample/other_video.mp4',
    pts_unit="sec",
    output_format="TCHW"
)


@NoiseRegistry.register("frame_drop") # 帧丢失
def add_frame_loss(video: torch.Tensor, ratio):
    drop_indices = sample_noise_frame(video, ratio)
    
    noisy_video = video.clone().detach()
    if len(drop_indices) > 0:
        noisy_video[drop_indices] = 0 
    
    return noisy_video.to(torch.uint8) 


@NoiseRegistry.register("frame_replace") # 帧错序
def add_frame_replacement(video: torch.Tensor, ratio):
    replace_indices = sample_noise_frame(video, ratio)

    if not replace_indices:
        return video
    
    num = max(len(replace_indices), 1)
    selected_frames = video[replace_indices].clone().detach()
    shuffle_idx = torch.randperm(num, device=video.device)

    shuffled_frames = selected_frames[shuffle_idx]  
    noisy_video = video.clone().detach()
    noisy_video[replace_indices] = shuffled_frames
    
    return noisy_video.to(torch.uint8) 


@NoiseRegistry.register("frame_repeat") # 帧重复
def add_frame_repeat(video: torch.Tensor, ratio):
    noisy_video = video.clone().detach()
    repeat_indices = sorted(sample_noise_frame(video, 1 - ratio))
    if not repeat_indices:
        repeat_indices = [0]

    for t in range(len(video)):
        if t not in repeat_indices:
            nearest = min(repeat_indices, key=lambda x: abs(x - t))
            noisy_video[t] = video[nearest]
    return noisy_video.to(torch.uint8) 


@NoiseRegistry.register("temporal_jitter") # 时序抖动
def add_temporal_jitter(video: torch.Tensor, 
                        ratio: float,
                        execution_order: str = "replace_first"):
    noisy_video = video.clone().detach()
    replace_ratio = ratio * random.random()
    drop_ratio = 1 - replace_ratio
    
    if execution_order == "replace_first":
        noisy_video = add_frame_replacement(noisy_video, replace_ratio)
        noisy_video = add_frame_loss(noisy_video, drop_ratio)
    else:
        noisy_video = add_frame_loss(noisy_video, drop_ratio)
        noisy_video = add_frame_replacement(noisy_video, replace_ratio)
    return noisy_video.to(torch.uint8) 


# @NoiseRegistry.register("other_video") # 别的视频的帧
# def add_other_video(video: torch.Tensor, ratio):
#     noise_indice = sample_noise_frame(video, ratio)
#     for i in noise_indice:
#         if i < 32:
#             video[i] = other_frames[i]
#     return video.to(torch.uint8) 


@NoiseRegistry.register("bit_error")
def add_bit_error(video: torch.Tensor, ratio: float, severity: str = "medium") -> torch.Tensor:
    """
    在视频的随机帧中，对随机选定的区域（至少是图片的 1/4 区域）进行条纹化处理。
    
    Args:
        video: 输入视频张量，形状为 [T, C, H, W]
        ratio: 需要处理的帧比例，例如 0.5 表示 50% 的帧被选中
        severity: 条纹化的严重程度，可选 "low"、"medium"、"high"
    
    Returns:
        添加条纹化效果的视频张量
    """
    video_out = video.clone()  # 复制输入视频
    T, C, H, W = video_out.shape  # 提取视频的维度

    # 确定条纹化的强度
    severity_to_stripe_width = {
        "low": 15,      # 条纹宽度较宽
        "medium": 10,   # 条纹宽度适中
        "high": 5       # 条纹宽度较窄
    }
    stripe_width = severity_to_stripe_width.get(severity, 10)

    # 随机采样需要添加条纹化的帧
    noise_indices = sample_noise_frame(video, ratio)

    # 对选中的帧进行条纹化处理
    for i in noise_indices:
        frame = video_out[i]  # 当前帧，形状为 [C, H, W]

        # 随机选定一个区域（至少 1/4 图像大小）
        region_height = np.random.randint(H // 4, H // 2)  # 高度范围：1/4 到 1/2
        region_width = np.random.randint(W // 4, W // 2)   # 宽度范围：1/4 到 1/2
        start_h = np.random.randint(0, H - region_height)  # 随机起始高度
        start_w = np.random.randint(0, W - region_width)   # 随机起始宽度

        # 获取该区域
        region = frame[:, start_h:start_h + region_height, start_w:start_w + region_width]

        # 对区域进行条纹化处理
        for col in range(0, region_width, stripe_width):
            # 随机选择一列的像素值作为条纹
            random_col = np.random.randint(region_width)
            region[:, :, col:col + stripe_width] = region[:, :, random_col:random_col + 1]

        # 将条纹化后的区域写回帧
        frame[:, start_h:start_h + region_height, start_w:start_w + region_width] = region

    return video_out.to(torch.uint8) 


# 首先定义一个自动安装并获取FFmpeg的函数
def setup_ffmpeg():
    """安装并设置FFmpeg，返回可执行文件路径"""
    try:
        # 尝试安装imageio-ffmpeg包，这个包自带FFmpeg二进制文件
        import subprocess
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'imageio-ffmpeg'], 
                      stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        
        # 导入以使用其FFmpeg可执行文件
        from imageio_ffmpeg import get_ffmpeg_exe
        ffmpeg_path = get_ffmpeg_exe()
        print(f"Using FFmpeg from imageio-ffmpeg: {ffmpeg_path}")
        return ffmpeg_path
    except Exception as e:
        print(f"Failed to install imageio-ffmpeg: {e}")
        
        # 备选方案：尝试安装ffmpeg-python
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', 'ffmpeg-python'], 
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            print("Installed ffmpeg-python")
            
            # 尝试使用系统FFmpeg
            try:
                process = subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if process.returncode == 0:
                    print("Using system FFmpeg")
                    return 'ffmpeg'  # 使用系统PATH中的ffmpeg
            except:
                pass
            
            raise RuntimeError("FFmpeg executable not found after installing ffmpeg-python")
        except Exception as e2:
            print(f"Failed to install ffmpeg-python: {e2}")
            raise RuntimeError("Could not set up FFmpeg. Please install it manually.")



@NoiseRegistry.register("h265_artifacts")
def add_h265_compression_artifacts(
    video: torch.Tensor, 
    severity: Union[str, float, int] = "extreme",
    input_path: Optional[str] = None
) -> torch.Tensor:
    """
    使用ffmpeg-python添加极端H.265压缩伪影，适配FFmpeg 4.2.2版本
    
    Args:
        video: 输入视频张量，形状为[T, C, H, W]
        severity: 压缩严重程度，默认"extreme"，或1-5的数值（5为最严重）
        input_path: 可选，原始MP4视频路径。如果提供，将直接使用这个文件进行转码
    
    Returns:
        应用了H.265压缩效果的视频张量
    """
    import ffmpeg
    from imageio_ffmpeg import get_ffmpeg_exe
    
    # 获取FFmpeg可执行文件路径
    ffmpeg_path = get_ffmpeg_exe()
    print(f"Using FFmpeg from imageio-ffmpeg: {ffmpeg_path}")
    
    # 获取FFmpeg版本信息
    try:
        import subprocess
        result = subprocess.run([ffmpeg_path, '-version'], capture_output=True, text=True)
        print(f"FFmpeg version info: {result.stdout.splitlines()[0]}")
    except Exception as e:
        print(f"Failed to get FFmpeg version: {e}")
    
    # 设置压缩参数
    severity_configs = {
        "low": {"crf": 32, "bitrate": "400k"},
        "medium": {"crf": 38, "bitrate": "200k"},
        "high": {"crf": 45, "bitrate": "100k"},
        "extreme": {"crf": 51, "bitrate": "50k"}  # 最大CRF值和极低比特率
    }
    
    if isinstance(severity, str):
        config = severity_configs.get(severity, severity_configs["extreme"])
    else:
        # 数值1-5映射到不同级别
        severity_level = min(5, max(1, int(float(severity))))
        if severity_level <= 2:
            config = severity_configs["low"]
        elif severity_level == 3:
            config = severity_configs["medium"]
        elif severity_level == 4:
            config = severity_configs["high"]
        else:  # 5
            config = severity_configs["extreme"]
    
    crf = config["crf"]
    bitrate = config["bitrate"]
    
    # 获取视频尺寸信息
    num_frames, channels, height, width = video.shape
    
    # 创建临时文件
    temp_files = []
    
    try:
        # 创建输出文件
        output_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        output_file.close()
        temp_files.append(output_file.name)
        
        # 如果没有提供原始MP4路径，则创建一个
        if input_path is None or not os.path.exists(input_path):
            input_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
            input_file.close()
            temp_files.append(input_file.name)
            
            # 转换为numpy并保存为临时视频文件
            video_np = video.permute(0, 2, 3, 1).cpu().numpy()  # [T, H, W, C]
            
            fps = 30
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(
                input_file.name, 
                fourcc, 
                fps, 
                (width, height), 
                isColor=(channels == 3)
            )
            
            for i in range(num_frames):
                if channels == 3:  # RGB转BGR
                    frame = cv2.cvtColor(video_np[i], cv2.COLOR_RGB2BGR)
                else:  # 灰度
                    frame = video_np[i, :, :, 0]  # 提取单通道
                writer.write(frame)
            
            writer.release()
            
            source_path = input_file.name
        else:
            # 使用提供的MP4文件
            source_path = input_path
        
        # 创建一个中间文件用于第一次压缩
        intermediate_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        intermediate_file.close()
        temp_files.append(intermediate_file.name)
        
        # 第一次压缩：故意降低分辨率
        # 如果宽高太小，会导致压缩失败，确保至少为16px
        scaled_width = max(16, width // 4)
        scaled_height = max(16, height // 4)
        
        # 适配FFmpeg 4.2.2的H.265编码
        try:
            print(f"Attempting first pass with H.265 encoding: CRF={crf}, bitrate={bitrate}")
            
            # 使用ffmpeg-python但避免使用x265_params参数
            (
                ffmpeg
                .input(source_path)
                .output(
                    intermediate_file.name,
                    vcodec='libx265',
                    vf=f'scale={scaled_width}:{scaled_height}',  # 先缩小
                    crf=crf,
                    preset='ultrafast',  # 使用最快预设，通常质量较差
                    video_bitrate=bitrate,
                    maxrate=bitrate,
                    bufsize=f"{int(bitrate[:-1]) * 2}k",
                    pix_fmt='yuv420p',
                    g=300,  # 大关键帧间隔
                    keyint_min=30,
                    sc_threshold=0  # 禁用场景变化检测
                    # 不使用x265_params参数
                )
                .global_args('-y')
                .run(cmd=ffmpeg_path, capture_stdout=True, capture_stderr=True)
            )
            
            # 第二次压缩：再放大回原始尺寸，二次压缩
            print(f"Attempting second pass with H.265 encoding")
            (
                ffmpeg
                .input(intermediate_file.name)
                .output(
                    output_file.name,
                    vcodec='libx265',  
                    vf=f'scale={width}:{height}',  # 放大回原始尺寸
                    crf=crf,
                    preset='ultrafast',
                    video_bitrate=f"{int(bitrate[:-1]) // 2}k",  # 更低的比特率
                    maxrate=f"{int(bitrate[:-1]) // 2}k",
                    bufsize=f"{int(bitrate[:-1])}k",
                    pix_fmt='yuv420p',
                    g=300,
                    keyint_min=30,
                    sc_threshold=0
                    # 不使用x265_params参数
                )
                .global_args('-y')
                .run(cmd=ffmpeg_path, capture_stdout=True, capture_stderr=True)
            )
            
            print("H.265 encoding successful!")
            
        except ffmpeg.Error as e:
            print(f"H.265 encoding failed: {e.stderr.decode() if hasattr(e, 'stderr') and e.stderr else str(e)}")
            print("Falling back to H.264 encoder")
            
            # 回退到H.264编码器
            try:
                # 第一次压缩
                (
                    ffmpeg
                    .input(source_path)
                    .output(
                        intermediate_file.name,
                        vcodec='libx264',
                        vf=f'scale={scaled_width}:{scaled_height}',
                        crf=crf,
                        preset='ultrafast',
                        video_bitrate=bitrate,
                        maxrate=bitrate,
                        bufsize=f"{int(bitrate[:-1]) * 2}k",
                        pix_fmt='yuv420p',
                        g=300,
                        keyint_min=30,
                        sc_threshold=0
                    )
                    .global_args('-y')
                    .run(cmd=ffmpeg_path, capture_stdout=True, capture_stderr=True)
                )
                
                # 二次压缩
                (
                    ffmpeg
                    .input(intermediate_file.name)
                    .output(
                        output_file.name,
                        vcodec='libx264',
                        vf=f'scale={width}:{height}',
                        crf=crf,
                        preset='ultrafast',
                        video_bitrate=f"{int(bitrate[:-1]) // 2}k",
                        maxrate=f"{int(bitrate[:-1]) // 2}k",
                        bufsize=f"{int(bitrate[:-1])}k",
                        pix_fmt='yuv420p',
                        g=300,
                        keyint_min=30,
                        sc_threshold=0
                    )
                    .global_args('-y')
                    .run(cmd=ffmpeg_path, capture_stdout=True, capture_stderr=True)
                )
                
                print("H.264 encoding successful!")
                
            except ffmpeg.Error as e2:
                print(f"H.264 encoding also failed: {e2.stderr.decode() if hasattr(e2, 'stderr') and e2.stderr else str(e2)}")
                
                # 作为最后的备选方案，尝试最基本的命令行
                try:
                    import subprocess
                    print("Trying direct subprocess command as last resort")
                    
                    # 最基本的命令
                    subprocess.run([
                        ffmpeg_path,
                        '-i', source_path,
                        '-c:v', 'libx264',
                        '-crf', str(crf),
                        '-preset', 'ultrafast',
                        '-y', output_file.name
                    ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    
                    print("Basic encoding successful!")
                    
                except subprocess.CalledProcessError as e3:
                    print(f"All encoding attempts failed: {e3}")
                    raise RuntimeError("Could not compress video with any method")
        
        # 确认输出文件存在且有效
        if not os.path.exists(output_file.name) or os.path.getsize(output_file.name) == 0:
            raise RuntimeError("FFmpeg produced an empty or non-existent output file")
        
        # 读取压缩后的视频
        compressed_frames = []
        cap = cv2.VideoCapture(output_file.name)
        
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open compressed video file: {output_file.name}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 确保颜色空间和尺寸正确
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))
                
            if channels == 3:  # BGR转RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:  # 灰度
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = np.expand_dims(frame, axis=2)
                
            compressed_frames.append(frame)
        
        cap.release()
        
        # 处理帧数不匹配的情况
        if not compressed_frames:
            raise RuntimeError("No frames were read from the compressed video")
        
        print(f"Original frames: {num_frames}, Compressed frames: {len(compressed_frames)}")
        
        if len(compressed_frames) < num_frames:
            # 压缩后帧数减少，重复最后一帧
            last_frame = compressed_frames[-1]
            compressed_frames.extend([last_frame] * (num_frames - len(compressed_frames)))
        elif len(compressed_frames) > num_frames:
            # 截断多余帧
            compressed_frames = compressed_frames[:num_frames]
        
        # 转换回PyTorch张量
        result_np = np.stack(compressed_frames)
        result_tensor = torch.from_numpy(result_np).permute(0, 3, 1, 2)  # [T, C, H, W]
        result_tensor.to(video.dtype).to(video.device)
        
        return result_tensor.to(torch.uint8) 
    
    finally:
        # 清理临时文件
        for file_path in temp_files:
            if os.path.exists(file_path):
                try:
                    os.unlink(file_path)
                except Exception as e:
                    print(f"Warning: Failed to delete temporary file {file_path}: {e}")