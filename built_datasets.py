# -*- coding: utf-8 -*-
import os
import time
from tqdm import tqdm
import torch
import torchvision
from video_noise.noise_applier import NoiseRegistry

def save_as_video(tensor: torch.Tensor, save_path: str, fps: int = 30):
    """
    将 (T, C, H, W) 格式的 tensor 以 mpeg4 编码保存为视频 (T, H, W, C)。
    """
    tensor = tensor.permute(0, 2, 3, 1).cpu().numpy()  # (T, H, W, C)
    torchvision.io.write_video(
        filename=save_path,
        video_array=tensor,
        fps=fps,
        video_codec="mpeg4"
    )

# --- 配置路径 ---
video_dir        = 'video'
output_base_dir  = 'outputs_video'
video_extensions = ('.mp4', '.avi', '.mov', '.mkv')

os.makedirs(output_base_dir, exist_ok=True)

# --- 列出所有原始视频（限制前5个）---
video_files = [
    f for f in os.listdir(video_dir)
    if f.lower().endswith(video_extensions)
][:5]
print("待处理视频:", video_files)

# --- 获取所有噪声类型 ---
noises = NoiseRegistry.list_noises()

start_all = time.perf_counter()

# --- 对每个视频，依次施加每种噪声 ---
for video_file in tqdm(video_files, desc="原始视频列表"):
    base_name, _ = os.path.splitext(video_file)
    input_path = os.path.join(video_dir, video_file)
    
    required_noises = []
    for noise in noises:
        noise_dir = os.path.join(output_base_dir, noise)
        output_path = os.path.join(noise_dir, f"{base_name}_{noise}.mp4")
        if not os.path.exists(output_path):
            required_noises.append( (noise, noise_dir, output_path) )
    
    # 如果无需处理任何噪声，跳过当前视频
    if not required_noises:
        print(f"跳过 {video_file}，已存在所有噪声视频")
        continue
    
    # 读取视频 (T, C, H, W)
    try:
        video_frames, _, info = torchvision.io.read_video(
            input_path,
            pts_unit="sec",
            output_format="TCHW",
        )
    except Exception as e:
        print(f"读取视频失败: {video_file}, 错误: {e}")
        continue
    
    print(f"视频 {video_file} 帧数: {video_frames.shape[0]}")
    fps = int(info.get("video_fps", 30))

    for noise, noise_dir, output_path in tqdm(required_noises, desc=f"处理 {video_file}", leave=False):
        os.makedirs(noise_dir, exist_ok=True)  # 确保目录存在
        
        # 施加噪声，第二个参数为强度（0.9 表示默认强度）
        try:
            noisy_video = NoiseRegistry.get_noise(noise)(video_frames.clone(), 0.9)
            save_as_video(noisy_video, output_path, fps=fps)
        except Exception as e:
            print(f"处理失败: 视频={video_file}, 噪声={noise}, 错误: {e}")

end_all = time.perf_counter()
print(f"全部处理完毕，总耗时：{end_all - start_all:.2f} 秒")