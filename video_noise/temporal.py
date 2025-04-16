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


