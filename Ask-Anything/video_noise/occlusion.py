# Denis
# -*- coding: utf-8 -*-
from .noise_applier import NoiseRegistry
from .utils import *
from ultralytics import YOLO

Model = YOLO('yolov8n.pt')


@NoiseRegistry.register("random_block") # 随机遮挡
def add_random_block(video: torch.Tensor, ratio, min_ratio=0.3, max_ratio=0.7):
    noise_indice = sample_noise_frame(video, ratio)
    _, _, H, W = video.shape
    device = video.device

    for i in noise_indice:
        h = int(H * torch.empty(1).uniform_(min_ratio, max_ratio).item())
        w = int(W * torch.empty(1).uniform_(min_ratio, max_ratio).item())
        h, w = max(h, 1), max(w, 1)  

        y = torch.randint(0, H - h + 1, (1,), device=device).item()
        x = torch.randint(0, W - w + 1, (1,), device=device).item()
        video[i][:, y:y+h, x:x+w] = 0
    return video.to(torch.uint8)


@NoiseRegistry.register("target_block") # 主要目标遮挡
def add_target_block(video: torch.Tensor, ratio):
    noise_indice = sample_noise_frame(video, ratio)
    _, _, H, W = video.shape
    Model.to(video.device)

    for i in noise_indice:
        inputs, scale, (top_pad, left_pad) = preprocess(video[i])
        results = Model(inputs.to(video.device), verbose=False)
        for result in results:
            boxes = result.boxes.xyxy.cpu().clone() 
            boxes[:, [0, 2]] -= left_pad
            boxes[:, [1, 3]] -= top_pad
            boxes /= scale
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, W)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, H)
            
            if boxes.numel() != 0:
                x1, y1, x2, y2 = boxes.int()[0].tolist()
                video[i][:, y1:y2, x1:x2] = 0
    return video.to(torch.uint8)


