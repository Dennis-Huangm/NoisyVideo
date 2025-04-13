#!/bin/bash

# 定义噪声类型数组（共35种）
noise_names=(
    gaussian impulse speckle poisson
    gaussian_blur motion_blur defocus_blur glass_blur zoom_blur
    jpeg_artifact random_block target_block
    frame_drop frame_replace frame_repeat temporal_jitter other_video
    bright_transform contrast elastic color_shift flicker
    overexposure underexposure rainy foggy snow frost
    reflect shadow random_pixel resolution_degrade
    stretch_squish edge_sawtooth color_quantized
)

# 定义模型数组（共9个）
models=(
    Video-LLaVA-7B-HF
    VideoChat2-HD
    Chat-UniVi-7B
    Chat-UniVi-7B-v1.5
    # LLaMA-VID-7B
    Video-ChatGPT
    PLLaVA-7B
    PLLaVA-13B
)

# 设置GPU数量（根据实际情况修改）
NUM_GPUS=2

# 循环执行所有组合
for noise in "${noise_names[@]}"; do
    for model in "${models[@]}"; do
        echo "------------------------------------------------------"
        echo "[$(date)] 开始执行组合：噪声类型->$noise | 模型->$model"
        
        # 执行核心命令
        torchrun \
            --nproc-per-node=$NUM_GPUS \
            run.py \
            --data MMBench_Video_8frame_nopack \
            --model "$model" \
            --judge gpt-4o \
            --noise_name "$noise" \
            --ratio 0.9
            
        echo "[$(date)] 执行完成，等待60秒..."
        sleep 1m
    done
done

echo "所有任务执行完成！"