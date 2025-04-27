#!/bin/bash

# 定义噪声类型数组（共36种）
noise_names=(
    gaussian impulse speckle poisson
    gaussian_blur motion_blur defocus_blur glass_blur zoom_blur
    jpeg_artifact random_block target_block
    frame_drop frame_replace frame_repeat temporal_jitter bit_error h265_artifacts
    bright_transform contrast elastic color_shift flicker
    overexposure underexposure rainy foggy snow frost
    reflect shadow rolling_shutter resolution_degrade
    stretch_squish edge_sawtooth color_quantized origin
)

# 定义模型数组（共9个）
models=(
    Video-LLaVA-7B-HF
    VideoChat2-HD
    Chat-UniVi-7B
    Chat-UniVi-7B-v1.5
    LLaMA-VID-7B
    Video-ChatGPT
    PLLaVA-7B
    # PLLaVA-13B
)

# 设置GPU数量（根据实际情况修改）
NUM_GPUS=2

# 循环执行所有组合
for model in "${models[@]}"; do
    for noise in "${noise_names[@]}"; do
        output_dir="outputs/${model}/${noise}"
        
        # 检查是否已存在JSON文件
        if [[ -d "${output_dir}" ]] && find "${output_dir}" -maxdepth 1 -type f -name "*.json" | read; then
            echo "------------------------------------------------------"
            echo "[$(date)] 跳过组合：噪声类型->${noise} | 模型->${model}，输出文件已存在"
            continue
        fi

        echo "------------------------------------------------------"
        echo "[$(date)] 开始执行组合：噪声类型->${noise} | 模型->${model}"

        if [ "${noise}" = "Origin" ]; then
            noise_arg=""
        else
            noise_arg="--noise_name ${noise}"
        fi
        
        # 执行核心命令
        torchrun \
            --nproc-per-node=${NUM_GPUS} \
            run.py \
            --data MMBench_Video_8frame_nopack \
            --model "${model}" \
            --judge gpt-4o \
            ${noise_arg} \
            --ratio 0.9
            
        echo "[$(date)] 执行完成，等待60秒..."
        sleep 1m
    done
done

echo "所有任务执行完成！"
