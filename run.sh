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
    stretch_squish edge_sawtooth color_quantized # origin
)

# 定义模型列表
models=(
    # Video-LLaVA-7B-HF
    # VideoChat2-HD
    # Chat-UniVi-7B
    # Chat-UniVi-7B-v1.5
    # LLaMA-VID-7B
    # Video-ChatGPT
    # PLLaVA-7B
    # PLLaVA-13B
    Qwen2.5-VL-3B-Instruct
)

# 设置 GPU 数量（根据实际情况修改）
NUM_GPUS=2

for model in "${models[@]}"; do
    # 根据 model 决定 ratio 列表
    if [[ "$model" == "Qwen2.5-VL-3B-Instruct" ]]; then
        ratios=(0.3 0.6 0.9)
    else
        ratios=(0.9)
    fi

    for noise in "${noise_names[@]}"; do
        # 定义所有 ratio 共用的输出目录
        output_dir="outputs/${model}/${noise}"
        mkdir -p "$output_dir"

        for ratio in "${ratios[@]}"; do
            # 只跳过已存在指定 ratio 的 JSON 文件
            # 假设 JSON 文件名里含有 "_ratio${ratio}_" 这个标识
            if find "$output_dir" -maxdepth 1 -type f -name "*_${ratio}_*.json" | read; then
                echo "[$(date)] 跳过：model=$model, noise=$noise, ratio=$ratio（已存在）"
                continue
            fi

            echo "------------------------------------------------------"
            echo "[$(date)] 开始：model=$model, noise=$noise, ratio=$ratio"

            # 如果是 origin，不传 noise 参数
            if [[ "$noise" == "origin" ]]; then
                noise_arg=""
            else
                noise_arg="--noise_name $noise"
            fi

            # 执行核心命令，输出文件默认写入 output_dir
            torchrun \
                --nproc-per-node="$NUM_GPUS" \
                run.py \
                --data MMBench_Video_8frame_nopack \
                --model "$model" \
                --judge gpt-4o \
                $noise_arg \
                --ratio "$ratio" \

            echo "[$(date)] 完成：model=$model, noise=$noise, ratio=$ratio"
            echo "等待 60 秒..."
            sleep 60
        done
    done
done

echo "所有任务执行完成！"
