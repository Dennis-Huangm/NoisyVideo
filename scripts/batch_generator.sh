#!/bin/bash

OUTPUT_FILE="batch_run.sh"
NUM_GPUS=8
DATASET="MMBench_Video_8frame_nopack"
MODEL="Qwen2.5-VL-3B-Instruct"
JODGER="gpt-4o"

# expected format
# uv run torchrun --nproc-per-node=8 run.py --data MMBench_Video_8frame_nopack --model Qwen2.5-VL-3B-Instruct --judge gpt-4o --noise_name gaussian --ratio 0.9

NOISES=("gaussian" "impulse" "speckle" "poisson" "gaussian_blur" "motion_blur" "defocus_blur" "glass_blur" "zoom_blur" "jpeg_artifact" "random_block" "target_block" "frame_drop" "frame_replace" "frame_repeat" "temporal_jitter" "other_video" "bright_transform" "contrast" "elastic" "color_shift" "flicker" "overexposure" "underexposure" "rainy" "foggy" "snow" "frost" "reflect" "shadow" "random_pixel" "resolution_degrade" "stretch_squish" "edge_sawtooth" "color_quantized")
# RATIOS=("0.3" "0.6" "0.9")
RATIO="0.9"

for NOISE in "${NOISES[@]}"; do
    echo "uv run torchrun --nproc-per-node=$NUM_GPUS run.py --data $DATASET --model $MODEL --judge $JODGER --noise_name $NOISE --ratio $RATIO" >> $OUTPUT_FILE
done

