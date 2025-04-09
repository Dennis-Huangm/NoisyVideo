# NRVL(Towards Noise-Robust Video Understanding with LLMs)

## Installation
python==3.9
```bash
conda activate --name ${your_env_name} python==3.9
cd VLMEvalKit
pip install -e .
pip install flash-attn --no-build-isolation
```
## Quick Start
 ### 1. 运行开源本地模型
```bash
torchrun \
	--nproc-per-node=${NUM_GPUS}  \
	--data MMBench_Video_${NUM_FRAMES}frame_nopack \
	--model ${MODEL} \
	--judge gpt-4o \
	--noise_name ${NOISE_TYPE} \
	--ratio ${NOISE_PROPORTION} \
	run.py
```
 -  --data是选择的数据集，具体可选的数据集位于vlmeval/dataset/video_dataset_config.py中，目前只考虑运行MMBench_Video_8frame_nopack与MMBench_Video_16frame_nopack，优先8帧。
 - --model是我们用来推理的模型，可选的模型位于vlmeval/config.py中，我们应该是先尝试Qwen2.5-VL-3B-Instruct。
 - --judge是用来评估结果的模型，我们这里统一采用gpt-4o。
 - --noise_name是我们使用的噪音，共有35种噪音：['gaussian', 'impulse', 'speckle', 'poisson', 'gaussian_blur', 'motion_blur', 'defocus_blur', 'glass_blur', 'zoom_blur', 'jpeg_artifact', 'random_block', 'target_block', 'frame_drop', 'frame_replace', 'frame_repeat', 'temporal_jitter', 'other_video', 'bright_transform', 'contrast', 'elastic', 'color_shift', 'flicker', 'overexposure', 'underexposure', 'rainy', 'foggy', 'snow', 'frost', 'reflect', 'shadow', 'random_pixel', 'resolution_degrade', 'stretch_squish', 'edge_sawtooth', 'color_quantized']。
 - --ratio是噪音帧在整个视频中的占比，有三个比例：[0.3, 0.6, 0.9]。优先测试0.9吧……

 ### 2. 运行闭源api模型
```bash
python run.py --data MMBench_Video_8frame_nopack --model GPT4V --judge gpt-4o 
```
配置与上述一样，具体噪音在后面添加即可，只不过api调用禁止多进程。
### Run
执行run.py后会自动创建一个文件夹，文件夹的命名可以基于运行的日期/分钟/秒。这个项目有一点做得很好，就是他会把你已经推理出的结果存储在一个pkl文件中，这样你再次运行就不需要重新跑了（这个主要好处是在跑开销极高的api时，万一程序在快跑完时断了，那就完犊子，赔了时间又赔了钱）。

## Result
运行完成后最重要的文件应该是{model_name}\_{dataset_name}\_{noise_name}_{ratio}\_{judge}_rating.json。主要内容如下：
```
"coarse_all": {
"CP": "1.98",
"FP-S": "1.67",
"FP-C": "1.57",
"HL": "2.40",
"LR": "1.36",
"AR": "1.94",
"RR": "1.75",
"CSR": "1.91",
"TR": "1.63",
"Perception": "1.74",
"Reasoning": "1.72",
"Overall": "1.74"
},
"coarse_valid": {
"CP": "1.98",
"FP-S": "1.67",
"FP-C": "1.57",
"HL": "2.40",
"LR": "1.36",
"AR": "1.94",
"RR": "1.75",
"CSR": "1.91",
"TR": "1.63",
"Perception": "1.74",
"Reasoning": "1.72",
"Overall": "1.74"
},
```
其中valid相比all的结果除外了得分为-1（即不正常）的样本，正常来说是不会出现-1的得分的，但我也确实遇到过，可能是api出了点问题，但更多时候valid和all的得分是相同的。Overall是我们主要参考的全局得分。
