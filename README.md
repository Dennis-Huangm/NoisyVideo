![输入图片说明](docs/benchmark.png)

# NoisyVideo

**NoisyVideo** is a toolkit for assessing Video-LLM performance on **question answering tasks** under various **noise conditons** based on `VLMEvalKit`, which is an **open-source evaluation toolkit** of **large vision-language models**. Our toolkit encompasses **36 noise types** in 8 categories, and **9 question types** to comprehensively evaluate the robustness of Video-LLMs. By using the toolkit, we can evaluate the performance of state-of-the-art Video-LLMs for the initial systematic evaluation from multiple perspectives.

## `video_noise` Directory Structure

| File                   | Functions                                                                                           | Distortion Category       |
|------------------------|-----------------------------------------------------------------------------------------------------|--------------------------|
| `visual_quality.py`           | `gaussian`, `impulse`, `speckle`, `poisson`                                 | Quality Noise            |
| `temporal.py`       | `frame_drop`, `frame_replace`, `frame_repeat`, `temporal_jitter`                                    | Temporal Distortion      |
| `blur.py`          | `gaussian_blur`, `motion_blur`, `defocus_blur`, `glass_blur`, `zoom_blur`                          | Blurring Effects         |
| `nature.py`    | `bright_transform`, `contrast`, `color_shift`, `flicker`, `overexposure`, `underexposure`| Lighting & Color         |
| `scene_interference.py`| `rainy`, `foggy`, `snow`, `frost`, `reflect`, `shadow`                                             | Scene Interference       |
| `digital_process.py`        | `rolling_shutter`, `resolution_degrade`, `stretch_squish`, `edge_sawtooth`, `color_quantized`, `elastic` | Digital Artifacts   |
| `occlusion.py`         | `random_block`, `target_block`                                                                     | Occlusion                |
| `compression.py`       | `jpeg_artifact`, `bit_error`, `h265_artifacts`                                                      | Compression Artifacts    |

- Each Python file defines a **modular family of video distortions**.
- All functions are designed for **plug-and-play use** and easy extension.

<details>
<summary>Details Information of Visual Noise</summary>


We introduce a wide range of noise types that reflect real-world situations in video data to evaluate the robustness of Video-LLMs. From the literature, we identify **36 types of different noise** due to capturing, processing, and saving.
To better analyze the impacts of different noises, we further categorize them into 8 groups by their characteristics. Namely, they are distinct noises related to **quality**, **temporality**, **blurring**, **lighting/color**, **scene interference**, **digitality**, **occlusion**, and **compression**.
![输入图片说明](docs/noise.png)
Noise implementations are stored in the `video_noise` directory, organized into individual files by noise category.
|Noise types| Specific Noise |
|:--:|--|
| **Quality** <br>(4 types) |                                                                                **Gaussian**: *Gaussian white noise exhibits normal amplitude distribution and uniform spectral energy.*<br> **Impulse**: *brief, random spikes of noise that create sudden bright or dark pixels (salt-and-pepper effect).*<br> **Speckle**: *granular, multiplicative noise that creates a grainy texture by causing small intensity variations.* <br> **Poisson**: *signal-dependent noise from random photon arrival events.* <br>|
|**Temporality** <br>(4 types)|**Frame drop**: *random removal of entire frames from a video sequence.* <br>        **Frame replace**: *misordering of frames within a video sequence, causing temporal playback jumps.* <br>   **Frame repeat**: *duplication of frames within a video sequence, causing stuttering due to repeated frames.*<br> **Temporal jitter**: *combination of frame drops and frame misordering, causing uneven frame intervals, skipped or out-of-order frames, and jittery playback.*|
|**Blurring** <br>(5 types)|**Gaussian blur**: *smoothing distortion produced by convolving selected frames with a Gaussian kernel.* <br>**Motion blur**: *directional smearing of moving objects caused by camera or subject motion during exposure.* <br> **Defocus blur**: *optical softening from being out of the focal plane, producing uniform blur and bokeh (circle-of-confusion) around objects.* <br> **Glass blur**: *localized refractive distortion simulating viewing through textured or frosted glass, randomly displacing pixels within small neighborhoods to produce blurred and warped effects.* <br>**Zoom blur**: *blur effect caused by scaling the image (zooming in or out), stretching details radially and reducing sharpness.*| 
|**Lighting/Color**<br>(6 types)|**Bright transform**: *adjust video brightness in the HSV color space by scaling the V (value) channel to increase or decrease overall luminance.* <br> **Contrast transform**: *adjust contrast by scaling pixel values around a pivot.* <br> **Color shift**: *apply random additive or multiplicative shifts to each color channel (e.g., R, G, B) of selected frames, causing hue, saturation, and overall color balance distortions.* <br> **Flicker**: *random temporal variations in frame luminance, causing rapid brightness fluctuations that produce a trembling or flickering appearance.* <br>**Overexposure**: *clipping of pixel values to their maximum due to excessive luminance, resulting in washed-out highlights and loss of detail.* <br>**Underexposure**: *insufficient luminance from low exposure or gain, causing pixel values to cluster near zero, deep shadows, and loss of detail.*|
|**Scene interference**<br>(6 types)|**Rainy**: *add rain effects to video by generating fractal-based raindrop streaks and splashes, then blending them into selected frames with varying intensity and motion blur to simulate realistic rainfall.*<br> **Foggy**: *overlay a plasma-fractal noise–based fog mask onto selected frames, blending with adjustable density, falloff, and blur to simulate realistic atmospheric haze.* <br> **Snow**: *overlay simulated falling snowflakes onto video frames by generating particle effects to mimic realistic snowfall.*<br> **Frost**: *overlay semi-transparent ice-crystal (frost) textures, blending in subtle specular highlights and light scattering to mimic frozen surfaces.*<br> **Reflect**: *overlay mirror-like reflections modulated by Perlin-noise–generated distortion maps, blending specular highlights and warped environment details to simulate uneven reflective surfaces.* <br> **Shadow**: *overlay dark masks or gradients onto frames—adjusting region shape, opacity, and position—to simulate object shadows effects.*|
|**Digitality**<br>(6 types)|**Rolling shutter**: *simulate the rolling shutter by reading selected frame’s rows (or columns) sequentially with a line-by-line time offset, causing uniform skew, wobble, and temporal distortion across the selected frames.* <br>**Resolution degrade**: *reduce frame resolution by downsampling, causing blocky artifacts and blurred details due to loss of high-frequency information.*<br>**Stretch squish**: *scale frames horizontally or vertically, stretching or compressing pixel dimensions to modify aspect ratio and introduce geometric distortion.*<br>**Edge sawtooth**: *apply periodic, asymmetric “sawtooth”–shaped distortions along detected edges by shifting pixel positions or intensities in a linear ramp pattern, creating jagged boundary artifacts.*<br> **Color quantized**: *reduce the number of distinct colors by mapping pixel values to a limited palette, causing posterization and visible banding artifacts.*<br>**Elastic**: *apply smooth, random displacement fields to each frame—warping pixels in an elastic manner to simulate stretchy, fluid-like distortions.*|
|**Occlusion**<br>(2 types)|**Random block**: *randomly select rectangular regions in frames and replace them with black blocks.*<br> **Target block**: *detect primary objects with YOLO and overlay occluding blocks on their bounding boxes, masking the target regions in selected frames.*<br>|
|**Compression**<br>(3 types)|**JPEG artifact**: *lossy compression artifacts from JPEG encoding, characterized by ringing halos around sharp edges, and subtle color banding or blur of fine details.*<br>**Bit error**: *choose a random region of the frame and simulate bit-level corruption to produce stripe artifacts.*<br>**H265 artifacts**: *compression artifacts from H.265/HEVC encoding, characterized by blockiness at CTU boundaries, quantization noise, ringing halos around edges, and blurring in high-detail regions.*|


</details>

# Installation
```
conda create --name ${your_env_name} python==3.9
cd VLMEvalKit
pip install -e .
pip install flash-attn --no-build-isolation
```
Some models cannot be directly invoked via the `transformers` library and require manual installation from their source code. For ​**​Chat-UniVi​**​, ​**​LLaMA-VID​**​, and ​**​LLaVA​**​, you must `cd their_source_code_directory` and run `pip install -e .` to install them. Note that **​LLaMA-VID-7B​** may not be able to run under certain transformer versions, we recommend to use `transformers==4.31.0` while evaluating.

# Data 
> **No extra data download required.**  
> Datasets are **automatically downloaded from Hugging Face** during evaluation.
> 
> <b><mark>Note:</mark></b>  
> We do **not** provide a pre-generated noisy dataset for direct download because the full dataset is extremely large (several terabytes).  
> Instead, we **apply noise dynamically during evaluation** to save storage space.  
> To ensure reproducibility, we set a fixed random seed in the code, so the noise generation is consistent across runs.

# Test the Noise code only (optional)

If you only want to test our noise code, you need to download the following data, or you can use your own videos.

- MMBench-Video dataset: [https://huggingface.co/datasets/opencompass/MMBench-Video](https://huggingface.co/datasets/opencompass/MMBench-Video)

Make sure you have completed the installation, you can use the following demo to evaluate the actual effects of each noise type:
```python
from video_noise.noise_applier import NoiseRegistry
from video_noise.utils import save_image

video_path = 'video_noise/sample/demo.mp4'
video_frames, _, info = torchvision.io.read_video(video_path, pts_unit="sec", output_format="TCHW")

print(video_frames.shape)
print(NoiseRegistry.list_noises())  # print all the video types
video = NoiseRegistry.get_noise('gaussian')(video_frames.clone(), 1)
save_image(video[0], 'sample.png')
```
We also provide demo data with noise already added for your reference, **the following link only cover demo noisy videos**:

- Visual Noise Demo: [https://www.kaggle.com/datasets/minruihuang/visual-noise-demo](https://www.kaggle.com/datasets/minruihuang/visual-noise-demo).

# QuickStart for benchmark
Before initiating the evaluation, please configure the environment according to the specified procedures. 
Once configured, you may utilize `run.py` and `evaluation.py` to perform a comprehensive and multiple perspectives assessment of the target Video-LLM.

<details>
<summary><strong>Step 0: Setup API keys</strong></summary>
To use API models (e.g., GPT-4o, Gemini-Pro-V) for inference, or to utilize the LLM API as an evaluator or selector-extractor, you must first configure your API key. We recommend utilizing the OpenAI-compatible API schema to access all Video-LLMs.

 - ​ If you need to use the API, enter your key in the corresponding key field. The API keys will be automatically loaded during inference and evaluation. You can place the required API keys in the `$VideoNoise/.env` file or set them directly as environment variables. If you choose to create a `.env` file, its contents should look like this:
 ```
# The .env file, place it under $VideoNoise
# API Keys of Proprietary VLMs
# OpenAI API
OPENAI_API_KEY=
OPENAI_API_BASE=
# You can also set a proxy for calling api models during the evaluation stage
EVAL_PROXY=
```
</details>

<details>
<summary><strong>Step 1: Prediction</strong></summary>

We use `run.py` to get the prediction under diverse visual noise and the basic **GPT score** judged by conventional LLMs.

Our toolkit supports the evaluation of **any Video-LLM**. Here, we demonstrate the evaluation process using **Qwen2.5-VL-3B-Instruct** and **Gaussian Noise** as an example.


**Argrments**

 - `--data`: Set the dataset names. In our benchmark, we test different Video-LLMs by applying noise to **MMBench-Video**.
 - `--model`: Set the Video-LLM names currently supported. 
 - `--judge`: Set the API model names as the **judge**. We adopt gpt-4o in our benchmark.
 - `--ratio`: Set the ratio of noisy frames in the input video.
 - `--noise_name`: Set the noise names you want to evluate, you can find all 36 supported noise types in the following code:
```python
  from video_noise.noise_applier import NoiseRegistry
  print(NoiseRegistry.list_noises())
```
**Command for evaluating a local model**
```shell
torchrun \
	--nproc-per-node=${NUM_GPUS}  \
	run.py \
	--data MMBench_Video_${NUM_FRAMES}frame_nopack \
	--model ${MODEL} \
	--judge gpt-4o \
	--noise_name ${NOISE_TYPE} \
	--ratio ${NOISE_PROPORTION}
```
Example:
```
torchrun --nproc-per-node=2 run.py --data MMBench_Video_8frame_nopack --model Qwen2.5-VL-3B-Instruct --judge gpt-4o --noise_name gaussian --ratio 0.9
```
**Command for evaluating an API model**
Example:
```
python run.py --data MMBench_Video_8frame_nopack --model Qwen2.5-VL-3B-Instruct --judge gpt-4o 
```
The configuration remains identical to the above settings. Simply append specific noise parameters afterward, but ensure API calls are restricted to single-process execution.
To disable noise addition, simply omit the `noise` and `ratio` parameters. 
</details>

<details>
<summary><strong>Step 2: Evaluation</strong></summary>
Following prior work, we incorporate a traditional metric in our benchmark: the ​**​GPT score​**​. However, as this metric relies solely on a single model’s judgment, we propose complementary evaluations: (1) the ​**​SBERT score​**​ for semantic alignment, and (2) ​**​accuracy​**​ on selection-based tasks as statistical indicators.

`evaluation.py` computes the ​​**​GPT score​**, **​SBERT score​**​ and ​**​Accuracy for True/False questions​**​ across multiple perspectives, supporting diverse Video-LLMs and noise parameters.
The input is sourced from prediction results generated by `run.py`, so please run `run.py` before executing `evaluation.py`.

**Argrments**

 - `--ratio (int)`: **Proportion** of noise frames.
 - `--model (list[str])`: **List** of model names to process.
 - `--noise (list[str])`: **List** of noise types to process.
 - `--metric (str, choice are ['acc', 'sbert', 'gpt'])`: Metric to compute.
 - `--perspective (str, choice are ['qtype', 'vtype'])`: Perspective to analyse.

Example:
```
python evaluation.py --metric gpt --noise gaussian --model Qwen2.5-VL-3B-Instruct --ratio 0.9 --perspective qtype
```
The evaluation results will be printed as logs, besides. **Result Files** will also be generated in the directory `$WORK_DIR/{model_name}/{noise_name}`(If no noise is applied, the filename defaults to `"origin"`) including  **GPT score**, **SBERT Score**, and **Accuracy for True/False questions** across **multiple perspectives**.
- **.xlsx files**: Contain inference results.
- **rating.json/gpt\*.json**: Stores the overall **GPT score** and per-question-type breakdown.
- **acc\*.json**: Stores the overall **Accuracy for True/False questions** and per-question-type breakdown.
- **sbert\*.json**: Stores the overall **SBERT Score** and per-question-type breakdown.
- **score\*.xlsx**: Records scores for each individual QA pair.
</details>



# Acknowledgement
We sincerely thank [VLMEvalkit](https://github.com/open-compass/VLMEvalKit) for their pioneering works on large vision-language model evaluation. 
We also gratefully acknowledge the following open-source projects and pre-trained models, which significantly contributed to our implementation: [imagecorruptions](https://github.com/bethgelab/imagecorruptions), [Ask-Anything](https://github.com/OpenGVLab/Ask-Anything), [Chat-UniVi](https://github.com/PKU-YuanGroup/Chat-UniVi), [LLaMA-VID](https://github.com/dvlab-research/LLaMA-VID), [LLaVA](https://github.com/haotian-liu/LLaVA), [PLLaVA](https://github.com/magic-research/PLLaVA), [Video-ChatGPT](https://github.com/Amshaker/Mobile-VideoGPT).

# License

This project is released under the Apache-2.0 license. Please see the LICENSE file for more information.
