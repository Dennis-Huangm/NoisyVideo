import torch
import warnings
import copy as cp
import numpy as np
import sys
from PIL import Image
import torchvision
import logging
from ..base import BaseModel
from ...smp import isimg, listinstr, get_rank_and_world_size
from ...dataset import DATASET_TYPE
from huggingface_hub import snapshot_download
from video_noise import NoiseRegistry


class PLLaVA(BaseModel):

    INSTALL_REQ = True
    INTERLEAVE = False
    VIDEO_LLM = True

    def __init__(self, model_path='ermu2001/pllava-13b', dir_root=None, **kwargs):
        sys.path.append(dir_root)
        try:
            from tasks.eval.model_utils import load_pllava
        except Exception as err:
            logging.critical(
                'Please first install requirements and set the root path to use PLLaVA. \
                Follow the instructions at https://github.com/magic-research/PLLaVA.'
            )
            raise err

        rank, world_size = get_rank_and_world_size()
        self.nframe = 8
        self.use_lora = True
        self.lora_alpha = 4
        self.pooling_shape = (16, 12, 12)
        self.RESOLUTION = 672
        self.model_path = model_path
        # remind that, once the model goes larger (30B+) may cause the memory to be heavily used up. Even Tearing Nodes.
        weight_dir = snapshot_download(model_path)
        self.model, self.processor = load_pllava(
            model_path, num_frames=self.nframe, use_lora=self.use_lora,
            weight_dir=weight_dir, lora_alpha=self.lora_alpha, pooling_shape=self.pooling_shape
        )

        #  position embedding
        self.model = self.model.to(torch.device(rank))
        self.model = self.model.eval()

    # def load_video(self, video_path, num_segments=8, resolution=336):
    #     from decord import VideoReader, cpu
    #     transforms = torchvision.transforms.Resize(size=resolution)
    #     vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    #     num_frames = len(vr)
    #     frame_indices = self.get_index(num_frames, num_segments)

    #     ### 添加代码，将帧转为torch数据，进行操作后再转换回去，注意通道位置

    #     images_group = list()
    #     for frame_index in frame_indices:
    #         img = Image.fromarray(vr[frame_index].asnumpy())
    #         images_group.append(transforms(img))
    #     return images_group
    
    def load_video(self, video_path, nframe=8, resolution=336, noise_name=None, ratio=None):
        from decord import VideoReader, cpu
        transforms = torchvision.transforms.Resize(size=resolution)
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        num_frames = len(vr)
        
        # 生成8个均匀分布的帧索引
        frame_indices = np.linspace(0, num_frames-1, num=nframe, dtype=int)
        frame_idx = frame_indices.tolist()
        
        spare_frames = vr.get_batch(frame_idx).asnumpy()
        video_tensor = torch.from_numpy(spare_frames).permute(0, 3, 1, 2)
        if noise_name is not None and ratio:
            video_tensor = NoiseRegistry.get_noise(noise_name)(video_tensor, ratio).cpu().permute(0, 2, 3, 1).numpy()
        images_group = list()
        for frame in video_tensor:
            img = Image.fromarray(frame)
            images_group.append(transforms(img))
        return images_group

    def get_index(self, num_frames, num_segments):
        seg_size = float(num_frames - 1) / num_segments
        start = int(seg_size / 2)
        offsets = np.array([
            start + int(np.round(seg_size * idx)) for idx in range(num_segments)
        ])
        return offsets

    def generate_inner(self, message, noise_name=None, ratio=None, dataset=None):
        from tasks.eval.model_utils import pllava_answer
        from tasks.eval.eval_utils import conv_templates

        question, video = self.message_to_promptvideo(message)

        img_list = self.load_video(video, nframe=self.nframe, resolution=self.RESOLUTION, noise_name=noise_name, ratio=ratio)

        if self.model_path == 'ermu2001/pllava-34b':  # using slightly different conversation mode for 34b model
            if listinstr(['Video-MCQ'], DATASET_TYPE(dataset)):  # MCQ dataset
                conv_mode = 'eval_mvbench_llavanext'
            else:  # VQA dataset
                conv_mode = 'eval_videoqa_llavanext'
        else:
            if listinstr(['Video-MCQ'], DATASET_TYPE(dataset)):  # MCQ dataset
                conv_mode = 'eval_mvbench'
            else:  # VQA dataset
                conv_mode = 'eval_videoqabench'

        conv = conv_templates[conv_mode].copy()
        if dataset in ['MVBench', 'MVBench_MP4']:
            conv.user_query(message[1]['value'], message[0]['value'], message[-2]['value'], is_mm=True)
            conv.assistant_response(message[-1]['value'])
        else:
            conv.user_query(question, is_mm=True)
        llm_response, conv = pllava_answer(
            conv=conv, model=self.model, processor=self.processor,
            do_sample=False, img_list=img_list, max_new_tokens=512, print_res=False
        )
        if dataset in ['MVBench', 'MVBench_MP4']:
            llm_response = '(' + ''.join(llm_response.split(message[-1]['value'])[1:])
        return llm_response
