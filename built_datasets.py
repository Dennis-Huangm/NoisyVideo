import os
import time
from tqdm import tqdm
import torch
import torch.distributed as dist
import torchvision
from video_noise.noise_applier import NoiseRegistry

def save_as_video(tensor: torch.Tensor, save_path: str, fps: int = 30):
    array = tensor.permute(0,2,3,1).cpu().numpy()
    torchvision.io.write_video(
        filename=save_path,
        video_array=array,
        fps=fps,
        video_codec="mpeg4"
    )

def main():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)

    video_dir       = 'video'
    output_base_dir = 'output_video'
    os.makedirs(output_base_dir, exist_ok=True)
    exts = ('.mp4','.avi','.mov','.mkv')
    all_videos = [f for f in os.listdir(video_dir) if f.lower().endswith(exts)]
    my_videos = all_videos[rank::world_size]

    noises = NoiseRegistry.list_noises()

    start = time.perf_counter()
    for video_file in tqdm(my_videos, desc=f"[GPU {local_rank}] 视频列表"):
        basename, _ = os.path.splitext(video_file)
        in_path = os.path.join(video_dir, video_file)

        try:
            frames, _, info = torchvision.io.read_video(
                in_path, pts_unit="sec", output_format="TCHW"
            )
        except Exception as e:
            print(f"[{rank}] 读视频失败 {video_file}: {e}")
            continue

        fps = int(info.get("video_fps", 30))
        frames = frames.to(device)

        # 噪声循环也加进度条
        for noise in tqdm(noises, desc=f"[GPU {local_rank}] {video_file} 噪声列表", leave=False):
            out_dir  = os.path.join(output_base_dir, noise)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{basename}_{noise}.mp4")
            if os.path.exists(out_path):
                continue

            try:
                noisy = NoiseRegistry.get_noise(noise)(frames.clone(), 0.9)
                save_as_video(noisy.to('cpu'), out_path, fps=fps)
            except Exception as e:
                print(f"[{rank}] 处理失败: {video_file} / {noise}: {e}")

    elapsed = time.perf_counter() - start
    print(f"[Rank {rank}] 全部完成，用时 {elapsed:.1f}s")

if __name__ == "__main__":
    main()
