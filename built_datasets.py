import os
import time
from tqdm import tqdm
import torch
import torch.distributed as dist
import torchvision
from video_noise.noise_applier import NoiseRegistry
import decord
decord.bridge.set_bridge("torch")

fnames = ['0C1aaSXIG_o.mp4', '68191uKawYw.mp4', '83yqxdMA4A4.mp4', '9-r4VLHQRlM_processed.mp4', 'Ak1eEUNrpgo_processed.mp4', 'Eg64S0DhAaI.mp4', 'Nv_W7Agoqio.mp4', 'P3bevifVByk.mp4', 'TUlxth701GM_processed.mp4', 'Uz1eS_85bCY.mp4', '_Zt1EuIEhvw_processed.mp4', '_e7NvQFJ2uA.mp4', 'ddzjFNvpZhM.mp4', 'lnShWOBzgGM.mp4', 'nM0cdiAn864_processed.mp4', 'ruKJCiAOmfg.mp4', 'xUkqUL5bXSE.mp4', 'zBv_fuKyg5E.mp4', 'zpnq5Hl8uwQ.mp4']

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

    try: 
        video_dir       = 'video'
        output_base_dir = 'outputs_video'
        os.makedirs(output_base_dir, exist_ok=True)
        exts = ('.mp4','.avi','.mov','.mkv')
        all_videos = sorted([
            f for f in fnames
            if f.lower().endswith(exts)
        ])
        all_videos = all_videos[:100]
        my_videos = all_videos[rank::world_size]

        noises = NoiseRegistry.list_noises() 

        start = time.perf_counter()
        for video_file in tqdm(my_videos, desc=f"[GPU {local_rank}] 视频列表"):
            basename, _ = os.path.splitext(video_file)
            in_path = os.path.join(video_dir, video_file)

            required_noises = []
            for noise in noises:
                noise_dir = os.path.join(output_base_dir, noise)
                output_path = os.path.join(noise_dir, f"{basename}_{noise}.mp4")
                if not os.path.exists(output_path):
                    required_noises.append((noise, noise_dir, output_path))
            
            if not required_noises:
                print(f"Rank {rank}: 跳过 {video_file}")
                continue

            try:
                # frames, _, info = torchvision.io.read_video(
                #     in_path, pts_unit="sec", output_format="TCHW"
                # )
                # fps = int(info.get("video_fps", 30))
                vr = decord.VideoReader(in_path, ctx=decord.cpu(0))
                total_frames = len(vr)
                fps = int(round(vr.get_avg_fps()))
                idx = torch.linspace(0, total_frames - 1, 8).round().long().tolist()

                all_frames = vr.get_batch(range(total_frames)).permute(0, 3, 1, 2)
                print(f"视频 {video_file} 帧数: {all_frames.shape[0]}")
            except Exception as e:
                print(f"[{rank}] 读视频失败 {video_file}: {e}")
                continue
            

            # 噪声循环也加进度条
            # for noise, noise_dir, output_path in tqdm(required_noises, desc=f"[GPU {local_rank}] {video_file} {noise} 噪声处理中"):
            for idx_, (noise, noise_dir, output_path) in enumerate(
                tqdm(required_noises, 
                    desc=f"[GPU {local_rank}] {video_file[:10]}.. 噪声处理", 
                    leave=False,
                    unit="noise",
                    bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}"),
                start=1
            ):
                os.makedirs(noise_dir, exist_ok=True)
                # 在日志中记录当前处理的噪声名称
                if idx_ == 1:  # 只在第一个噪声时打印视频信息
                    tqdm.write(f"[GPU {local_rank}] 正在处理 {video_file} | 噪声数: {len(required_noises)}")
                tqdm.write(f"  正在处理噪声: {noise[:15]}...")  # 截断长名称

         
                selected_frames = all_frames[idx]
                noisy_selected = NoiseRegistry.get_noise(noise)(selected_frames.clone(), 0.9)

                processed_frames = all_frames.clone()
                for i, frame_idx in enumerate(idx):
                    processed_frames[frame_idx] = noisy_selected[i]

                save_as_video(processed_frames.to('cpu'), output_path, fps=fps)


        elapsed = time.perf_counter() - start
        print(f"[Rank {rank}] 全部完成，用时 {elapsed:.1f}s")

    finally:
        dist.destroy_process_group()
        # 这里可以添加清理代码，比如删除临时文件等
        print(f"[Rank {rank}] 清理完成")

if __name__ == "__main__":
    main()
