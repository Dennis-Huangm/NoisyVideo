# Denis
# -*- coding: utf-8 -*-
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import torchvision
from video_noise.noise_applier import NoiseRegistry
from video_noise.utils import save_image
import time


def compare(ori_video, noi_video):
    ori_video = ori_video.permute(0, 2, 3, 1).cpu().numpy()
    noi_video = noi_video.permute(0, 2, 3, 1).cpu().numpy()
    nframe = ori_video.shape[0]

    ssim_score = 0
    psnr_score = 0

    for source, noisy in zip(ori_video, noi_video):
        ssim_score += ssim(source, noisy, data_range=255, channel_axis=-1)
        psnr_score += psnr(source, noisy, data_range=255)
    
    print(f"SSIM: {ssim_score / nframe:.4f}")
    print(f"PSNR: {psnr_score / nframe:.2f} dB")


video_path = "video_noise/sample/other_video.mp4"
video_frames, _, info = torchvision.io.read_video(
    video_path,
    pts_unit="sec",
    output_format="TCHW",
)
print(video_frames.shape)

start = time.perf_counter()  # 性能计数器
print(NoiseRegistry.list_noises())
video = NoiseRegistry.get_noise("shadow")(video_frames.clone(), 1)
end = time.perf_counter()
print(f"高精度计时: {end - start:.8f} 秒")
save_image(video[0])
compare(video_frames, video)

