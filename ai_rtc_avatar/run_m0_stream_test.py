import os
import numpy as np
import soundfile as sf
from loguru import logger

# 关键：导入你刚覆盖的 lite_avatar.py
# 如果你的 python path 不包含 /home/pb/web-robot/lite-avatar
# 就用 sys.path.append 手动加
import sys
sys.path.append("/home/pb/web-robot/lite-avatar")
from lite_avatar import liteAvatar


def main():
    data_dir = "/home/pb/web-robot/lite-avatar/data/preload"
    audio_file = "/home/pb/web-robot/lite-avatar/audio/1.wav"

    # 例：0-4秒 60fps，5-9秒 30fps，10-14秒 15fps，之后默认 30fps
    fps_schedule = {0: 60, 1: 60, 2: 60, 3: 60, 4: 60,
                    5: 30, 6: 30, 7: 30, 8: 30, 9: 30,
                    10: 15, 11: 15, 12: 15, 13: 15, 14: 15}

    avatar = liteAvatar(data_dir=data_dir, num_threads=1, generate_offline=True, fps=30)

    video_count = 0
    audio_count = 0

    def on_video_frame(frame_bgr, pts, dur, frame_index, sec_index, fps_sec):
        nonlocal video_count
        video_count += 1
        if video_count % 50 == 0:
            logger.info(f"[video] idx={frame_index} pts={pts:.3f}s dur={dur:.3f}s sec={sec_index} fps_sec={fps_sec} shape={frame_bgr.shape}")

    def on_audio_frame(pcm16_bytes, pts, dur):
        nonlocal audio_count
        audio_count += 1
        if audio_count % 50 == 0:
            logger.info(f"[audio] chunk={audio_count} pts={pts:.3f}s dur={dur:.3f}s bytes={len(pcm16_bytes)}")

    # pace_realtime=False：先不 sleep，让它尽快跑完，便于看“调度是否按秒生效”
    avatar.stream(
        audio_file_path=audio_file,
        fps_schedule=fps_schedule,
        default_fps=30,
        on_video_frame=on_video_frame,
        on_audio_frame=on_audio_frame,
        pace_realtime=False,
        audio_chunk_ms=20
    )

    logger.info(f"done. video_frames={video_count}, audio_chunks={audio_count}")


if __name__ == "__main__":
    main()
