import argparse
import asyncio
import json
import os
import sys
from aiohttp import web
from loguru import logger
from aiortc import RTCPeerConnection, RTCSessionDescription

sys.path.append("/home/pb/web-robot/lite-avatar")
from lite_avatar import liteAvatar

from .tracks import QueueVideoTrack, QueueAudioTrack


ROOT = os.path.dirname(os.path.abspath(__file__))


async def offer(request: web.Request):
    params = await request.json()
    sdp = params["sdp"]
    type_ = params["type"]

    pc = RTCPeerConnection()
    request.app["pcs"].add(pc)


    video_track = QueueVideoTrack()
    audio_track = QueueAudioTrack(sample_rate=16000, channels=1)

    pc.addTrack(video_track)
    pc.addTrack(audio_track)

    @pc.on("connectionstatechange")
    async def on_state_change():
        logger.info(f"connectionState={pc.connectionState}")

    # 设置远端 offer
    await pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type=type_))
    # 生成 answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    # 启动 liteAvatar 流 -> 喂 track（必须 pace_realtime=True，否则会把队列塞爆）
    data_dir = request.app["data_dir"]
    audio_file = request.app["audio_file"]
    fps_schedule = request.app["fps_schedule"]

    avatar = liteAvatar(data_dir=data_dir, num_threads=1, generate_offline=True, fps=30)
    loop = asyncio.get_event_loop()

    def on_video_frame(frame_bgr, pts, dur, frame_index, sec_index, fps_sec):
        # 从回调线程安全投递到 asyncio
        loop.call_soon_threadsafe(asyncio.create_task, video_track.put_frame(frame_bgr, pts, fps=max(fps_sec, 1)))

    def on_audio_frame(pcm16_bytes, pts, dur):
        loop.call_soon_threadsafe(asyncio.create_task, audio_track.put_pcm16(pcm16_bytes, pts_sec=pts))

    # 后台线程启动
    avatar.start_stream(
        audio_file_path=audio_file,
        fps_schedule=fps_schedule,
        default_fps=30,
        on_video_frame=on_video_frame,
        on_audio_frame=on_audio_frame,
        pace_realtime=True,
        audio_chunk_ms=20
    )
    request.app["avatars"].add(avatar)

    return web.json_response({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})


async def on_shutdown(app: web.Application):
    avatars = list(app.get("avatars", []))
    for avatar in avatars:
        try:
            avatar.stop_stream(join=True, timeout=2.0)
        except Exception:
            pass

    pcs = list(app.get("pcs", []))
    for pc in pcs:
        try:
            await pc.close()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--data_dir", default="/home/pb/web-robot/lite-avatar/data/preload")
    parser.add_argument("--audio_file", default="/home/pb/web-robot/lite-avatar/audio/1.wav")
    parser.add_argument("--fps_schedule_json", default="")
    args = parser.parse_args()

    if args.fps_schedule_json:
        fps_schedule = json.loads(args.fps_schedule_json)
        fps_schedule = {int(k): int(v) for k, v in fps_schedule.items()}
    else:
        fps_schedule = {0: 30}

    app = web.Application()
    app["pcs"] = set()
    app["avatars"] = set()
    app["data_dir"] = args.data_dir
    app["audio_file"] = args.audio_file
    app["fps_schedule"] = fps_schedule
    app.router.add_post("/offer", offer)
    app.on_shutdown.append(on_shutdown)

    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
