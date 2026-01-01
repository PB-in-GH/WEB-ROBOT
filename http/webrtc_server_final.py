import argparse
import asyncio
import fractions
import threading
import time
from collections import deque

import av
import numpy as np
import soundfile as sf
import librosa

from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer
from aiortc import AudioStreamTrack, VideoStreamTrack

from lite_avatar import liteAvatar


# -----------------------------
# FPS Controller (thread-safe)
# -----------------------------
class FpsController:
    """
    out_fps: RTC 视频输出帧率（发送节拍 / 视频每秒更新多少帧）
    param_fps: params 序列的语义帧率（用于从 params 映射到 out_fps 时间轴，避免嘴型变速）
    """
    def __init__(self, out_fps: float, param_fps: float):
        self._lock = threading.Lock()
        self._out_fps = max(1.0, float(out_fps))
        self._param_fps = max(1.0, float(param_fps))

    def get(self):
        with self._lock:
            return self._out_fps, self._param_fps

    def set_out_fps(self, out_fps: float):
        with self._lock:
            self._out_fps = max(1.0, float(out_fps))

    def set_param_fps(self, param_fps: float):
        with self._lock:
            self._param_fps = max(1.0, float(param_fps))


# -----------------------------
# Audio source: PCM16 mono 48k
# -----------------------------
class PCM48kAudioSource:
    def __init__(self):
        self.q = deque()  # np.int16 1D
        self.lock = threading.Lock()
        self.ended = False

    def push(self, pcm16_1d: np.ndarray):
        assert pcm16_1d.dtype == np.int16 and pcm16_1d.ndim == 1
        with self.lock:
            self.q.append(pcm16_1d.copy())

    def mark_end(self):
        with self.lock:
            self.ended = True

    def pop_samples(self, n: int) -> np.ndarray | None:
        """
        取 n 个样本；不足返回 None。
        """
        with self.lock:
            if not self.q:
                return None
            out = []
            remain = n
            while remain > 0 and self.q:
                cur = self.q[0]
                if len(cur) <= remain:
                    out.append(cur)
                    remain -= len(cur)
                    self.q.popleft()
                else:
                    out.append(cur[:remain])
                    self.q[0] = cur[remain:]
                    remain = 0

            if remain > 0:
                # 不够，放回
                if out:
                    merged = np.concatenate(out)
                    self.q.appendleft(merged)
                return None

            return np.concatenate(out)


class SyncAudioTrack(AudioStreamTrack):
    """
    48kHz, 20ms/frame=960 samples
    墙钟 pacing：绝不“瞬间播完”
    """
    kind = "audio"

    def __init__(self, source: PCM48kAudioSource):
        super().__init__()
        self.src = source
        self.sample_rate = 48000
        self.samples_per_frame = 960
        self.time_base = fractions.Fraction(1, self.sample_rate)
        self.audio_pts_samples = 0
        self._t0 = None  # 第一次 recv 时设定

    async def recv(self):
        if self._t0 is None:
            self._t0 = time.time()

        # 当前音频帧的理论播放时间
        t_audio = self.audio_pts_samples / self.sample_rate
        target_wall = self._t0 + t_audio
        wait = target_wall - time.time()
        if wait > 0:
            await asyncio.sleep(wait)

        samples = self.src.pop_samples(self.samples_per_frame)
        if samples is None:
            # 没数据就补静音，保证 track 不停
            samples = np.zeros(self.samples_per_frame, dtype=np.int16)

        frame = av.AudioFrame.from_ndarray(samples.reshape(1, -1), format="s16", layout="mono")
        frame.sample_rate = self.sample_rate
        frame.pts = self.audio_pts_samples
        frame.time_base = self.time_base

        self.audio_pts_samples += self.samples_per_frame
        return frame


# -----------------------------
# Video FIFO buffer
# -----------------------------
class VideoFrameSource:
    """
    FIFO: 按顺序消费，否则会“跳到结尾”
    max_frames: 有界队列，防止无限堆积
    """
    def __init__(self, max_frames: int = 200):
        self.q = deque()
        self.lock = threading.Lock()
        self.last = None
        self.max_frames = int(max_frames)

    def push(self, bgr: np.ndarray):
        with self.lock:
            if len(self.q) >= self.max_frames:
                self.q.popleft()
            self.q.append(bgr)
            self.last = bgr

    def pop_next(self) -> np.ndarray | None:
        with self.lock:
            if not self.q:
                return None
            return self.q.popleft()

    def get_last(self) -> np.ndarray | None:
        with self.lock:
            return self.last


class SyncVideoTrack(VideoStreamTrack):
    """
    视频墙钟 pacing（绝对时间表）：
    - target_fps 可动态修改
    - 每次 recv 只取 FIFO 下一帧；如果没有则复用 last
    """
    kind = "video"

    def __init__(self, vsrc: VideoFrameSource, init_fps: float):
        super().__init__()
        self.vsrc = vsrc
        self.time_base = fractions.Fraction(1, 90000)
        self.target_fps = max(1.0, float(init_fps))

        self._t0 = None
        self._next_wall = None
        self._frame_index = 0

    def set_target_fps(self, fps: float):
        self.target_fps = max(1.0, float(fps))

    async def recv(self):
        if self._t0 is None:
            self._t0 = time.time()
            self._next_wall = self._t0

        # pacing
        wait = self._next_wall - time.time()
        if wait > 0:
            await asyncio.sleep(wait)

        period = 1.0 / self.target_fps
        self._next_wall += period

        frame = self.vsrc.pop_next()
        if frame is None:
            frame = self.vsrc.get_last()
        if frame is None:
            frame = np.zeros((512, 512, 3), dtype=np.uint8)

        vf = av.VideoFrame.from_ndarray(frame, format="bgr24")
        vf.time_base = self.time_base
        vf.pts = int(self._frame_index * (90000 * period))
        self._frame_index += 1

        if self._frame_index % 50 == 0:
            print("video out fps target=", self.target_fps, "wall=", time.time())

        return vf


# -----------------------------
# Producer thread:
# wav -> push audio (48k) + audio2param -> baseline face_gen_loop render -> push video FIFO
# -----------------------------
def _compute_bg_frame_id(avatar: liteAvatar, frame_id: int) -> int:
    # 与 lite_avatar.handle1 一致：背景帧来回往返播放
    if int(frame_id / avatar.bg_video_frame_count) % 2 == 0:
        return frame_id % avatar.bg_video_frame_count
    else:
        return avatar.bg_video_frame_count - 1 - frame_id % avatar.bg_video_frame_count


def start_wav_driven_producers(
    avatar: liteAvatar,
    wav_path: str,
    audio_src: PCM48kAudioSource,
    video_src: VideoFrameSource,
    ctrl: FpsController,
):
    """
    baseline 渲染路径：input_queue -> face_gen_loop -> output_queue
    关键：params 采样按 (out_fps, param_fps) 对齐，避免你遇到的“改了 liteavatar fps 后嘴型变速”。
    """
    def _run():
        # 1) 读 wav（float32）
        wav, sr = sf.read(wav_path, dtype="float32")
        if wav.ndim > 1:
            wav = wav[:, 0]

        # 2) 音频推送：resample->48k, PCM16, 分块 push（不按实时 sleep；AudioTrack 自己 pacing）
        wav48 = librosa.resample(wav, orig_sr=sr, target_sr=48000)
        pcm16 = np.clip(wav48 * 32767.0, -32768, 32767).astype(np.int16)
        duration_sec = len(pcm16) / 48000.0

        chunk = 48000 // 10  # 100ms
        for i in range(0, len(pcm16), chunk):
            audio_src.push(pcm16[i:i + chunk])

        # 3) 生成 params（整段）
        audio_bytes = avatar.read_wav_to_bytes(wav_path)
        params = avatar.audio2param(audio_bytes, is_complete=True)
        n_params = len(params)

        # 4) 按 out_fps 时间轴决定“输出总帧数”
        out_fps0, _ = ctrl.get()
        total_out_frames = int(duration_sec * out_fps0)
        if total_out_frames <= 0:
            total_out_frames = 1

        # 5) 生产渲染任务（baseline：塞 input_queue，face_gen_loop 会打印逐帧耗时日志）
        for k in range(total_out_frames):
            out_fps, param_fps = ctrl.get()

            # 关键映射：i_k = floor(k * param_fps / out_fps)
            i = int(k * (param_fps / out_fps))
            if i >= n_params:
                i = n_params - 1
            p = params[i]

            bg_frame_id = _compute_bg_frame_id(avatar, k)
            avatar.input_queue.put((p, bg_frame_id, k))

        # 6) 消费 output_queue（可能乱序，用 buffer 按 k 顺序推到 video FIFO）
        next_k = 0
        buf = {}

        while next_k < total_out_frames:
            item = avatar.output_queue.get()  # 阻塞
            if item is None:
                break
            k, full_img, _ = item
            buf[k] = full_img

            while next_k in buf:
                video_src.push(buf.pop(next_k))
                next_k += 1

        # 音频标记结束（允许后续变静音继续播）
        audio_src.mark_end()

        # 7) idle：维持最后一帧（无限流 demo）
        last = video_src.get_last()
        if last is not None:
            while True:
                video_src.push(last)
                time.sleep(0.05)

    threading.Thread(target=_run, daemon=True).start()


# -----------------------------
# Web handlers
# -----------------------------
async def index(request):
    return web.FileResponse(request.app["index_html"])


async def set_fps(request):
    """
    POST /set_fps
    body: {"out_fps": 20, "param_fps": 30}
    两个字段可只传其一。
    """
    data = await request.json()
    ctrl: FpsController = request.app["ctrl"]

    if "out_fps" in data:
        ctrl.set_out_fps(data["out_fps"])
    if "param_fps" in data:
        ctrl.set_param_fps(data["param_fps"])

    # 同步更新当前 vtrack 的发送节拍（单连接 demo）
    vtrack = request.app.get("current_vtrack", None)
    if vtrack is not None and "out_fps" in data:
        vtrack.set_target_fps(data["out_fps"])

    out_fps, param_fps = ctrl.get()
    return web.json_response({"out_fps": out_fps, "param_fps": param_fps})


async def offer(request):
    payload = await request.json()
    offer = RTCSessionDescription(sdp=payload["sdp"], type=payload["type"])

    config = RTCConfiguration(iceServers=[RTCIceServer(urls=["stun:stun.l.google.com:19302"])])
    pc = RTCPeerConnection(configuration=config)

    data_dir = request.app["data_dir"]
    wav_path = request.app["audio_file"]
    use_gpu = request.app["use_gpu"]
    queue_max = request.app["queue_max"]

    ctrl: FpsController = request.app["ctrl"]
    out_fps, _ = ctrl.get()

    # LiteAvatar：你说它的 fps 你会手动调，这里不干涉；但建议与你期望一致
    avatar = liteAvatar(data_dir=data_dir, num_threads=1, generate_offline=True, fps=25, use_gpu=use_gpu)

    audio_src = PCM48kAudioSource()
    video_src = VideoFrameSource(max_frames=queue_max)

    atrack = SyncAudioTrack(audio_src)
    vtrack = SyncVideoTrack(video_src, init_fps=out_fps)

    pc.addTrack(atrack)
    pc.addTrack(vtrack)

    # 让 /set_fps 可以修改当前连接的视频发送 fps（单连接 demo）
    request.app["current_vtrack"] = vtrack

    # 启动生产线程（ctrl 从 app 里取出来后传入，绝不会未定义）
    start_wav_driven_producers(avatar, wav_path, audio_src, video_src, ctrl)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        if pc.connectionState in ("failed", "closed", "disconnected"):
            await pc.close()

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.json_response({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--audio_file", type=str, required=True)
    parser.add_argument("--use_gpu", type=int, default=1)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--index_html", type=str, default="index.html")

    # 关键可调参数（不写死）
    parser.add_argument("--out_fps", type=float, default=25.0, help="RTC视频发送帧率（可运行中 /set_fps 修改）")
    parser.add_argument("--param_fps", type=float, default=30.0, help="params 序列语义帧率（映射用，可 /set_fps 修改）")
    parser.add_argument("--queue_max", type=int, default=200, help="视频 FIFO 队列最大长度")

    args = parser.parse_args()

    app = web.Application()
    app["data_dir"] = args.data_dir
    app["audio_file"] = args.audio_file
    app["use_gpu"] = args.use_gpu
    app["index_html"] = args.index_html
    app["queue_max"] = args.queue_max
    app["ctrl"] = FpsController(out_fps=args.out_fps, param_fps=args.param_fps)

    app.router.add_get("/", index)
    app.router.add_post("/offer", offer)
    app.router.add_post("/set_fps", set_fps)

    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
