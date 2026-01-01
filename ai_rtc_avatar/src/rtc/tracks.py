import asyncio
import fractions
from typing import Optional

import numpy as np
from aiortc import MediaStreamTrack
from av import AudioFrame, VideoFrame


class QueueVideoTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self):
        super().__init__()
        self._q: asyncio.Queue = asyncio.Queue()
        self._last_pts: Optional[int] = None

    async def put_frame(self, frame_bgr: np.ndarray, pts_sec: float, fps: int):
        """
        frame_bgr: HxWx3 uint8 BGR
        pts_sec: presentation time in seconds
        fps: nominal fps for time_base
        """
        # pts 以 90kHz 或以 fps time_base 都行；这里用 fps time_base，易读
        # time_base = 1/fps
        pts = int(round(pts_sec * fps))
        await self._q.put((frame_bgr, pts, fps))

    async def recv(self) -> VideoFrame:
        frame_bgr, pts, fps = await self._q.get()
        vf = VideoFrame.from_ndarray(frame_bgr, format="bgr24")
        vf.pts = pts
        vf.time_base = fractions.Fraction(1, fps)
        return vf


class QueueAudioTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        super().__init__()
        self.sample_rate = sample_rate
        self.channels = channels
        self._q: asyncio.Queue = asyncio.Queue()

    async def put_pcm16(self, pcm16_bytes: bytes, pts_sec: float):
        """
        pcm16_bytes: little-endian int16, mono
        PyAV 期望 ndarray 维度为 (channels, samples)
        """
        samples = np.frombuffer(pcm16_bytes, dtype=np.int16)

        # (samples,) -> (1, samples)
        samples_2d = samples.reshape(1, -1)

        af = AudioFrame.from_ndarray(samples_2d, format="s16", layout="mono")
        af.sample_rate = self.sample_rate

        # pts 用 sample_rate time_base
        af.pts = int(round(pts_sec * self.sample_rate))
        af.time_base = fractions.Fraction(1, self.sample_rate)

        await self._q.put(af)


    async def recv(self) -> AudioFrame:
        af = await self._q.get()
        return af
