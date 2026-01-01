# qoe_transport_lab/src/qoe_lab/rtc/webrtc_streamer.py
from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import dataclass
from typing import Optional, Set

import numpy as np
import av
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.sdp import candidate_from_sdp, candidate_to_sdp

import time
import soundfile as sf
import librosa

from aiortc import AudioStreamTrack
from fractions import Fraction
from typing import Tuple, Optional

HTML_PAGE = r"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>MuseTalk WebRTC Viewer</title>
  <style>
    body { font-family: sans-serif; margin: 20px; }
    video { width: 720px; max-width: 100%; background: #000; }
    pre { background: #f5f5f5; padding: 10px; }
  </style>
</head>
<body>
  <h2>MuseTalk WebRTC Viewer</h2>
  <video id="v" autoplay playsinline controls></video>
  <p>状态：<span id="st">connecting...</span></p>
  <pre id="log"></pre>

<script>
(function () {
  const st = document.getElementById("st");
  const log = document.getElementById("log");
  function println(s){ log.textContent += s + "\n"; }

  const wsUrl = (location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws";
  const ws = new WebSocket(wsUrl);

  const pc = new RTCPeerConnection({
    iceServers: [{ urls: ["stun:stun.l.google.com:19302"] }]
  });

  pc.ontrack = (evt) => {
    println("[pc] ontrack kind=" + evt.track.kind);
    const v = document.getElementById("v");
    v.srcObject = evt.streams[0];
  };

  pc.onicecandidate = (evt) => {
    if (evt.candidate) {
      ws.send(JSON.stringify({
        type: "candidate",
        candidate: evt.candidate.candidate,
        sdpMid: evt.candidate.sdpMid,
        sdpMLineIndex: evt.candidate.sdpMLineIndex
      }));
    }
  };

  ws.onopen = async () => {
    println("[ws] open");
    st.textContent = "ws open, creating offer...";
    pc.addTransceiver("video", { direction: "recvonly" });
    pc.addTransceiver("audio", { direction: "recvonly" });
    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);
    ws.send(JSON.stringify({ type: "offer", sdp: pc.localDescription.sdp }));
  };

  ws.onmessage = async (evt) => {
    const msg = JSON.parse(evt.data);
    if (msg.type === "answer") {
      println("[ws] got answer");
      await pc.setRemoteDescription({ type: "answer", sdp: msg.sdp });
      st.textContent = "connected";
    } else if (msg.type === "candidate") {
      if (msg.candidate) {
        try {
          await pc.addIceCandidate({
            candidate: msg.candidate,
            sdpMid: msg.sdpMid,
            sdpMLineIndex: msg.sdpMLineIndex
          });
        } catch (e) {
          println("[pc] addIceCandidate error: " + e);
        }
      }
    } else if (msg.type === "info") {
      println("[info] " + msg.message);
    }
  };

  ws.onclose = () => {
    println("[ws] close");
    st.textContent = "ws closed";
  };
})();
</script>
</body>
</html>
"""

VIDEO_TIME_BASE = Fraction(1, 90000)  # WebRTC video 常用 90kHz 时钟


class _FrameQueueVideoTrack(VideoStreamTrack):
    """
    从队列取 (frame_bgr, ts_ms) 并按 ts_ms 节流输出，避免“生成太快导致播放倍速”。
    ts_ms 是全局呈现时间戳（从某个 t0 起算的毫秒）。
    """
    def __init__(self, q):
        super().__init__()
        self._q = q

        # 统一时钟：把“某个 ts_ms”锚定到 wall-clock
        self._start_wall: Optional[float] = None
        self._t0_ts_ms: Optional[int] = None

    async def recv(self) -> av.VideoFrame:
        item = await self._q.get()

        # 兼容旧队列：如果旧代码只放了 frame
        if isinstance(item, tuple) and len(item) == 2:
            frame_bgr, ts_ms = item
            ts_ms = int(ts_ms)
        else:
            frame_bgr = item
            # 兜底：没有 ts 时用 wall-clock 生成单调 ts
            ts_ms = int(time.time() * 1000)

        # 第一次拿到帧时，建立“ts_ms -> wall-clock”的映射
        if self._start_wall is None:
            self._start_wall = time.time()
            self._t0_ts_ms = ts_ms

        assert self._start_wall is not None and self._t0_ts_ms is not None

        # 按 ts_ms 节流：等待到该帧应呈现的时刻
        target_wall = self._start_wall + (ts_ms - self._t0_ts_ms) / 1000.0
        now = time.time()
        delay = target_wall - now
        if delay > 0:
            await asyncio.sleep(delay)

        vf = av.VideoFrame.from_ndarray(frame_bgr, format="bgr24")

        # 关键：设置 pts/time_base，让浏览器按时间戳播放
        vf.pts = int(ts_ms * 90)          # 90kHz: 1ms -> 90 ticks
        vf.time_base = VIDEO_TIME_BASE

        return vf


class _WavAudioTrack(AudioStreamTrack):
    """
    wav 音轨：48kHz mono s16，每帧 20ms（960 samples）
    关键修复：等待视频开始事件，再启动 wall-clock 计时，避免音频提前跑完。
    """
    def __init__(
        self,
        wav_path: str,
        *,
        loop: bool = False,
        frame_samples: int = 960,
        sample_rate: int = 48000,
        start_evt: Optional[threading.Event] = None,  # <-- 新增
    ):
        super().__init__()
        self._wav_path = str(wav_path)
        self._loop = bool(loop)
        self._frame_samples = int(frame_samples)
        self._sr = int(sample_rate)
        self._start_evt = start_evt

        data, sr = sf.read(self._wav_path, dtype="float32", always_2d=True)
        x = data[:, 0]
        if int(sr) != self._sr:
            x = librosa.resample(x, orig_sr=int(sr), target_sr=self._sr)

        x = np.clip(x, -1.0, 1.0)
        self._pcm = (x * 32767.0).astype(np.int16)
        self._pos = 0
        self._pts = 0

        # 注意：不在 __init__ 里启动 _start_wall
        self._start_wall: Optional[float] = None
        self._started = False

        self._time_base = Fraction(1, self._sr)

    async def recv(self) -> av.AudioFrame:
        # 0) 等待视频开始：避免音频先跑完导致后续全静音
        if not self._started:
            if self._start_evt is not None:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self._start_evt.wait)
            self._start_wall = time.time()
            self._started = True

        assert self._start_wall is not None

        # 1) 实时节流：目标每 20ms 发一次
        expected_wall = self._start_wall + (self._pts / self._sr)
        now = time.time()
        delay = expected_wall - now
        if delay > 0:
            await asyncio.sleep(delay)

        # 2) 取样本（下面保持你原来的逻辑即可）
        end = self._pos + self._frame_samples
        if end > len(self._pcm):
            if self._loop and len(self._pcm) > 0:
                self._pos = 0
                end = self._frame_samples
                chunk = self._pcm[self._pos:end]
            else:
                chunk = np.zeros((self._frame_samples,), dtype=np.int16)
        else:
            chunk = self._pcm[self._pos:end]

        self._pos += self._frame_samples

        chunk_cs = chunk.reshape(1, -1)
        af = av.AudioFrame.from_ndarray(chunk_cs, format="s16", layout="mono")
        af.sample_rate = self._sr
        af.pts = self._pts
        af.time_base = self._time_base
        self._pts += self._frame_samples
        return af



@dataclass
class WebRtcStreamerConfig:
    host: str = "127.0.0.1"
    port: int = 8080
    queue_maxsize: int = 0
    audio_path: Optional[str] = None
    audio_loop: bool = False



class WebRtcStreamer:
    """
    在独立线程中启动 aiohttp + aiortc 的 WebRTC 推流服务。
    MuseTalk 侧用 push_frame(bgr) 推帧即可。
    """
    def __init__(self, cfg: WebRtcStreamerConfig):
        self.cfg = cfg
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._queue: Optional[asyncio.Queue] = None
        self._pcs: Set[RTCPeerConnection] = set()
        self._runner: Optional[web.AppRunner] = None
        self._site: Optional[web.TCPSite] = None
        self._started_evt = threading.Event()
        self._last_ts_ms: Optional[int] = None
        self._t0_wall_ms: Optional[int] = None
        self._video_started_evt = threading.Event()


    def start(self) -> None:
        if self._thread is not None:
            return
        t = threading.Thread(target=self._run, name="webrtc-streamer", daemon=True)
        self._thread = t
        t.start()
        self._started_evt.wait(timeout=10.0)

    def push_frame(self, frame_bgr: np.ndarray, ts_ms: Optional[int] = None) -> None:
        """
        推入一帧视频。ts_ms 是“呈现时间戳(毫秒)”：
        - 推荐由上层调度器给出严格单调的 ts_ms（从 0 起或从某基准起）
        - 若不传，则用 wall-clock 自动生成单调 ts_ms（仅用于快速验证）
        """
        if self._queue is None:
            # 还没 start()，直接丢弃或抛错都行；这里选择静默丢弃并提示一次
            return

        if ts_ms is None:
            now_ms = int(time.time() * 1000)
            if self._t0_wall_ms is None:
                self._t0_wall_ms = now_ms
            ts_ms = now_ms - self._t0_wall_ms
        else:
            ts_ms = int(ts_ms)

        # 保证严格单调（至少 +1ms），避免回退
        if self._last_ts_ms is not None and ts_ms <= self._last_ts_ms:
            ts_ms = self._last_ts_ms + 1
        self._last_ts_ms = ts_ms
        if not self._video_started_evt.is_set():
            self._video_started_evt.set()

        q = self._queue

        # maxsize==0 代表无限队列：不要丢帧
        if getattr(q, "maxsize", 0) == 0:
          try:
              q.put_nowait((frame_bgr, ts_ms))
              if ts_ms % 1000 < 5:
                  print(f"[VERIFY-3] webrtc_qsize={q.qsize()} maxsize={q.maxsize}")
          except Exception:
              pass
          return


        # 有限队列：队列满则丢旧帧（保持“最新画面优先”）
        try:
            q.put_nowait((frame_bgr, ts_ms))
            if ts_ms % 1000 < 5:
              print(f"[VERIFY-3] webrtc_qsize={q.qsize()} maxsize={q.maxsize}")
        except asyncio.QueueFull:
            try:
                _ = q.get_nowait()
            except Exception:
                pass
            try:
                q.put_nowait((frame_bgr, ts_ms))
            except Exception:
                pass

    def stop(self) -> None:
        if self._loop is None:
            return

        async def _shutdown():
            # 关闭所有 peer
            for pc in list(self._pcs):
                try:
                    await pc.close()
                except Exception:
                    pass
            self._pcs.clear()

            if self._site is not None:
                try:
                    await self._site.stop()
                except Exception:
                    pass
            if self._runner is not None:
                try:
                    await self._runner.cleanup()
                except Exception:
                    pass

        # 先在 event loop 内清理资源
        fut = asyncio.run_coroutine_threadsafe(_shutdown(), self._loop)
        try:
            fut.result(timeout=10.0)
        except Exception:
            pass

        # 再停止 event loop（关键：否则 run_forever 不会退出）
        try:
            self._loop.call_soon_threadsafe(self._loop.stop)
        except Exception:
            pass

        # 最后 join 线程（可选但推荐）
        if self._thread is not None:
            try:
                self._thread.join(timeout=2.0)
            except Exception:
                pass
            self._thread = None


    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._queue = asyncio.Queue(maxsize=int(self.cfg.queue_maxsize))
        track = _FrameQueueVideoTrack(self._queue)
        audio_track = None
        if self.cfg.audio_path:
            audio_track = _WavAudioTrack(
                self.cfg.audio_path,
                loop=self.cfg.audio_loop,
                start_evt=self._video_started_evt,
            )


        app = web.Application()

        async def index(_req: web.Request) -> web.Response:
            return web.Response(text=HTML_PAGE, content_type="text/html")

        async def ws_handler(req: web.Request) -> web.WebSocketResponse:
            ws = web.WebSocketResponse()
            await ws.prepare(req)

            pc = RTCPeerConnection()
            self._pcs.add(pc)
            pc.addTrack(track)
            if audio_track is not None:
              pc.addTrack(audio_track)

            await ws.send_str(json.dumps({"type": "info", "message": "peer created"}))

            @pc.on("icecandidate")
            async def on_icecandidate(candidate):
                if candidate is None:
                    return
                await ws.send_str(json.dumps({
                    "type": "candidate",
                    "candidate": candidate_to_sdp(candidate),
                    "sdpMid": candidate.sdpMid,
                    "sdpMLineIndex": candidate.sdpMLineIndex,
                }))

            try:
                async for msg in ws:
                    if msg.type != web.WSMsgType.TEXT:
                        continue
                    data = json.loads(msg.data)

                    if data.get("type") == "offer":
                        offer = RTCSessionDescription(sdp=data["sdp"], type="offer")
                        await pc.setRemoteDescription(offer)
                        answer = await pc.createAnswer()
                        await pc.setLocalDescription(answer)
                        await ws.send_str(json.dumps({"type": "answer", "sdp": pc.localDescription.sdp}))

                    elif data.get("type") == "candidate":
                        cand_sdp = data.get("candidate")
                        if cand_sdp:
                            cand = candidate_from_sdp(cand_sdp)
                            cand.sdpMid = data.get("sdpMid")
                            cand.sdpMLineIndex = data.get("sdpMLineIndex")
                            await pc.addIceCandidate(cand)

            finally:
                try:
                    await pc.close()
                except Exception:
                    pass
                self._pcs.discard(pc)

            return ws

        app.router.add_get("/", index)
        app.router.add_get("/ws", ws_handler)

        async def _start():
            self._runner = web.AppRunner(app)
            await self._runner.setup()
            self._site = web.TCPSite(self._runner, host=self.cfg.host, port=int(self.cfg.port))
            await self._site.start()

        loop.run_until_complete(_start())
        self._started_evt.set()

        try:
            loop.run_forever()
        finally:
            # 1) 尽量先关闭站点/runner（如果还没关）
            try:
                if self._site is not None:
                    loop.run_until_complete(self._site.stop())
            except Exception:
                pass
            try:
                if self._runner is not None:
                    loop.run_until_complete(self._runner.cleanup())
            except Exception:
                pass

            # 2) 取消所有 pending tasks，避免 "Task was destroyed but pending"
            try:
                pending = asyncio.all_tasks(loop)
                for t in pending:
                    t.cancel()
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception:
                pass

            # 3) 关闭 async generators
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            except Exception:
                pass

            # 4) 关闭 loop，避免 "Event loop is closed" 的后续误用
            try:
                loop.close()
            except Exception:
                pass

