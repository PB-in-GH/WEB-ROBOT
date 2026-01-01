# qoe_transport_lab/src/qoe_lab/source/synthetic.py
from __future__ import annotations

from qoe_lab.source.base import FrameSource, GeneratedFrame


class SyntheticFrameSource:
    def __init__(self, frame_size_bytes: int = 25_000):
        self.frame_size_bytes = int(frame_size_bytes)

    def start(self) -> None:
        return

    def close(self) -> None:
        return

    def generate_second(self, second_idx: int, fps: int) -> list[GeneratedFrame]:
        if fps <= 0:
            return []
        start_ms = second_idx * 1000
        # 用整数 ms 间隔近似；第一版足够
        interval_ms = max(1, int(round(1000 / fps)))

        out = []
        t = start_ms
        end_ms = start_ms + 1000
        while t < end_ms:
            out.append(GeneratedFrame(ts_ms=t, size_bytes=self.frame_size_bytes, payload=None))
            t += interval_ms
        return out
