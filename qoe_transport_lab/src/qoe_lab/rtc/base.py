# qoe_transport_lab/src/qoe_lab/rtc/base.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Dict, Any


@dataclass(frozen=True)
class VideoFrame:
    frame_id: int
    ts_ms: int
    size_bytes: int


@dataclass(frozen=True)
class DeliveryEvent:
    frame: VideoFrame
    sent_time_ms: int
    arrival_time_ms: Optional[int]  # None 表示丢包


@dataclass
class RtcFeedback:
    now_ms: int
    window_ms: int
    recv_rate_bps: float
    loss_rate: float
    avg_one_way_delay_ms: float
    avg_end_to_end_delay_ms: float
    queue_bytes: int
    queue_frames: int
    client_buffer_ms: float
    extra: Dict[str, Any]


class RtcModule(Protocol):
    def send_video(self, frame: VideoFrame) -> None:
        ...

    def step(self, dt_ms: int) -> list[DeliveryEvent]:
        ...

    def get_feedback(self) -> RtcFeedback:
        ...

    def now_ms(self) -> int:
        ...
