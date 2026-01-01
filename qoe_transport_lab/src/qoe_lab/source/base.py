# qoe_transport_lab/src/qoe_lab/source/base.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Optional, Any


@dataclass(frozen=True)
class GeneratedFrame:
    """
    帧源输出给 runner 的“可发送帧”。
    - ts_ms: 理想播放时刻（由 runner 按秒/fps 生成）
    - size_bytes: 该帧对应的网络负载大小（第一阶段可以先用估计值；
                 等你接编码器/压缩后再用真实 byte size）
    - payload: 可选：真正的图像/编码后的 bytes（目前 trace_emulator 不需要它）
    """
    ts_ms: int
    size_bytes: int
    payload: Optional[Any] = None


class FrameSource(Protocol):
    """
    给定“这一秒需要 fps 帧”，帧源负责产出这些帧。
    """
    def start(self) -> None:
        ...

    def close(self) -> None:
        ...

    def generate_second(self, second_idx: int, fps: int) -> list[GeneratedFrame]:
        """
        生成 second_idx 这一秒的 fps 帧。
        注意：这里的 ts_ms 通常应覆盖 [second*1000, (second+1)*1000)。
        """
        ...
