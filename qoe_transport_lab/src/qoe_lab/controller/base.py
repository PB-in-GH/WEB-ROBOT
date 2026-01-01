# qoe_transport_lab/src/qoe_lab/controller/base.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Optional, Dict, Any

from qoe_lab.rtc.base import RtcFeedback


@dataclass(frozen=True)
class ControlDecision:
    """
    决策输出：本秒要发送/生成的 fps（整数）。
    备注：后续你若引入“关键帧/非关键帧”或“分层编码”，可以扩展这里。
    """
    fps: int
    extra: Dict[str, Any]  # 预留：调试信息、Lyapunov 变量等


class Controller(Protocol):
    """
    上层控制器统一接口：每秒调用一次。
    """
    def decide(self, second_idx: int, feedback: Optional[RtcFeedback]) -> ControlDecision:
        ...
