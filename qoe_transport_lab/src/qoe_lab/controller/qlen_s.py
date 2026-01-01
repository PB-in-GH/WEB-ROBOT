# qoe_lab/controller/qlen_s.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

from qoe_lab.controller.base import Controller, ControlDecision
from qoe_lab.rtc.base import RtcFeedback


@dataclass
class QLenSController(Controller):
    """
    QLen-S: 基于队列长度/字节数的阈值控制（baseline）。
    语义：队列太大 -> 降 fps；队列很小 -> 升 fps；中间保持。
    """

    min_fps: int = 5
    max_fps: int = 30
    init_fps: int = 15

    # 阈值：你可以先按字节做（更贴近网络负载）
    high_water_bytes: int = 800_000
    low_water_bytes: int = 200_000

    step_up: int = 2
    step_down: int = 6

    def __post_init__(self) -> None:
        self._cur_fps = int(self.init_fps)

    def decide(self, second_idx: int, feedback: Optional[RtcFeedback]) -> ControlDecision:
        if feedback is None:
            self._cur_fps = int(self.init_fps)
            return ControlDecision(fps=self._cur_fps, extra={"type": "QLen-S", "reason": "init"})

        q_bytes = int(getattr(feedback, "queue_bytes", 0))
        if q_bytes >= int(self.high_water_bytes):
            self._cur_fps = max(int(self.min_fps), int(self._cur_fps - self.step_down))
            reason = "q_bytes>=high -> down"
        elif q_bytes <= int(self.low_water_bytes):
            self._cur_fps = min(int(self.max_fps), int(self._cur_fps + self.step_up))
            reason = "q_bytes<=low -> up"
        else:
            reason = "between -> hold"

        return ControlDecision(
            fps=int(self._cur_fps),
            extra={
                "type": "QLen-S",
                "q_bytes": q_bytes,
                "low": int(self.low_water_bytes),
                "high": int(self.high_water_bytes),
                "reason": reason,
            },
        )
