# qoe_lab/controller/qwait_s.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from qoe_lab.controller.base import Controller, ControlDecision
from qoe_lab.rtc.base import RtcFeedback


@dataclass
class QWaitSController(Controller):
    """
    QWait-S: 基于队列等待时延(queue_delay_ms)的阈值控制（baseline）。
    依赖 trace_emulator 在 feedback.extra["queue_delay_ms"] 提供队首等待时延。
    """

    min_fps: int = 5
    max_fps: int = 30
    init_fps: int = 15

    # AFR 论文里会用 very small delay；你这里先给“实验可用”的值，后续再扫参
    high_water_ms: int = 32   # >= high -> 降 fps
    low_water_ms: int = 4     # <= low  -> 升 fps

    step_up: int = 2
    step_down: int = 6

    def __post_init__(self) -> None:
        self._cur_fps = int(self.init_fps)

    def decide(self, second_idx: int, feedback: Optional[RtcFeedback]) -> ControlDecision:
        if feedback is None:
            self._cur_fps = int(self.init_fps)
            return ControlDecision(fps=self._cur_fps, extra={"type": "QWait-S", "reason": "init"})

        qwait = float(feedback.extra.get("queue_delay_ms", 0.0))

        if qwait >= float(self.high_water_ms):
            self._cur_fps = max(int(self.min_fps), int(self._cur_fps - self.step_down))
            reason = "qwait>=high -> down"
        elif qwait <= float(self.low_water_ms):
            self._cur_fps = min(int(self.max_fps), int(self._cur_fps + self.step_up))
            reason = "qwait<=low -> up"
        else:
            reason = "between -> hold"

        return ControlDecision(
            fps=int(self._cur_fps),
            extra={
                "type": "QWait-S",
                "qwait_ms": qwait,
                "low_ms": int(self.low_water_ms),
                "high_ms": int(self.high_water_ms),
                "reason": reason,
            },
        )
