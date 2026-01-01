from __future__ import annotations

from qoe_lab.controller.base import ControlDecision
from qoe_lab.rtc.base import RtcFeedback


class FixedFpsController:
    def __init__(self, fps: int = 30):
        self.fps = int(fps)

    def decide(self, second_idx: int, feedback: RtcFeedback | None) -> ControlDecision:
        return ControlDecision(fps=self.fps, extra={"mode": "fixed", "fps": self.fps})
