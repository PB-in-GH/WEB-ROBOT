# qoe_transport_lab/src/qoe_lab/controller/queue_guard.py
from __future__ import annotations

from qoe_lab.controller.base import ControlDecision
from qoe_lab.rtc.base import RtcFeedback


class QueueGuardController:
    """
    简单策略：
    - 如果发送端排队字节 > high_water_bytes：降低 fps
    - 如果排队字节 < low_water_bytes：提高 fps
    - 否则保持不变

    目的：验证“按秒调度 fps”会影响队列与 e2e 延迟。
    """
    def __init__(
        self,
        *,
        min_fps: int = 5,
        max_fps: int = 30,
        init_fps: int = 30,
        low_water_bytes: int = 200_000,   # 0.2MB
        high_water_bytes: int = 1_000_000, # 1MB
        step_up: int = 2,
        step_down: int = 5,
    ):
        if not (0 < min_fps <= init_fps <= max_fps):
            raise ValueError("fps 范围设置不合法")
        self.min_fps = int(min_fps)
        self.max_fps = int(max_fps)
        self.fps = int(init_fps)
        self.low = int(low_water_bytes)
        self.high = int(high_water_bytes)
        self.step_up = int(step_up)
        self.step_down = int(step_down)

    def decide(self, second_idx: int, feedback: RtcFeedback | None) -> ControlDecision:
        if feedback is None:
            return ControlDecision(fps=self.fps, extra={"mode": "queue_guard", "reason": "no_feedback"})

        qbytes = int(feedback.queue_bytes)

        if qbytes > self.high:
            self.fps = max(self.min_fps, self.fps - self.step_down)
            reason = "queue_high"
        elif qbytes < self.low:
            self.fps = min(self.max_fps, self.fps + self.step_up)
            reason = "queue_low"
        else:
            reason = "queue_mid"

        return ControlDecision(
            fps=int(self.fps),
            extra={
                "mode": "queue_guard",
                "reason": reason,
                "queue_bytes": qbytes,
                "low": self.low,
                "high": self.high,
            },
        )
