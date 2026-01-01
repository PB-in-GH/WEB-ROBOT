from __future__ import annotations

from typing import List

from qoe_lab.controller.base import ControlDecision
from qoe_lab.rtc.base import RtcFeedback


class LyapunovFpsController:
    """
    Drift+Penalty（离散 fps）版本，接口与 QueueGuardController 对齐：
      decide(second_idx, feedback) -> ControlDecision

    使用字段：
      - feedback.queue_bytes
      - feedback.avg_end_to_end_delay_ms
      - feedback.recv_rate_bps
      - feedback.extra['bandwidth_bps'] / ['rtt_ms']（trace 真值）
    """

    def __init__(
        self,
        *,
        fps_candidates: List[int] = [5, 8, 10, 12, 15, 18, 24, 30],
        init_fps: int = 30,
        min_fps: int = 5,
        max_fps: int = 30,
        target_e2e_ms: float = 300.0,
        max_e2e_ms: float = 400.0,
        frame_size_bytes: int = 25_000,
        V: float = 1.0,
        w_q: float = 1.0,
        w_d: float = 2.0,
        w_smooth: float = 0.2,
    ):
        self.fps_candidates = sorted(set(int(x) for x in fps_candidates))
        self.fps = int(init_fps)
        self.min_fps = int(min_fps)
        self.max_fps = int(max_fps)
        self.target_e2e_ms = float(target_e2e_ms)
        self.max_e2e_ms = float(max_e2e_ms)
        self.frame_size_bytes = int(frame_size_bytes)
        self.V = float(V)
        self.w_q = float(w_q)
        self.w_d = float(w_d)
        self.w_smooth = float(w_smooth)

        if not self.fps_candidates:
            raise ValueError("fps_candidates 不能为空")
        if not (self.min_fps <= self.fps <= self.max_fps):
            raise ValueError("init_fps 超出范围")

    def decide(self, second_idx: int, feedback: RtcFeedback | None) -> ControlDecision:
        if feedback is None:
            return ControlDecision(fps=self.fps, extra={"mode": "lyapunov", "reason": "no_feedback"})

        q_bytes = float(feedback.queue_bytes)
        e2e_ms = float(feedback.avg_end_to_end_delay_ms)

        # trace 真值（与你 _make_log_record 的 trace_bw_mbps 对齐）
        trace_bw_bps = float(feedback.extra.get("bandwidth_bps", 0.0))
        rx_bps = float(feedback.recv_rate_bps)

        # 服务能力估计：优先用 trace，否则用 rx
        service_bps = trace_bw_bps if trace_bw_bps > 1e-6 else rx_bps
        service_bytes_per_s = service_bps / 8.0

        best_fps = self.fps
        best_score = None

        for fps in self.fps_candidates:
            fps = int(max(self.min_fps, min(self.max_fps, fps)))
            send_bytes_per_s = fps * self.frame_size_bytes

            # 队列预测 Q' = max(0, Q + send - service)
            q_next = q_bytes + send_bytes_per_s - service_bytes_per_s
            if q_next < 0:
                q_next = 0.0

            # drift proxy：Q'^2 - Q^2
            drift = (q_next * q_next) - (q_bytes * q_bytes)

            # 延迟惩罚：超过 target 开始，超过 max 加速惩罚
            if e2e_ms <= self.target_e2e_ms:
                d_pen = 0.0
            else:
                d_pen = (e2e_ms - self.target_e2e_ms)
                if e2e_ms > self.max_e2e_ms:
                    d_pen += 3.0 * (e2e_ms - self.max_e2e_ms)

            smooth_pen = abs(fps - self.fps)

            # penalty：希望 fps 高，所以用 -fps
            score = self.w_q * drift + self.w_d * d_pen + self.w_smooth * smooth_pen + self.V * (-float(fps))

            if best_score is None or score < best_score:
                best_score = score
                best_fps = fps

        old = self.fps
        self.fps = int(best_fps)

        return ControlDecision(
            fps=self.fps,
            extra={
                "mode": "lyapunov",
                "old_fps": old,
                "q_bytes": int(q_bytes),
                "e2e_ms": e2e_ms,
                "trace_bw_mbps": trace_bw_bps / 1e6 if trace_bw_bps > 0 else 0.0,
                "rx_rate_mbps": rx_bps / 1e6 if rx_bps > 0 else 0.0,
                "chosen_score": float(best_score) if best_score is not None else None,
                "params": {
                    "target_e2e_ms": self.target_e2e_ms,
                    "max_e2e_ms": self.max_e2e_ms,
                    "frame_size_bytes": self.frame_size_bytes,
                    "V": self.V,
                    "w_q": self.w_q,
                    "w_d": self.w_d,
                    "w_smooth": self.w_smooth,
                },
            },
        )
