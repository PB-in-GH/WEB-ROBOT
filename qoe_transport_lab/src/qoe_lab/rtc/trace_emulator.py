# qoe_lab/rtc/trace_emulator.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Deque, Tuple, Dict, Any
from collections import deque
import math
import random

from qoe_lab.rtc.base import VideoFrame, DeliveryEvent, RtcFeedback


@dataclass(frozen=True)
class TracePoint:
    start_ms: int
    bandwidth_bps: float
    rtt_ms: float
    loss_rate: float  # 0~1，按帧丢（第一版简化）


@dataclass
class _QueuedFrame:
    frame: VideoFrame
    enqueue_time_ms: int
    start_tx_time_ms: Optional[float] = None
    remaining_bytes: int = 0
    finish_tx_time_ms: Optional[int] = None


class TraceRtcEmulator:
    """
    Trace 驱动的简化 RTC Emulator。

    新增：
      - max_queue_bytes：队列字节上限，超出则丢（默认丢最旧帧，控制时延）
      - max_queue_delay_ms：队首排队过久则丢（默认丢最旧帧）
      - 丢弃计数进入 feedback.extra，便于 Runner 打印/写日志
      - queue_delay_ms：队首等待时延（QWait-S 需要）
    """

    def __init__(
        self,
        trace: List[TracePoint],
        *,
        stats_window_ms: int = 1000,
        seed: int = 1234,
        initial_now_ms: int = 0,
        max_queue_bytes: Optional[int] = None,
        max_queue_delay_ms: Optional[int] = None,
        drop_policy: str = "drop_oldest",
    ):
        if not trace:
            raise ValueError("trace 不能为空")

        trace_sorted = sorted(trace, key=lambda x: x.start_ms)
        if trace_sorted[0].start_ms != 0:
            trace_sorted = [
                TracePoint(
                    0,
                    trace_sorted[0].bandwidth_bps,
                    trace_sorted[0].rtt_ms,
                    trace_sorted[0].loss_rate,
                )
            ] + trace_sorted

        self._trace: List[TracePoint] = trace_sorted
        self._stats_window_ms = int(stats_window_ms)
        self._rng = random.Random(seed)
        self._now_ms = int(initial_now_ms)

        self._q: Deque[_QueuedFrame] = deque()
        self._q_bytes = 0
        self._inflight: Optional[_QueuedFrame] = None

        self._pending_deliveries: List[DeliveryEvent] = []
        self._stat_events: Deque[Tuple[int, int, bool, float, float]] = deque()
        self._latest_arrival_ms: Optional[int] = None

        self._max_queue_bytes = int(max_queue_bytes) if max_queue_bytes is not None else None
        self._max_queue_delay_ms = int(max_queue_delay_ms) if max_queue_delay_ms is not None else None
        self._drop_policy = str(drop_policy)

        self._drop_overflow_total = 0
        self._drop_late_total = 0
        self._delivered_total = 0

        self._drop_overflow_win = 0
        self._drop_late_win = 0

    def now_ms(self) -> int:
        return self._now_ms

    def _trace_at(self, t_ms: int) -> TracePoint:
        cur = self._trace[0]
        for p in self._trace:
            if p.start_ms <= t_ms:
                cur = p
            else:
                break
        return cur

    def _next_trace_change_after(self, t_ms: int) -> int:
        for p in self._trace:
            if p.start_ms > t_ms:
                return p.start_ms
        return 10**18

    # ---------------- 主动丢弃工具函数 ----------------
    def _inflight_bytes(self) -> int:
        return int(self._inflight.remaining_bytes) if self._inflight else 0

    def _queue_total_bytes(self) -> int:
        return int(self._q_bytes + self._inflight_bytes())

    def _drop_one_from_queue_head(self) -> bool:
        if self._drop_policy != "drop_oldest":
            raise ValueError(f"unsupported drop_policy={self._drop_policy}")

        if not self._q:
            return False
        victim = self._q.popleft()
        self._q_bytes -= victim.frame.size_bytes
        return True

    def _enforce_queue_limits_on_enqueue(self) -> None:
        if self._max_queue_bytes is None:
            return

        while self._queue_total_bytes() > self._max_queue_bytes:
            ok = self._drop_one_from_queue_head()
            if not ok:
                break
            self._drop_overflow_total += 1
            self._drop_overflow_win += 1

    def _enforce_delay_limits_before_tx(self) -> None:
        if self._max_queue_delay_ms is None:
            return

        while self._q:
            head = self._q[0]
            queued_ms = self._now_ms - head.enqueue_time_ms
            if queued_ms <= self._max_queue_delay_ms:
                break
            self._q.popleft()
            self._q_bytes -= head.frame.size_bytes
            self._drop_late_total += 1
            self._drop_late_win += 1

    # ---------------- 外部接口 ----------------
    def send_video(self, frame: VideoFrame) -> None:
        if frame.size_bytes <= 0:
            raise ValueError("frame.size_bytes 必须为正")
        qf = _QueuedFrame(
            frame=frame,
            enqueue_time_ms=self._now_ms,
            start_tx_time_ms=None,
            remaining_bytes=frame.size_bytes,
            finish_tx_time_ms=None,
        )
        self._q.append(qf)
        self._q_bytes += frame.size_bytes
        self._enforce_queue_limits_on_enqueue()

    def step(self, dt_ms: int) -> List[DeliveryEvent]:
        if dt_ms <= 0:
            return []

        end_ms = self._now_ms + int(dt_ms)

        t = self._now_ms
        while t < end_ms:
            self._enforce_delay_limits_before_tx()

            cfg = self._trace_at(t)
            next_change_ms = self._next_trace_change_after(t)
            seg_end = min(end_ms, next_change_ms)
            self._simulate_tx_segment(t, seg_end, cfg)
            t = seg_end

        self._now_ms = end_ms

        delivered_now: List[DeliveryEvent] = []
        if self._pending_deliveries:
            i = 0
            while i < len(self._pending_deliveries):
                ev = self._pending_deliveries[i]
                event_time = ev.arrival_time_ms if ev.arrival_time_ms is not None else ev.sent_time_ms
                if event_time <= self._now_ms:
                    delivered_now.append(ev)
                    i += 1
                else:
                    break
            if i > 0:
                self._pending_deliveries = self._pending_deliveries[i:]

        return delivered_now

    def get_feedback(self) -> RtcFeedback:
        now = self._now_ms
        win = self._stats_window_ms
        cutoff = now - win

        while self._stat_events and self._stat_events[0][0] < cutoff:
            self._stat_events.popleft()

        delivered_bytes = 0
        lost_frames = 0
        total_frames = 0
        one_way_sum = 0.0
        one_way_cnt = 0
        e2e_sum = 0.0
        e2e_cnt = 0

        for (event_time_ms, bytes_delivered, is_lost, one_way_delay, e2e_delay) in self._stat_events:
            total_frames += 1
            if is_lost:
                lost_frames += 1
            else:
                delivered_bytes += bytes_delivered
                one_way_sum += one_way_delay
                one_way_cnt += 1
                e2e_sum += e2e_delay
                e2e_cnt += 1

        recv_rate_bps = (delivered_bytes * 8.0) / (win / 1000.0) if win > 0 else 0.0
        loss_rate = (lost_frames / total_frames) if total_frames > 0 else 0.0
        avg_one_way = (one_way_sum / one_way_cnt) if one_way_cnt > 0 else 0.0
        avg_e2e = (e2e_sum / e2e_cnt) if e2e_cnt > 0 else 0.0

        if self._latest_arrival_ms is None:
            client_buffer_ms = 0.0
        else:
            client_buffer_ms = max(0.0, float(self._latest_arrival_ms - now))

        # ---- 新增：队首等待时延（QWait-S 需要）----
        if self._q:
            queue_delay_ms = float(now - self._q[0].enqueue_time_ms)
        else:
            queue_delay_ms = 0.0

        cfg = self._trace_at(now)
        extra: Dict[str, Any] = {
            "bandwidth_bps": cfg.bandwidth_bps,
            "rtt_ms": cfg.rtt_ms,
            "trace_loss_rate": cfg.loss_rate,

            "queue_delay_ms": queue_delay_ms,

            "drop_overflow_total": self._drop_overflow_total,
            "drop_late_total": self._drop_late_total,
            "drop_overflow_window": self._drop_overflow_win,
            "drop_late_window": self._drop_late_win,
            "delivered_total": self._delivered_total,

            "max_queue_bytes": self._max_queue_bytes,
            "max_queue_delay_ms": self._max_queue_delay_ms,
            "drop_policy": self._drop_policy,
        }

        inflight_bytes = self._inflight.remaining_bytes if self._inflight else 0
        return RtcFeedback(
            now_ms=now,
            window_ms=win,
            recv_rate_bps=recv_rate_bps,
            loss_rate=loss_rate,
            avg_one_way_delay_ms=avg_one_way,
            avg_end_to_end_delay_ms=avg_e2e,
            queue_bytes=int(self._q_bytes + inflight_bytes),
            queue_frames=int(len(self._q) + (1 if self._inflight else 0)),
            client_buffer_ms=client_buffer_ms,
            extra=extra,
        )

    def _simulate_tx_segment(self, seg_start_ms: int, seg_end_ms: int, cfg: TracePoint) -> None:
        if seg_end_ms <= seg_start_ms:
            return
        duration_s = (seg_end_ms - seg_start_ms) / 1000.0
        capacity_bytes = cfg.bandwidth_bps * duration_s / 8.0
        if capacity_bytes <= 0:
            return

        cur_t = float(seg_start_ms)
        remaining_capacity = float(capacity_bytes)

        while remaining_capacity > 1e-9:
            if self._inflight is None:
                if not self._q:
                    break
                self._inflight = self._q.popleft()
                self._q_bytes -= self._inflight.frame.size_bytes
                if self._inflight.start_tx_time_ms is None:
                    self._inflight.start_tx_time_ms = cur_t

            f = self._inflight
            assert f is not None

            send_bytes = min(float(f.remaining_bytes), remaining_capacity)
            send_time_s = (send_bytes * 8.0) / cfg.bandwidth_bps
            send_time_ms = send_time_s * 1000.0

            f.remaining_bytes -= int(math.floor(send_bytes + 1e-9))
            remaining_capacity -= send_bytes
            cur_t += send_time_ms

            if f.remaining_bytes <= 0:
                f.finish_tx_time_ms = int(round(cur_t))
                self._on_frame_tx_finished(f, cfg)
                self._inflight = None

            if cur_t >= seg_end_ms:
                break

    def _on_frame_tx_finished(self, qf: _QueuedFrame, cfg: TracePoint) -> None:
        sent_time = qf.finish_tx_time_ms if qf.finish_tx_time_ms is not None else self._now_ms
        one_way = cfg.rtt_ms / 2.0

        lost = (self._rng.random() < cfg.loss_rate)
        if lost:
            ev = DeliveryEvent(frame=qf.frame, sent_time_ms=int(sent_time), arrival_time_ms=None)
            event_time = int(sent_time)
            self._stat_events.append((event_time, 0, True, 0.0, 0.0))
            self._insert_pending(ev, event_time)
        else:
            arrival = int(round(sent_time + one_way))
            ev = DeliveryEvent(frame=qf.frame, sent_time_ms=int(sent_time), arrival_time_ms=arrival)
            e2e = float(arrival - qf.frame.ts_ms)
            self._stat_events.append((arrival, qf.frame.size_bytes, False, float(one_way), e2e))
            self._insert_pending(ev, arrival)
            self._delivered_total += 1

    def _insert_pending(self, ev: DeliveryEvent, event_time_ms: int) -> None:
        if ev.arrival_time_ms is not None:
            if (self._latest_arrival_ms is None) or (ev.arrival_time_ms > self._latest_arrival_ms):
                self._latest_arrival_ms = ev.arrival_time_ms

        idx = 0
        while idx < len(self._pending_deliveries):
            other = self._pending_deliveries[idx]
            other_time = other.arrival_time_ms if other.arrival_time_ms is not None else other.sent_time_ms
            if other_time > event_time_ms:
                break
            idx += 1
        self._pending_deliveries.insert(idx, ev)


def make_simple_trace() -> List[TracePoint]:
    return [
        TracePoint(start_ms=0, bandwidth_bps=6_000_000, rtt_ms=50, loss_rate=0.0),
        TracePoint(start_ms=5_000, bandwidth_bps=1_000_000, rtt_ms=120, loss_rate=0.2),
        TracePoint(start_ms=10_000, bandwidth_bps=3_000_000, rtt_ms=80, loss_rate=0.01),
    ]
