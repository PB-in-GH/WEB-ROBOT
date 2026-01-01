# qoe_transport_lab/src/qoe_lab/experiments/runner.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
import threading
import queue
import time
import heapq
import csv
import os

from qoe_lab.rtc.base import VideoFrame, RtcFeedback
from qoe_lab.rtc.trace_emulator import TraceRtcEmulator
from qoe_lab.controller.base import Controller, ControlDecision
from qoe_lab.source.base import FrameSource, GeneratedFrame


@dataclass
class RunnerConfig:
    total_s: int = 30
    tick_ms: int = 50
    print_every_s: int = 1

    # 生成侧目标缓冲（内容时间，秒）：希望“已生成但尚未播放”的未来内容至少这么多秒
    gen_buffer_target_s: int = 2

    # ---- RTC 行为：过期丢帧（late drop）----
    # 若某视频帧的内容时间戳 ts_ms < now_ms - late_drop_ms，则认为已经过时，直接丢弃不再发送
    late_drop_ms: int = 300

    # ---- 日志输出 ----
    # 若设置，将在 run() 结束后把所有记录写到 CSV（列为所有出现过的 key 的并集）
    # 若不设置且 sink 是 JsonlSink，则默认导出到 metrics.jsonl 同目录下的 metrics.csv
    csv_path: Optional[str] = None


class ExperimentRunner:
    """
    并行在线生成 + 帧级发送（tick 驱动） 的实验 runner。

    - 控制器按秒决策（second_idx -> fps）
    - 生成线程按计划生成未来秒（可提前生成；目标缓冲由 cfg.gen_buffer_target_s 控制）
    - 发送侧（run tick 循环）只发送“已 ready 且 ts_ms <= now_ms”的帧
    """

    def __init__(
        self,
        rtc: TraceRtcEmulator,
        controller: Controller,
        source: FrameSource,
        cfg: RunnerConfig,
        sink=None,
    ):
        self.rtc = rtc
        self.controller = controller
        self.source = source
        self.cfg = cfg
        self.sink = sink

        # ---- 计划：second -> fps ----
        self._plan: Dict[int, int] = {}

        # ---- 生成线程：输入 jobs（sec），输出 results（sec->frames） ----
        self._jobs: "queue.Queue[int]" = queue.Queue()  # 只放 sec
        self._results: "queue.Queue[Tuple[int, int, List[GeneratedFrame], int]]" = queue.Queue()
        # (sec, ready_ms, frames, gen_ms)

        self._stop_gen = threading.Event()
        self._gen_thread: Optional[threading.Thread] = None

        # ---- ready 帧与待发送帧 ----
        self._ready_frames: Dict[int, List[GeneratedFrame]] = {}
        self._send_heap: List[Tuple[int, int, GeneratedFrame]] = []  # (ts_ms, seq, frame)
        self._seq = 0

        # ---- 统计 ----
        self._inflight: Dict[int, int] = {}  # sec -> fps
        self._last_gen_ms: int = 0
        self._starvation_ms: int = 0
        self._last_send_wall_ms: Optional[int] = None

        # ---- 丢帧统计（发送侧主动丢弃“过时帧”）----
        self._dropped_total: int = 0
        self._dropped_window: int = 0

        # ---- 丢帧统计（网络/接收侧：由 rtc/emulator 维护；用于打印“窗口增量”）----
        self._last_drop_overflow_total: int = 0
        self._last_drop_late_total: int = 0

        # ---- source 结束：若某 sec 生成结果为空，则认为从该 sec 起没有内容 ----
        self._source_end_sec: Optional[int] = None

        self._plan_lock = threading.Lock()

        # ---- CSV 输出路径（优先 cfg.csv_path，否则从 sink 推断）----
        self._csv_path: Optional[str] = cfg.csv_path
        if self._csv_path is None and sink is not None:
            p = getattr(sink, "metrics_path", None)
            if isinstance(p, str) and p.endswith(".jsonl"):
                self._csv_path = p[:-5] + ".csv"

        # ---- 当前秒与决策 ----
        self.cur_second_idx: int = 0
        self.cur_fps: int = 0
        self.cur_decision: Optional[ControlDecision] = None

    # ---------------------- 生成线程 ----------------------
    def _gen_worker(self) -> None:
        while not self._stop_gen.is_set():
            try:
                sec = self._jobs.get(timeout=0.1)
            except queue.Empty:
                continue

            if self._source_end_sec is not None and sec >= self._source_end_sec:
                continue

            fps = self._plan.get(sec, self.cur_fps if self.cur_fps > 0 else 15)

            t0 = time.perf_counter()
            frames = self.source.generate_second(sec, fps=fps)
            gen_ms = int((time.perf_counter() - t0) * 1000)

            ready_ms = int(self.rtc.now_ms())
            if not frames:
                self._source_end_sec = sec
            self._results.put((sec, ready_ms, frames, gen_ms))

    def _start_generator_thread(self) -> None:
        if self._gen_thread is not None:
            return
        self._stop_gen.clear()
        self._gen_thread = threading.Thread(target=self._gen_worker, daemon=True)
        self._gen_thread.start()

    def _stop_generator_thread(self) -> None:
        if self._gen_thread is None:
            return
        self._stop_gen.set()
        self._gen_thread.join(timeout=2)
        self._gen_thread = None

    # ---------------------- 计划与决策 ----------------------
    def _start_new_second(self, sec: int, feedback: Optional[RtcFeedback]) -> None:
        self.cur_second_idx = sec
        decision = self.controller.decide(second_idx=sec, feedback=feedback)
        self.cur_decision = decision
        self.cur_fps = int(decision.fps)

        with self._plan_lock:
            self._plan[sec] = self.cur_fps

        print(f"[GEN] start sec={sec} fps={self.cur_fps} submit_ms={int(self.rtc.now_ms())}")

        if self._source_end_sec is None or sec < self._source_end_sec:
            self._inflight[sec] = self.cur_fps
            self._jobs.put(sec)

    def _ensure_plan_and_enqueue(self, now_ms: int, cur_sec: int) -> None:
        """
        保持生成侧缓冲 >= gen_buffer_target_s：提前调度未来的秒生成。
        """
        target_sec = cur_sec + int(self.cfg.gen_buffer_target_s)
        for s in range(cur_sec, target_sec + 1):
            if self._source_end_sec is not None and s >= self._source_end_sec:
                continue
            if s not in self._plan and s not in self._inflight and s not in self._ready_frames:
                with self._plan_lock:
                    self._plan[s] = self.cur_fps if self.cur_fps > 0 else 15
                self._inflight[s] = self._plan[s]
                self._jobs.put(s)

    def _drain_results(self) -> None:
        while True:
            try:
                sec, ready_ms, frames, gen_ms = self._results.get_nowait()
            except queue.Empty:
                break

            fps_plan = self._plan.get(sec, -1)
            frames_n = len(frames)

            # 新增：输出生成完成日志
            print(f"[GEN] done  sec={sec} fps={fps_plan} gen_ms={gen_ms} frames={frames_n} ready_ms={ready_ms}")

            self._inflight.pop(sec, None)
            self._ready_frames.setdefault(sec, []).extend(frames)
            self._last_gen_ms = gen_ms


            # 把这一秒的帧推进 send_heap
            for fr in frames:
                ts_ms = int(fr.ts_ms)
                self._seq += 1
                heapq.heappush(self._send_heap, (ts_ms, self._seq, fr))

    def _drop_late_frames(self, now_ms: int) -> int:
        """
        发送侧过时丢弃：丢弃还没送进 rtc 的 send_heap 中过时帧。
        这是“发送侧兜底”，与 emulator 内部的“队列/延迟约束丢弃”是两类不同 drop。
        """
        if not self._send_heap:
            return 0

        dropped = 0
        late_before = now_ms - int(self.cfg.late_drop_ms)

        while self._send_heap and self._send_heap[0][0] < late_before:
            _ts, _seq, _fr = heapq.heappop(self._send_heap)
            dropped += 1

        if dropped > 0:
            self._dropped_total += dropped
            self._dropped_window += dropped

        return dropped
    def _send_due_frames(self, now_ms: int) -> None:
        """
        把 ts_ms <= now_ms 的帧送入 rtc。
        RTC / TraceEmulator 只关心 ts_ms、size_bytes 和 frame_id（用于跟踪/去重/统计）。
        """
        while self._send_heap and self._send_heap[0][0] <= now_ms:
            _ts, _seq, fr = heapq.heappop(self._send_heap)

            size_bytes = int(
                getattr(fr, "size_bytes", 0)
                or getattr(fr, "bytes", 0)
                or 0
            )
            if size_bytes <= 0:
                size_bytes = 25_000

            # 关键：VideoFrame 需要 frame_id
            vf = VideoFrame(
                frame_id=int(_seq),
                ts_ms=int(fr.ts_ms),
                size_bytes=int(size_bytes),
            )

            self.rtc.send_video(vf)

    def _gen_buffer_seconds(self, now_ms: int) -> float:
        """
        估计生成侧 buffer（内容时间）：
        ready_frames + send_heap 中的最大 ts，与 now_ms 的差。
        """
        max_ts = now_ms
        if self._send_heap:
            max_ts = max(max_ts, int(self._send_heap[-1][0]) if len(self._send_heap) > 0 else now_ms)
            # heap 末尾不保证最大；保守做一次扫描（heap 通常不大，且只用于指标显示）
            max_ts = max(max_ts, max(t for (t, _, _) in self._send_heap))
        return max(0.0, (max_ts - now_ms) / 1000.0)

    def _update_starvation(self, now_ms: int, tick_ms: int, fb: RtcFeedback) -> None:
        """
        一个简化的 starvation 指标：如果接收侧队列为空且 recv_rate 很低，就累加 starvation_ms。
        """
        starving = (fb.queue_frames <= 0) and (fb.recv_rate_bps <= 1e-9)
        if starving:
            self._starvation_ms += int(tick_ms)
        else:
            self._starvation_ms = 0

    def _write_csv(self, path: str, logs: List[Dict[str, Any]]) -> None:
        if not logs:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        # 列 = 所有 key 的并集（保持相对稳定：先按第一次出现顺序，再补充新增）
        keys: List[str] = []
        seen = set()
        for rec in logs:
            for k in rec.keys():
                if k not in seen:
                    seen.add(k)
                    keys.append(k)

        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for rec in logs:
                w.writerow(rec)

    # ---------------------- 主循环 ----------------------
    def run(self) -> List[Dict[str, Any]]:
        logs: List[Dict[str, Any]] = []

        total_ms = int(self.cfg.total_s * 1000)
        tick = int(self.cfg.tick_ms)

        self.source.start()
        self._start_generator_thread()
        try:
            self._start_new_second(0, feedback=None)
            now = int(self.rtc.now_ms())
            self._ensure_plan_and_enqueue(now_ms=now, cur_sec=0)

            next_print_ms = 0

            while self.rtc.now_ms() < total_ms:
                now = int(self.rtc.now_ms())
                cur_sec = int(now // 1000)

                # 1) 秒边界：做控制决策
                if cur_sec != self.cur_second_idx:
                    fb = self.rtc.get_feedback()
                    self._start_new_second(cur_sec, feedback=fb)

                # 2) 保持生成侧 buffer
                self._ensure_plan_and_enqueue(now_ms=now, cur_sec=cur_sec)

                # 3) 收集生成结果 -> send_heap
                self._drain_results()

                # 4) 发送前先丢弃过时帧（发送侧兜底）
                self._drop_late_frames(now_ms=now)

                # 5) 发送到 rtc
                self._send_due_frames(now_ms=now)

                fb = self.rtc.get_feedback()
                self._update_starvation(now_ms=now, tick_ms=tick, fb=fb)

                # 6) 推进网络（仿真时钟）
                _events = self.rtc.step(tick)

                # 7) 关键：让“仿真时间”与“真实时间”对齐，否则生成线程永远追不上仿真
                time.sleep(tick / 1000.0)

                # 8) 打印与 sink
                if self.cfg.print_every_s > 0 and now >= next_print_ms:
                    fb = self.rtc.get_feedback()
                    rec = self._make_log_record(fb)
                    logs.append(rec)
                    self._print_status(fb, extra=rec)
                    if self.sink is not None:
                        self.sink.write(rec)
                    next_print_ms += int(self.cfg.print_every_s * 1000)

                # 如果 source 已结束，且没有任何待发送/待生成/网络队列，则提前结束
                if (
                    self._source_end_sec is not None
                    and not self._send_heap
                    and not self._inflight
                    and self.rtc.get_feedback().queue_frames == 0
                ):
                    print(f"[GEN] end  sec={self._source_end_sec} now_ms={self.rtc.now_ms()}")
                    break


            fb = self.rtc.get_feedback()
            logs.append(self._make_log_record(fb))

            # ---- 可选：导出 CSV（便于 Excel/统计分析）----
            if self._csv_path:
                try:
                    self._write_csv(self._csv_path, logs)
                except Exception as e:
                    # 不阻塞主流程；在终端提示即可
                    print(f"[warn] failed to write csv to {self._csv_path}: {e}")

            return logs
        finally:
            self._stop_generator_thread()
            self.source.close()
            if self.sink is not None:
                self.sink.close()

    def _make_log_record(self, fb: RtcFeedback) -> Dict[str, Any]:
        dec = self.cur_decision

        send_fps = float(self.cur_fps)
        rx_rate_mbps = float(fb.recv_rate_bps) / 1e6
        net_loss_pct = float(fb.loss_rate) * 100.0

        trace_bw_mbps = float(fb.extra.get("bandwidth_bps", 0.0)) / 1e6
        trace_rtt_ms = float(fb.extra.get("rtt_ms", 0.0))

        record = {
            "t_s": fb.now_ms / 1000.0,
            "second_idx": self.cur_second_idx,

            # ---- 控制/生成侧 ----
            "send_fps": send_fps,
            "gen_buffer_s": float(self._gen_buffer_seconds(int(fb.now_ms))),
            "gen_inflight": int(len(self._inflight)),
            "last_gen_ms": int(self._last_gen_ms),
            "starvation_ms": int(self._starvation_ms),

            # ---- 网络/接收侧 ----
            "rx_rate_mbps": rx_rate_mbps,
            "net_loss_pct": net_loss_pct,
            "one_way_ms": float(getattr(fb, "avg_one_way_delay_ms", 0.0)),
            "e2e_ms": float(getattr(fb, "avg_end_to_end_delay_ms", 0.0)),
            "queue_frames": int(fb.queue_frames),
            "queue_bytes": int(fb.queue_bytes),
            "client_buffer_ms": float(getattr(fb, "client_buffer_ms", 0.0)),

            # ---- trace 真值 ----
            "trace_bw_mbps": trace_bw_mbps,
            "trace_rtt_ms": trace_rtt_ms,

            # ---- 丢弃统计（区分发送侧/网络侧）----
            "send_drop_late_total": int(self._dropped_total),
            "send_drop_late_window": int(self._dropped_window),
            "net_drop_overflow_total": int(fb.extra.get("drop_overflow_total", 0)),
            "net_drop_late_total": int(fb.extra.get("drop_late_total", 0)),

            "decision_extra": (dec.extra if dec else {}),

            # ---- 旧字段（兼容保留）----
            "fps": send_fps,
            "recv_rate_mbps": rx_rate_mbps,
            "loss_pct": net_loss_pct,
            "dropped_total": int(self._dropped_total),
            "dropped_window": int(self._dropped_window),
        }

        if record["trace_bw_mbps"] <= 0:
            record["schema_warning"] = "trace_bw_mbps<=0 (ground truth bandwidth missing or zero)"

        return record

    def _print_status(self, fb: RtcFeedback, extra: Optional[Dict[str, Any]] = None) -> None:
        """
        终端状态行：保持单行、字段口径清晰。

        - send_drop_late: runner 发送前“过时丢弃”
        - net_drop_overflow/net_drop_late: rtc/emulator 内部“队列溢出/排队过久丢弃”
        """
        gb = extra.get("gen_buffer_s") if extra else self._gen_buffer_seconds(int(fb.now_ms))
        lg = extra.get("last_gen_ms") if extra else self._last_gen_ms
        st = extra.get("starvation_ms") if extra else self._starvation_ms
        inf = extra.get("gen_inflight") if extra else len(self._inflight)

        # --- 发送侧：过时丢弃（runner）---
        send_dw = extra.get("dropped_window") if extra else self._dropped_window
        send_dt = extra.get("dropped_total") if extra else self._dropped_total

        # --- 网络/接收侧：由 rtc/emulator 维护 ---
        do_tot = int(fb.extra.get("drop_overflow_total", 0))
        dl_tot = int(fb.extra.get("drop_late_total", 0))
        do_win = do_tot - self._last_drop_overflow_total
        dl_win = dl_tot - self._last_drop_late_total
        self._last_drop_overflow_total = do_tot
        self._last_drop_late_total = dl_tot

        one_way = float(getattr(fb, "avg_one_way_delay_ms", 0.0))
        e2e = float(getattr(fb, "avg_end_to_end_delay_ms", 0.0))
        cb = float(getattr(fb, "client_buffer_ms", 0.0))

        bw_mbps = float(fb.extra.get("bandwidth_bps", 0.0)) / 1e6
        rtt_ms = float(fb.extra.get("rtt_ms", 0.0))

        print(
            f"t={fb.now_ms/1000:5.1f}s  "
            f"fps={self.cur_fps:2d}  "
            f"gen_buf={gb:4.1f}s inflight={inf:2d} last_gen={lg:4d}ms  "
            f"send_drop_late(w/t)={int(send_dw):3d}/{int(send_dt):5d}  "
            f"net_drop(of/late)(w/t)={do_win:3d}/{dl_win:3d} tot={do_tot:5d}/{dl_tot:5d}  "
            f"starve={st:5d}ms  "
            f"bw={bw_mbps:4.1f}Mbps rtt={rtt_ms:5.1f}ms  "
            f"recv={fb.recv_rate_bps/1e6:4.2f}Mbps  "
            f"loss={fb.loss_rate*100:5.1f}%  "
            f"oneway={one_way:6.1f}ms e2e={e2e:7.1f}ms cb={cb:6.1f}ms  "
            f"q={fb.queue_frames:4d}f/{fb.queue_bytes/1e6:6.2f}MB"
        )

        # 每次打印后清零发送侧窗口丢帧计数（便于观察“每秒丢多少帧”）
        self._dropped_window = 0
