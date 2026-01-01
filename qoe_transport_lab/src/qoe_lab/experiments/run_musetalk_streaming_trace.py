# qoe_lab/experiments/run_musetalk_streaming_trace.py
from __future__ import annotations

import os
import json
from typing import Any, Dict

from qoe_lab.rtc.trace_emulator import TraceRtcEmulator, make_simple_trace
from qoe_lab.experiments.runner import RunnerConfig, ExperimentRunner

# 控制器：五选一
from qoe_lab.controller.fixed import FixedFpsController
from qoe_lab.controller.queue_guard import QueueGuardController
from qoe_lab.controller.lyapunov import LyapunovFpsController
from qoe_lab.controller.qlen_s import QLenSController
from qoe_lab.controller.qwait_s import QWaitSController

from qoe_lab.source.musetalk_streaming import MusetalkStreamingFrameSource, MusetalkModelConfig
from qoe_lab.logging.sinks import make_run_dir, JsonlSink, write_meta
from qoe_lab.metrics.network_score import compute_network_score_from_jsonl, NetworkScoreConfig, NetworkScoreWeights


# =========================
# 只改这个数字即可切换 controller
# 1 Fixed | 2 QueueGuard | 3 Lyapunov | 4 QLen-S | 5 QWait-S
# 也可用环境变量覆盖：CONTROLLER_ID=5 python -m ...
# =========================
CONTROLLER_ID = int(os.getenv("CONTROLLER_ID", "5"))


# =========================
# 集中配置：避免到处改参数
# =========================
CONTROLLER_CFG: Dict[int, Dict[str, Any]] = {
    1: dict(
        fps=15,
    ),
    2: dict(
        min_fps=5,
        max_fps=15,
        init_fps=15,
        low_water_bytes=200_000,
        high_water_bytes=800_000,
        step_up=2,
        step_down=6,
    ),
    3: dict(
        fps_candidates=[5, 8, 10, 12, 15],
        init_fps=15,
        min_fps=5,
        max_fps=15,
        target_e2e_ms=300.0,
        max_e2e_ms=400.0,
        frame_size_bytes=25_000,
        V=1.0,
        w_q=1.0,
        w_d=2.0,
        w_smooth=0.2,
    ),
    4: dict(
        min_fps=5,
        max_fps=15,
        init_fps=15,
        low_water_bytes=200_000,
        high_water_bytes=800_000,
        step_up=2,
        step_down=6,
    ),
    5: dict(
        min_fps=5,
        max_fps=15,
        init_fps=15,
        low_water_ms=4,
        high_water_ms=32,
        step_up=2,
        step_down=6,
    ),
}


def controller_name(controller_id: int) -> str:
    return {
        1: "Fixed",
        2: "QueueGuard",
        3: "Lyapunov",
        4: "QLen-S",
        5: "QWait-S",
    }.get(controller_id, f"Unknown({controller_id})")


def build_controller(controller_id: int):
    cfg = CONTROLLER_CFG.get(controller_id)
    if cfg is None:
        raise ValueError(f"Unknown CONTROLLER_ID={controller_id}. Valid: {sorted(CONTROLLER_CFG.keys())}")

    if controller_id == 1:
        return FixedFpsController(**cfg)

    if controller_id == 2:
        return QueueGuardController(**cfg)

    if controller_id == 3:
        return LyapunovFpsController(**cfg)

    if controller_id == 4:
        return QLenSController(**cfg)

    if controller_id == 5:
        return QWaitSController(**cfg)

    raise ValueError(f"Unknown CONTROLLER_ID={controller_id}")


def main():
    # ---------------- RTC（Trace Emulator）----------------
    rtc = TraceRtcEmulator(
        make_simple_trace(),
        stats_window_ms=1000,
        seed=2,
        max_queue_delay_ms=400,
        max_queue_bytes=2 * 1024 * 1024,
        drop_policy="drop_oldest",
    )

    # ---------------- Controller ----------------
    controller = build_controller(CONTROLLER_ID)
    cname = controller_name(CONTROLLER_ID)
    ccfg = CONTROLLER_CFG.get(CONTROLLER_ID, {})

    # QWait-S 必须依赖 feedback.extra["queue_delay_ms"]
    if CONTROLLER_ID == 5:
        fb = rtc.get_feedback()
        if "queue_delay_ms" not in fb.extra:
            raise RuntimeError(
                "QWait-S 需要 TraceRtcEmulator.get_feedback().extra['queue_delay_ms']。\n"
                "你当前 emulator 未提供该字段：请确认已按我们之前的补丁更新 trace_emulator.py。"
            )

    # ---------------- MuseTalk Source ----------------
    muse_cfg = MusetalkModelConfig(
        musetalk_root="/home/pb/web-robot/MuseTalk",
        version="v15",
        gpu_id=2,
        batch_size=20,
        use_fp16=False,
        jpeg_quality=None,
        constant_size_bytes=25_000,
    )

    source = MusetalkStreamingFrameSource(
        muse_cfg,
        inference_yaml="/home/pb/web-robot/MuseTalk/configs/inference/realtime.yaml",
        avatar_id="avator_1",
        audio_key="audio_0",
        base_fps=30,
        limit_seconds=60,
    )

    # ---------------- Runner Config ----------------
    cfg = RunnerConfig(
        total_s=30,
        tick_ms=50,
        print_every_s=1,
        gen_buffer_target_s=2,
        late_drop_ms=300,
        csv_path=None,
    )

    # ---------------- Run Dir / Sink ----------------
    # 关键：run 名字写上 controller，避免覆盖或混淆
    tag = f"musetalk_streaming_trace_c{CONTROLLER_ID}_{cname}"
    run = make_run_dir("/home/pb/web-robot/qoe_transport_lab/runs", tag)
    sink = JsonlSink(run.metrics_path)

    # ---------------- Meta ----------------
    write_meta(
        run.meta_path,
        {
            "entry": "run_musetalk_streaming_trace",
            "controller_id": CONTROLLER_ID,
            "controller_name": cname,
            "controller_cfg": ccfg,
            "trace": "make_simple_trace()",
            "runner": cfg.__dict__,
            "controller": {"type": controller.__class__.__name__, **controller.__dict__},
            "source": {
                "type": "MusetalkStreamingFrameSource",
                "avatar_id": "avator_1",
                "audio_key": "audio_0",
                "base_fps": 30,
                "limit_seconds": 60,
                "gpu_id": 2,
                "use_fp16": False,
                "constant_size_bytes": 25000,
            },
            "rtc": {
                "type": "TraceRtcEmulator",
                "max_queue_delay_ms": 400,
                "max_queue_bytes": 2 * 1024 * 1024,
                "drop_policy": "drop_oldest",
                "feedback_signals": ["queue_frames", "queue_bytes", "extra.queue_delay_ms"],
            },
        },
    )

    # ---------------- Run ----------------
    runner = ExperimentRunner(rtc, controller, source, cfg, sink=sink)

    print(f"\n===== run_musetalk_streaming_trace | controller={CONTROLLER_ID}({cname}) =====")
    logs = runner.run()

    # ---------------- Score ----------------
    ns_cfg = NetworkScoreConfig(
        max_delay_ms=400.0,
        ground_truth_c_mbps=None,
        clip_0_100=True,
        weights=NetworkScoreWeights(0.1, 0.1, 0.5),
    )
    ns = compute_network_score_from_jsonl(run.metrics_path, ns_cfg)

    summary_path = os.path.join(run.out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "controller_id": CONTROLLER_ID,
                "controller_name": cname,
                "network_score": ns.network_score,
                "delay_score": ns.delay_score,
                "recv_score": ns.recv_score,
                "loss_score": ns.loss_score,
                "delay_95th_ms": ns.delay_95th_ms,
                "min_delay_ms": ns.min_delay_ms,
                "avg_recv_mbps": ns.avg_recv_mbps,
                "avg_trace_bw_mbps": ns.avg_bw_mbps,
                "avg_loss_pct": ns.avg_loss_ratio * 100.0,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    last = logs[-1]
    print(
        f"[summary] controller={CONTROLLER_ID}({cname}) "
        f"t={last['t_s']:.1f}s fps={last['send_fps']} "
        f"recv={last['rx_rate_mbps']:.2f}Mbps loss={last['net_loss_pct']:.1f}% "
        f"e2e={last['e2e_ms']:.1f}ms qwait={last.get('queue_delay_ms', 0.0):.1f}ms "
        f"q={last['queue_frames']}f/{last['queue_bytes']/1e6:.2f}MB"
    )
    print(f"[score] Ns={ns.network_score:.3f} (summary.json written)")


if __name__ == "__main__":
    main()
