# qoe_transport_lab/src/qoe_lab/experiments/sweep_trace_csv_runner.py
from __future__ import annotations

import argparse
import inspect
import json
import os
import time
from typing import Dict, Any, List, Optional

from qoe_lab.trace.loader import load_trace_csv_piecewise
from qoe_lab.rtc.trace_emulator import TraceRtcEmulator, TracePoint
from qoe_lab.experiments.runner import RunnerConfig, ExperimentRunner

from qoe_lab.controller.fixed import FixedFpsController
from qoe_lab.controller.queue_guard import QueueGuardController
from qoe_lab.controller.lyapunov import LyapunovFpsController

from qoe_lab.source.synthetic import SyntheticFrameSource

from qoe_lab.metrics.network_score import (
    compute_network_score_from_jsonl,
    NetworkScoreConfig,
    NetworkScoreWeights,
)


def _make_run_dir(runs_root: str, name: str) -> str:
    ts = int(time.time() * 1000)
    run_dir = os.path.join(runs_root, f"{name}_{ts}")
    os.makedirs(run_dir, exist_ok=False)
    return run_dir


def _write_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _write_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _trace_meta(trace: List[TracePoint]) -> Dict[str, Any]:
    return {
        "num_points": len(trace),
        "points": [
            {
                "start_ms": p.start_ms,
                "bandwidth_bps": p.bandwidth_bps,
                "rtt_ms": p.rtt_ms,
                "loss_rate": p.loss_rate,
            }
            for p in trace
        ],
    }


def _build_source(source_name: str, frame_size_bytes: int, musetalk_kwargs: Optional[Dict[str, Any]]):
    """
    统一创建 source。
    - synthetic: SyntheticFrameSource(frame_size_bytes=...)
    - musetalk_streaming: MusetalkStreamingFrameSource(**filtered_kwargs)
        * 不硬编码 __init__ 参数名；通过 inspect.signature 过滤 kwargs
        * 缺必填参数会抛异常，提示具体缺失字段
    """
    source_name = source_name.strip().lower()

    if source_name == "synthetic":
        return SyntheticFrameSource(frame_size_bytes=int(frame_size_bytes))

    if source_name == "musetalk_streaming":
        if musetalk_kwargs is None:
            raise ValueError("选择 musetalk_streaming 时，必须提供 --musetalk_config_json 指向一个 JSON 文件。")

        # 延迟 import：避免没有 MuseTalk 环境时影响 synthetic
        from qoe_lab.source.musetalk_streaming import MusetalkStreamingFrameSource, MusetalkModelConfig  # type: ignore

        # ---- 1) 先构造 cfg: MusetalkModelConfig ----
        cfg_obj = None
        if "cfg_json" in musetalk_kwargs:
            cfg_json = musetalk_kwargs["cfg_json"]
            if not isinstance(cfg_json, dict):
                raise ValueError("musetalk_config_json 里的 cfg_json 必须是一个 JSON object（字典）")
            # 过滤到 MusetalkModelConfig.__init__ 支持的键
            cfg_sig = inspect.signature(MusetalkModelConfig.__init__)
            cfg_params = cfg_sig.parameters
            cfg_filtered = {k: v for k, v in cfg_json.items() if k in cfg_params}
            # 检查 cfg 必填
            cfg_missing = []
            for name, p in cfg_params.items():
                if name == "self":
                    continue
                if p.default is inspect._empty and name not in cfg_filtered:
                    cfg_missing.append(name)
            if cfg_missing:
                raise ValueError(
                    "MusetalkModelConfig 缺少必填参数："
                    + ", ".join(cfg_missing)
                    + f"\n你提供的 cfg_json keys={sorted(list(cfg_json.keys()))}\n"
                    + f"MusetalkModelConfig.__init__ 签名={cfg_sig}"
                )
            cfg_obj = MusetalkModelConfig(**cfg_filtered)

        # ---- 2) 构造 MusetalkStreamingFrameSource 参数 ----
        sig = inspect.signature(MusetalkStreamingFrameSource.__init__)
        params = sig.parameters

        filtered: Dict[str, Any] = {}
        for k, v in musetalk_kwargs.items():
            if k in params:
                filtered[k] = v

        # cfg 处理：如果用户直接给了 "cfg"，允许；否则用 cfg_json 构造出的 cfg_obj
        if "cfg" in params:
            if "cfg" in musetalk_kwargs:
                filtered["cfg"] = musetalk_kwargs["cfg"]
            else:
                if cfg_obj is None:
                    raise ValueError(
                        "source=musetalk_streaming 需要 cfg: MusetalkModelConfig。\n"
                        "请在 musetalk_config_json 中提供 cfg_json（用于构造 cfg）。"
                    )
                filtered["cfg"] = cfg_obj

        # 兼容你之前写的别名：inference_config -> inference_yaml, fps_base -> base_fps
        if "inference_yaml" in params and "inference_yaml" not in filtered and "inference_config" in musetalk_kwargs:
            filtered["inference_yaml"] = musetalk_kwargs["inference_config"]
        if "base_fps" in params and "base_fps" not in filtered and "fps_base" in musetalk_kwargs:
            filtered["base_fps"] = musetalk_kwargs["fps_base"]

        # 检查必填参数是否齐全（排除 self）
        missing = []
        for name, p in params.items():
            if name == "self":
                continue
            if p.default is inspect._empty and name not in filtered:
                missing.append(name)

        if missing:
            raise ValueError(
                "MusetalkStreamingFrameSource 缺少必填参数："
                + ", ".join(missing)
                + f"\n你提供的 keys={sorted(list(musetalk_kwargs.keys()))}\n"
                + f"__init__ 签名={sig}"
            )

        return MusetalkStreamingFrameSource(**filtered)


    raise ValueError(f"未知 source={source_name}，仅支持 synthetic / musetalk_streaming")


def run_one(
    *,
    runs_root: str,
    base_name: str,
    tag: str,
    trace: List[TracePoint],
    controller,
    source_name: str,
    source: Any,
    total_s: int,
    tick_ms: int,
    print_every_s: int,
) -> str:
    rtc = TraceRtcEmulator(trace, stats_window_ms=1000, seed=2)
    cfg = RunnerConfig(total_s=total_s, tick_ms=tick_ms, print_every_s=print_every_s)

    runner = ExperimentRunner(rtc, controller, source, cfg)
    print(f"\n===== {base_name}::{tag} (source={source_name}) =====")
    logs = runner.run()

    run_dir = _make_run_dir(runs_root, f"{base_name}__{source_name}__{tag}")
    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    meta_path = os.path.join(run_dir, "meta.json")
    summary_path = os.path.join(run_dir, "summary.json")

    _write_jsonl(metrics_path, logs)
    _write_json(meta_path, {
        "base_name": base_name,
        "tag": tag,
        "source_name": source_name,
        "runner": {"total_s": total_s, "tick_ms": tick_ms, "print_every_s": print_every_s},
        "trace": _trace_meta(trace),
        "controller": getattr(controller, "__class__", type(controller)).__name__,
    })

    ns_cfg = NetworkScoreConfig(
        max_delay_ms=400.0,
        ground_truth_c_mbps=None,
        clip_0_100=True,
        weights=NetworkScoreWeights(0.1, 0.1, 0.5),
    )
    ns = compute_network_score_from_jsonl(metrics_path, ns_cfg)

    _write_json(summary_path, {
        "network_score": ns.network_score,
        "delay_score": ns.delay_score,
        "recv_score": ns.recv_score,
        "loss_score": ns.loss_score,
        "delay_95th_ms": ns.delay_95th_ms,
        "min_delay_ms": ns.min_delay_ms,
        "avg_recv_mbps": ns.avg_recv_mbps,
        "avg_trace_bw_mbps": ns.avg_bw_mbps,
        "avg_loss_pct": ns.avg_loss_ratio * 100.0,
    })

    last = logs[-1]
    print(
        f"[summary] {base_name}::{tag}: "
        f"Ns={ns.network_score:.3f} ds={ns.delay_score:.3f} cs={ns.recv_score:.3f} ls={ns.loss_score:.3f}  "
        f"p95={ns.delay_95th_ms:.1f}ms recv={ns.avg_recv_mbps:.2f}Mbps bw={ns.avg_bw_mbps:.2f}Mbps loss={ns.avg_loss_ratio*100:.2f}%  "
        f"(t={last['t_s']:.1f}s fps={last['fps']} q={last['queue_frames']}f/{last['queue_bytes']/1e6:.2f}MB)"
    )

    return run_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace_csv", type=str, required=True, help="CSV: start_ms,bandwidth_bps,rtt_ms,loss_rate")
    ap.add_argument("--runs_root", type=str, default="/home/pb/web-robot/qoe_transport_lab/runs")
    ap.add_argument("--name", type=str, default=None, help="base name; default: trace file stem")
    ap.add_argument("--total_s", type=int, default=15)
    ap.add_argument("--tick_ms", type=int, default=50)
    ap.add_argument("--print_every_s", type=int, default=1)

    # 选择 source
    ap.add_argument("--source", type=str, default="synthetic", choices=["synthetic", "musetalk_streaming"])
    ap.add_argument("--frame_size_bytes", type=int, default=25_000, help="仅 synthetic 使用；musetalk 由模型决定/或其参数决定")
    ap.add_argument("--musetalk_config_json", type=str, default=None, help="source=musetalk_streaming 时必填：JSON 文件路径")

    # controller params
    ap.add_argument("--fixed_fps", type=int, default=30)

    ap.add_argument("--qg_min_fps", type=int, default=5)
    ap.add_argument("--qg_max_fps", type=int, default=30)
    ap.add_argument("--qg_init_fps", type=int, default=30)
    ap.add_argument("--qg_low", type=int, default=200_000)
    ap.add_argument("--qg_high", type=int, default=1_000_000)
    ap.add_argument("--qg_step_up", type=int, default=2)
    ap.add_argument("--qg_step_down", type=int, default=5)

    ap.add_argument("--ly_init_fps", type=int, default=30)
    ap.add_argument("--ly_target_e2e_ms", type=float, default=300.0)
    ap.add_argument("--ly_max_e2e_ms", type=float, default=400.0)
    ap.add_argument("--ly_V", type=float, default=1.0)
    ap.add_argument("--ly_w_q", type=float, default=1.0)
    ap.add_argument("--ly_w_d", type=float, default=2.0)
    ap.add_argument("--ly_w_smooth", type=float, default=0.2)

    args = ap.parse_args()

    trace = load_trace_csv_piecewise(args.trace_csv)

    base = os.path.splitext(os.path.basename(args.trace_csv))[0] if args.name is None else args.name
    os.makedirs(args.runs_root, exist_ok=True)

    musetalk_kwargs = None
    if args.source == "musetalk_streaming":
        if not args.musetalk_config_json:
            raise ValueError("--source musetalk_streaming 时必须提供 --musetalk_config_json")
        with open(args.musetalk_config_json, "r", encoding="utf-8") as f:
            musetalk_kwargs = json.load(f)
        if not isinstance(musetalk_kwargs, dict):
            raise ValueError("musetalk_config_json 必须是一个 JSON object（字典）")

    # 一次创建 source（每个 controller 共用同一份 source 实例通常是危险的，因为 source 内可能有状态）
    # 所以这里：每个 run_one 都重新 build_source，保证隔离。
    fixed = FixedFpsController(fps=args.fixed_fps)
    qg = QueueGuardController(
        min_fps=args.qg_min_fps,
        max_fps=args.qg_max_fps,
        init_fps=args.qg_init_fps,
        low_water_bytes=args.qg_low,
        high_water_bytes=args.qg_high,
        step_up=args.qg_step_up,
        step_down=args.qg_step_down,
    )
    ly = LyapunovFpsController(
        init_fps=args.ly_init_fps,
        min_fps=min(args.qg_min_fps, args.ly_init_fps),
        max_fps=max(args.qg_max_fps, args.ly_init_fps),
        target_e2e_ms=args.ly_target_e2e_ms,
        max_e2e_ms=args.ly_max_e2e_ms,
        frame_size_bytes=args.frame_size_bytes,  # synthetic 时代表 frame size；musetalk 时用于 lyapunov 的队列预测，可在 json 里另行校准
        V=args.ly_V,
        w_q=args.ly_w_q,
        w_d=args.ly_w_d,
        w_smooth=args.ly_w_smooth,
    )

    for tag, ctrl in [("fixed", fixed), ("queue_guard", qg), ("lyapunov", ly)]:
        source = _build_source(args.source, args.frame_size_bytes, musetalk_kwargs)
        run_one(
            runs_root=args.runs_root,
            base_name=base,
            tag=tag,
            trace=trace,
            controller=ctrl,
            source_name=args.source,
            source=source,
            total_s=args.total_s,
            tick_ms=args.tick_ms,
            print_every_s=args.print_every_s,
        )


if __name__ == "__main__":
    main()
